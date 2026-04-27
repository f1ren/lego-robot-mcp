"""
Persistent SSH client for executing Python code on the Raspberry Pi.

Usage:
    client = RPiClient("rpi.local", "rpi")
    result = client.run_python(\"\"\"
        import json
        from buildhat import Motor
        m = Motor('A')
        print(json.dumps({'pos': m.get_position()}))
    \"\"\")
    # result == {'pos': 9}

All scripts must write a single JSON object to stdout.
stderr (libcamera logs, etc.) is suppressed on the RPi side.
"""
import json
import threading
import textwrap
import paramiko

from mcp_robot import config


class RPiClient:
    def __init__(self, host: str = config.RPI_HOST, user: str = config.RPI_USER):
        self.host = host
        self.user = user
        self._ssh: paramiko.SSHClient | None = None
        self._lock = threading.Lock()

    # ── connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.host,
            username=self.user,
            timeout=config.SSH_TIMEOUT,
            look_for_keys=True,
            allow_agent=True,
        )
        # SSH_TIMEOUT is for the TCP handshake only. Clear it so slow RPi
        # commands (e.g. BuildHat init) don't trigger a socket timeout.
        client.get_transport().sock.settimeout(None)
        self._ssh = client

    def _ensure_connected(self) -> paramiko.SSHClient:
        transport = self._ssh.get_transport() if self._ssh else None
        if transport is None or not transport.is_active():
            self._connect()
        return self._ssh  # type: ignore[return-value]

    # ── execution ─────────────────────────────────────────────────────────────

    def run_python(self, script: str, timeout: int = 30) -> dict:
        """
        Execute *script* on the RPi via SSH.

        The script must print exactly one JSON-serialisable object to stdout.
        libcamera / BuildHat noise on stderr is discarded.
        Raises RuntimeError if no JSON output is received.
        """
        # Wrap the script so any uncaught exception is emitted as JSON to stdout
        # (otherwise it would silently vanish if stderr is suppressed).
        # We also redirect C-level fd-2 so libcamera INFO noise doesn't corrupt output.
        wrapper = textwrap.dedent("""\
            import sys, os, json as _json, traceback as _tb
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)  # silence libcamera C logs
            try:
        """)
        indented = textwrap.indent(textwrap.dedent(script), "    ")
        footer = textwrap.dedent("""\
            except Exception as _e:
                print(_json.dumps({"__error__": str(_e), "__trace__": _tb.format_exc()}))
        """)
        full_script = wrapper + indented + "\n" + footer

        with self._lock:
            ssh = self._ensure_connected()
            stdin, stdout, stderr = ssh.exec_command("python3 -", timeout=timeout)
            stdin.write(full_script.encode())
            stdin.channel.shutdown_write()

            try:
                raw = stdout.read().decode().strip()
            except TimeoutError as exc:
                raise RuntimeError(
                    "RPi script timed out — BuildHAT may be missing or unresponsive"
                ) from exc
            if not raw:
                err = stderr.read().decode().strip()
                raise RuntimeError(
                    f"RPi script produced no output.\nstderr: {err}\n"
                    f"script:\n{full_script}"
                )
            try:
                result = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"RPi script output is not valid JSON: {raw!r}"
                ) from exc
            if "__error__" in result:
                raise RuntimeError(
                    f"RPi script raised: {result['__error__']}\n{result.get('__trace__', '')}"
                )
            return result

    def stream_python(
        self,
        script: str,
        on_line,
        stop_event: "threading.Event | None" = None,
    ) -> None:
        """
        Run *script* on the RPi over a *separate* SSH connection, calling
        on_line(dict) for each newline-delimited JSON object it prints.

        Blocks until the remote process exits or stop_event is set.
        Uses its own connection so it never contends with run_python's lock.
        """
        # Fresh connection — don't share with the command singleton.
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            self.host,
            username=self.user,
            timeout=config.SSH_TIMEOUT,
            look_for_keys=True,
            allow_agent=True,
        )
        transport = ssh.get_transport()
        # Clear the connect-phase timeout so slow camera init doesn't abort mid-stream.
        transport.sock.settimeout(None)
        # Keepalive so the SSH server detects a dead client within ~30 s instead of hours.
        transport.set_keepalive(15)

        # Release any stale camera session left by a previous crashed stream.
        _, _so, _ = ssh.exec_command(
            "fuser -k -TERM /dev/video0 /dev/video1 /dev/media0 /dev/media1 /dev/media2"
            " 2>/dev/null; sleep 0.4"
        )
        _so.channel.recv_exit_status()  # wait for cleanup to finish

        preamble = "import os; os.dup2(os.open('/dev/null', os.O_WRONLY), 2)\n"
        try:
            stdin, stdout, _ = ssh.exec_command("python3 -", timeout=None)
            stdin.write((preamble + textwrap.dedent(script)).encode())
            stdin.channel.shutdown_write()
            frame_count = 0
            for raw in stdout:
                if stop_event and stop_event.is_set():
                    break
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    on_line(json.loads(raw))
                    frame_count += 1
                except (json.JSONDecodeError, Exception):
                    pass
            if frame_count == 0:
                raise RuntimeError(
                    "RPi stream script exited without producing any frames. "
                    "Check camera availability and that picamera2/Pillow are installed on the RPi."
                )
        finally:
            ssh.close()

    def close(self) -> None:
        if self._ssh:
            self._ssh.close()
            self._ssh = None


# Module-level singleton — shared across all tool calls.
_client: RPiClient | None = None


def get_client() -> RPiClient:
    global _client
    if _client is None:
        _client = RPiClient()
    return _client
