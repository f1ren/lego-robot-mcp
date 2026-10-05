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
import logging
import threading
import textwrap
import paramiko

from mcp_robot import config

log = logging.getLogger(__name__)


_HAT_RESET_CMD = (
    "sudo lsof -t /dev/ttyS0 /dev/serial0 2>/dev/null | xargs -r sudo kill -9; "
    "sleep 0.3; "
    "gpioset gpiochip0 4=0 && sleep 0.1 && gpioset gpiochip0 4=1 && sleep 0.5"
)


# Prepended to every script while any motor is held (see RPiClient.hold_motor).
# Each run_python script is its own process with its own buildhat session,
# and buildhat's exit path cuts power to every port: BuildHAT.shutdown (a
# weakref.finalize/atexit hook) sends "pwm ; coast ; off" to all four ports,
# and Device.__del__ sends "off" for each Motor the script created. Left
# alone, any unrelated call (drive, navigate_to, get_robot_state...) would
# therefore drop a held gripper's object. The patch skips __del__'s "off" for
# held ports, and re-sends their open-loop PWM right after shutdown's
# all-port coast — a few ms gap, rather than filtering shutdown's command
# string, so it doesn't depend on that string's exact format.
# Open-loop PWM (not a PID position hold) because PID feedback comes from the
# port's selected sensor mode, which every new script re-selects on init.
_HOLD_PREAMBLE = """\
import buildhat.serinterface as _bh_si, buildhat.devices as _bh_dev
_BH_HELD = {held!r}  # BuildHAT port index -> PWM to keep applied
_bh_orig_shutdown = _bh_si.BuildHAT.shutdown
def _bh_shutdown(self):
    _bh_orig_shutdown(self)
    for _p, _v in _BH_HELD.items():
        self.write(f"port {{_p}} ; pwm ; set {{_v}}\\r".encode())
_bh_si.BuildHAT.shutdown = _bh_shutdown
_bh_orig_del = _bh_dev.Device.__del__
def _bh_del(self):
    if getattr(self, "port", None) not in _BH_HELD:
        _bh_orig_del(self)
_bh_dev.Device.__del__ = _bh_del
"""


class RPiClient:
    def __init__(self, host: str = config.RPI_HOST, user: str = config.RPI_USER):
        self.host = host
        self.user = user
        self._ssh: paramiko.SSHClient | None = None
        self._lock = threading.Lock()
        # BuildHAT port letter -> PWM (-1..1) kept applied across scripts.
        self._held: dict[str, float] = {}

    # ── motor hold across scripts ────────────────────────────────────────────

    def hold_motor(self, port: str, pwm: float) -> None:
        """Keep *port* driven at open-loop *pwm* after every later script
        exits, until release_motor(port). Only affects scripts run after this
        call; the caller is responsible for actually starting the PWM."""
        self._held[port] = pwm

    def release_motor(self, port: str) -> None:
        """Stop re-asserting *port*'s hold. Doesn't itself touch the motor —
        the next script's normal exit coasts it."""
        self._held.pop(port, None)

    def is_held(self, port: str) -> bool:
        return port in self._held

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

        Recovers automatically from "HAT not found": runs a kill+GPIO-reset
        sequence on the RPi and retries once.
        """
        try:
            return self._run_python_once(script, timeout)
        except RuntimeError as exc:
            msg = str(exc)
            if "HAT not found" not in msg and "BuildHAT may be missing" not in msg:
                raise
            log.warning("BuildHAT unresponsive (%s) — running reset and retrying once.", msg.splitlines()[0])
            self._reset_hat()
            return self._run_python_once(script, timeout)

    def _build_script(self, script: str) -> str:
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
            finally:
                try:
                    import buildhat as _bh; _bh.Hat().deinit()
                except Exception:
                    pass
        """)
        if self._held:
            held = {ord(p) - ord("A"): v for p, v in self._held.items()}
            wrapper += textwrap.indent(_HOLD_PREAMBLE.format(held=held), "    ")
        return wrapper + indented + "\n" + footer

    def _run_python_once(self, script: str, timeout: int) -> dict:
        full_script = self._build_script(script)
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

    def _reset_hat(self) -> None:
        """
        Recover from "HAT not found": kill any process holding the serial port,
        then pulse GPIO 4 low/high to reset the BuildHAT.
        """
        with self._lock:
            ssh = self._ensure_connected()
            _, stdout, stderr = ssh.exec_command(_HAT_RESET_CMD, timeout=15)
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                err = stderr.read().decode().strip()
                log.warning("HAT reset exited with status %d: %s", exit_code, err)
            else:
                log.info("HAT reset completed.")

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
