(define (domain lego-robot)
  ; :typing is intentionally omitted — pyperplan 2.x has a bug with typed objects.
  ; Predicate structure constrains the search sufficiently without explicit types.
  (:requirements :strips)

  ; Robot direction is managed by other tools, and is NOT relevant for high-level planning
  (:predicates
    (robot-at ?l)
    (object-at ?o ?l)
    (holding ?o)
    (gripper-empty)
    (adjacent ?l1 ?l2)
    (arm-lowered)
    (arm-raised)
    (gripper-open)
    (is-in-grasp-pose)
  )

  ; Move to approach an object to grasp.  Requires (is-in-grasp-pose) rather
  ; than the raw arm-lowered/gripper-open facts directly, so the robot is
  ; grasp-ready for the whole approach instead of opening/lowering only
  ; after arriving next to the target. Declare (adjacent from to) for every
  ; navigable pair in the problem (:init).
  ; Convention: is-in-grasp-pose is asserted ONLY by open-gripper (see
  ; below) and MUST be retracted — (not (is-in-grasp-pose)) added to
  ; :effect — by any action that disturbs arm or gripper posture (raising
  ; the arm, closing/moving the gripper for a non-grasp purpose, etc. — see
  ; grasp/lift-arm for the pattern). That's what forces the planner to
  ; re-run lower-arm+open-gripper immediately before the next
  ; navigate/grasp, even when some unrelated action (e.g. a future
  ; button-press action) gets inserted in between and leaves the arm/
  ; gripper in a different physical posture. A new action that skips this
  ; convention will silently let plans skip re-prepping — same failure
  ; mode as the toggle-lights/click_button case that motivated this.
  ; Plan and execute a path that avoids obstacles, reverse and rotates when necessary.
  (:action navigate
    :parameters (?from ?to)
    :precondition (and (robot-at ?from) (adjacent ?from ?to)
                       (is-in-grasp-pose))
    :effect (and (not (robot-at ?from)) (robot-at ?to))
  )

  ; Move while already holding an object — the transport leg after grasp
  ; and before place. The gripper is necessarily closed around the held
  ; object here, so (unlike `navigate` above) this does not require
  ; gripper-open/arm-lowered.
  (:action navigate-holding
    :parameters (?from ?to ?o)
    :precondition (and (robot-at ?from) (adjacent ?from ?to) (holding ?o))
    :effect (and (not (robot-at ?from)) (robot-at ?to))
  )

  ; Open the gripper.  Requires the arm to already be lowered (enforces
  ; "lower arm, then open gripper") and the gripper to be empty — the latter
  ; also stops the planner from "reopening" the gripper while an object is
  ; held just to satisfy navigate's precondition again, which would drop it.
  ; The ONLY action that asserts (is-in-grasp-pose) — see convention note
  ; on navigate above.
  ; Convention: never assert (gripper-open) or (is-in-grasp-pose) in (:init),
  ; so the planner always generates this step before grasp regardless of
  ; observed state.
  (:action open-gripper
    :parameters ()
    :precondition (and (arm-lowered) (gripper-empty))
    :effect (and (gripper-open) (is-in-grasp-pose))
  )

  ; Lower the arm to ground level.  No precondition — safe to call even if
  ; already lowered.  Convention: never assert (arm-lowered) in (:init), so
  ; the planner always generates this step before grasp regardless of
  ; observed state.
  (:action lower-arm
    :parameters ()
    :precondition (and)
    :effect (and (arm-lowered) (not (arm-raised)))
  )

  ; Raising the arm ends the grasp pose — see convention note on navigate.
  (:action lift-arm
    :parameters (?o)
    :precondition (and (holding ?o) (arm-lowered))
    :effect (and (arm-raised) (not (arm-lowered)) (not (is-in-grasp-pose)))
  )

  ; Grasp an object at the robot's current location.
  ; Requires (is-in-grasp-pose) — arm lowered + gripper open, freshly
  ; asserted by open-gripper (see convention note on navigate).
  ; Closes the gripper, which ends the grasp pose.
  (:action grasp
    :parameters (?o ?l)
    :precondition (and (robot-at ?l) (object-at ?o ?l) (gripper-empty)
                       (is-in-grasp-pose))
    :effect (and (holding ?o) (not (gripper-empty)) (not (object-at ?o ?l))
                 (not (gripper-open)) (not (is-in-grasp-pose)))
  )

  ; Set an object down at the robot's current location (calls put MCP tool,
  ; which opens the gripper and raises the arm).
  (:action place
    :parameters (?o ?l)
    :precondition (and (robot-at ?l) (holding ?o))
    :effect (and (object-at ?o ?l) (not (holding ?o)) (gripper-empty)
                 (gripper-open) (not (arm-lowered)))
  )
)
