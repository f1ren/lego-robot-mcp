(define (domain lego-robot)
  ; :typing is intentionally omitted — pyperplan 2.x has a bug with typed objects.
  ; Predicate structure constrains the search sufficiently without explicit types.
  (:requirements :strips)

  (:predicates
    (robot-at ?l)
    (object-at ?o ?l)
    (holding ?o)
    (gripper-empty)
    (adjacent ?l1 ?l2)
    (arm-lowered)
    (gripper-open)
  )

  ; Move between two adjacent locations.  Declare (adjacent from to) for every
  ; navigable pair in the problem (:init).
  (:action navigate
    :parameters (?from ?to)
    :precondition (and (robot-at ?from) (adjacent ?from ?to))
    :effect (and (not (robot-at ?from)) (robot-at ?to))
  )

  ; Open the gripper.  No precondition — safe to call even if already open.
  ; Convention: never assert (gripper-open) in (:init), so the planner always
  ; generates this step before pick-up regardless of observed state.
  (:action open-gripper
    :parameters ()
    :precondition (and)
    :effect (gripper-open)
  )

  ; Lower the arm to ground level.  No precondition — safe to call even if
  ; already lowered.  Convention: never assert (arm-lowered) in (:init), so
  ; the planner always generates this step before pick-up regardless of
  ; observed state.
  (:action lower-arm
    :parameters ()
    :precondition (and)
    :effect (arm-lowered)
  )

  ; Grasp an object at the robot's current location.
  ; Requires the gripper to be open and the arm to be lowered first.
  ; Closes the gripper (not (gripper-open)); arm stays lowered.
  (:action pick-up
    :parameters (?o ?l)
    :precondition (and (robot-at ?l) (object-at ?o ?l) (gripper-empty)
                       (gripper-open) (arm-lowered))
    :effect (and (holding ?o) (not (gripper-empty)) (not (object-at ?o ?l))
                 (not (gripper-open)))
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
