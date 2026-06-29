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
  )

  ; Move between two adjacent locations.  Declare (adjacent from to) for every
  ; navigable pair in the problem (:init).
  (:action navigate
    :parameters (?from ?to)
    :precondition (and (robot-at ?from) (adjacent ?from ?to))
    :effect (and (not (robot-at ?from)) (robot-at ?to))
  )

  ; Grasp an object at the robot's current location.
  ; Arm-lowering and gripper mechanics are handled by the MCP tools, not modelled here.
  (:action pick-up
    :parameters (?o ?l)
    :precondition (and (robot-at ?l) (object-at ?o ?l) (gripper-empty))
    :effect (and (holding ?o) (not (gripper-empty)) (not (object-at ?o ?l)))
  )

  ; Set an object down at the robot's current location.
  (:action place
    :parameters (?o ?l)
    :precondition (and (robot-at ?l) (holding ?o))
    :effect (and (not (holding ?o)) (gripper-empty) (object-at ?o ?l))
  )
)
