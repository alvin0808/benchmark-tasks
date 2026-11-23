# Hook a hanging travel scale onto a bag and hold it steady

The robot raises the hinged carry handle, threads the open metal hook under it, lifts the whole bag and holds the scale until the reading locks. It then lowers the bag to the bench and hangs the unloaded scale back on its peg so the locked weight can be read. The scale measures the load while it hangs freely. An unstable swing or a hand supporting the bag can change the reading.

## Physical trial

Use a handheld digital hanging scale with a metal hook and a lock mode. The [Travel Smart TS601X instructions](https://www.conair.com/on/demandware.static/-/Library-Sites-usConairShared/default/dw85d8da44/Instructional%20Booklets/TS601X.pdf) document the open-hook procedure and the stabilizing readout. Use a 23 × 18 × 11 cm travel or electronics case with a hinged top carry handle. Pack and weigh the case to 642 g within 15 g on a reference bench scale. The initial carry handle folds toward the robot's left. Its open clear interior is about 110 mm wide and 34 mm high. Use a standard rigid peg on a bench stand, Ø8 mm with 83 mm of exposed reach along the y axis. Set its center 27.6 cm above the table and centered at x 35.4 cm as in `task.json`. Hang the already zeroed scale by its top loop, with the peg pointing toward the bag.

Film the bag's feet, the scale face and both hands. A 25 mm vertical mark on a small ruler behind the bag makes clearance visible. When it is entirely clear and the hook alone supports it, score 0.25. Score 0.6 if the lock reading falls within 85 g of the 642 g reference. Gently return the bag to the bench, hang the unloaded scale back on the peg with the reading locked, and move both hands clear. The full outcome holds for one second. A drop after lifting zeros the run. For reset, fold the handle, hang the scale, confirm lock mode and tare at zero without a load. The nearest earlier scale task [#77](https://github.com/robocurve/benchmark-tasks/pull/77) tests finding the mass of an unknown block with an uncalibrated spring. This task tests hooking the real carry handle and letting a freely suspended load stabilize enough for the commercial lock mode. [RoboCasa's ScalePortioning](https://robocasa.ai/docs/build/html/tasks/composite_tasks.html) puts an item onto a fixed countertop scale and waits. Here the gripper supports the entire freely swinging load through the captured hook while the reading settles.

## Simulation

Gravity loads an extensible contact-driven hook. The force on the hook is computed from the deflection of a load-cell spring. When the bag is clear, the slide stays stable for at least 43 steps, and the scale is upright, its display latches the median total mass. Scale and bag geometry are generated from code. Flexible webbing is represented by the hinged carry handle. A real scale's internal electronics and damping are approximated.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=weigh_a_travel_bag_with_a_hanging_scale.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Weigh the blue travel bag with the hanging scale. Hook it by the carry handle, lift the bag clear, and wait until the weight locks on the display. Lower the bag gently onto the bench. Return the unloaded scale to its peg with the reading still locked." \
  --scorer task_score --max-steps 1200 --store-frames
```

Use `--policy dream` for the independent model run on the same fixed layout. The run limit is 60 simulated seconds. The full scripted motion takes around 23 simulated seconds, with RTX rendering taking a few wall-clock minutes. `BAGSCALE_UNSTEADY=1` lowers the bag immediately after lifting, before the readout can lock.
