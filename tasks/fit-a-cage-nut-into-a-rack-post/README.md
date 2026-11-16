# Fit a cage nut into a rack rail's marked hole: hook one lip from behind, keep the far lip in with a screwdriver

A server rack's vertical mounting rail (an EIA-310 19-inch rail of 2 mm steel with real 9.53 mm square holes, three per
rack unit, U lines and numbers on both faces) stands in a steel foot clamped to the bench, its front, where equipment
mounts, facing away from the robot. One hole, U3's top hole, is marked with a yellow paint-pen square. An M6 cage nut
lies face down on the bench behind the rail, and a small flat-blade screwdriver lies on the bench in front of it, to
the right. The instruction asks for the cage nut to be fitted into the marked hole; it does not say how.

A cage nut is a square nut held in a spring-steel cage with two springy lips on opposite sides. It goes in from the
back of the rail (the inside of the rack): its lips pass through the square hole and spring out over the front face, so
that the nut stays behind the hole for the screw of whatever is mounted there. The procedure the task checks is the
rack makers'. Oracle, SPARC M8 and SPARC M7 Servers Installation Guide, [Insert Cage Nuts Into Rail Holes][oracle]:
"Retrieve a cage nut and align it with the desired RETMA rail hole from inside the rack. Ensure that the lips of the
cage nut face the outside of the rack." "Hook one lip of the cage nut in the square rail hole." Then, with a flat-blade
screwdriver, IBM, [Installing the cage nuts in a 7014-S25 or 7014-S11 rack][ibm]: "Hold the cage nut in one hand, and
compress the cage-nut clip with a flat-blade screwdriver. With the clip compressed, push the edge of the cage nut fully
into the hole from the inside of the rack-mounting flange. Release the screwdriver pressure on the clip to lock the cage
nut into place." (Lenovo's [rack cabinet guide][lenovo] gives the same two methods.)

Why two arms, as seen in the scene: once one lip is hooked, the nut has to be held and turned flat against the rail
while the far lip is kept in line with the hole. Turned home with nothing on the far lip, that lip's end swings out
past the hole's edge and catches behind the flange: the nut jams half turned, the hooked lip is levered back out of the
hole and the nut drops out when let go. So one gripper holds the nut and turns it home, and the other holds the
screwdriver's blade in the hole along its far side, keeping the far lip in until it is through. The task is the task
file [`fit_a_cage_nut_into_a_rack_post.py`](fit_a_cage_nut_into_a_rack_post.py) for the `yam_arena` embodiment (scene,
scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it. No third-party assets are used: every body is
built from boxes and cylinders in these two files.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them or put one on a spring.
The rail (with its holes), the cage nut and its two lips, and the screwdriver are bodies built from several shapes, and
each lip turns on a sprung hinge, so `bodykit.py` adds such bodies and joints: importing it hooks `yam_arena.scene` when
the simulator starts, without changing yam_arena, and gives `score()` each body's pose.

## What is modelled and what is not

Everything is true size. The rail follows EIA-310: 9.53 mm (0.375 in) square holes, three per U (6.35, 22.225 and
38.1 mm above each U line, so 15.875 mm apart within a U and 12.7 mm across a U line), U = 44.45 mm, a 2 mm flange that
ends 7.5 mm from the hole column on the rack-opening side, and a side flange 30 mm deep on the other. The cage nut is an
M6 cage nut for 9.53 mm holes and 1.83-2.64 mm panels, 13.72 mm across its lips (RS PRO 523-137, [datasheet][rs]):
a 10 mm square nut, 4.8 mm thick, with a 5 mm bore, in a cage 10.1 x 11.2 x 6.0 mm (Panduit's [M6 cage nut
drawing][panduit] gives a 10.1 mm body), 5.5 g in all. Each lip is a 5 mm wide folded strip: an inner leg standing 4.75 mm out
of the cage's face inside the hole, a fold, and an outer leg flaring out to its end, whose back is 2.3 mm in front of
the cage's face. Each lip is a rigid body on a hinge at its root with a spring (0.02 N m/rad; it opens up to 10 deg and
closes up to 50 deg), so pressed in it passes the hole and let go it springs out over the front face; it flexes as a
whole rather than along its length, and its spring is lighter than real spring steel. The nut's M6 thread is not cut (a plain 5 mm bore; only a later screw would use it).
The screwdriver has an ISO 2380-1 0.6 x 3.5 mm tip (thin for its last 9 mm), a 3.5 mm shank 60 mm long and a fluted
handle 25 mm across, 40 g. The rail is one upright standing in a bench foot, not a whole rack.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Poses
are read from the simulation and each gripper's centre from the arm's joints. A lip has sprung out over a hole when its
end is in front of the flange's front face and 0.2 mm or more past that hole's edge. These rules set the score to 0
for the rest of the run: the nut falls onto the bench (after being lifted, its middle below 2 cm above the bench with
no gripper within 6 cm of it); a lip springs out at another hole; a lip is forced 30 deg or more open (its clip bent).
Otherwise the score is 0.5 once the nut sits in the marked hole by itself, at least one lip sprung out over the front
face, both grippers' centres 3 cm or more from it and the screwdriver's tip out of the hole (1 cm or more in front of the flange, or 3 cm or more from the nut's middle). Full score: the nut seated in the marked hole (both lips sprung out over the front face, its axis within
8 deg of the flange's normal, its face from 1.2 mm behind to 0.5 mm in front of the flange's back face, its middle
within 1.5 mm across and 3.5 mm up or down of the hole's middle), released as above, still for 1 s. `-E object_jitter=0` keeps every object
exactly where the task file puts it, so every trial starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=fit_a_cage_nut_into_a_rack_post.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the cage nut into the marked hole on the rack rail." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/cagenut-dream
```

The scripted demo, which reaches full score (the left gripper takes the nut from the bench, turns it to face the rail
and, from behind, tilts it 45 deg and hooks its +y lip through the marked hole; the right gripper slides the
screwdriver's blade into the hole along its -y side, 6.8 mm past the flange; the left turns the nut flat about its
hooked lip and presses it home while the blade keeps the far lip in line; the right draws the blade out, the far lip
springs out over the front face, and both grippers move away):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=fit_a_cage_nut_into_a_rack_post.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the cage nut into the marked hole on the rack rail." \
  --scorer task_score --max-steps 560 --store-frames --log-dir ~/yam-eval/cagenut-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/cagenut-dream/*.json; do inspect-robots video "$log"; done
```

`CAGE_DEBUG=1` prints, every 10 steps (`CAGE_DEBUG_EVERY=5` for every 5), where the nut's face is and how it is turned,
each lip's opening angle, where each lip's end is (in front of the front face, past the hole's edge, above or below the
hole's middle), where each gripper and the screwdriver's tip are, and the stage reached.

`CAGE_ONE=1` runs the demo the wrong way, with one hand doing the work: the right gripper brings the screwdriver but its
blade stays out of the hole, and the left turns the hooked nut home alone. The far lip's end swings out past the hole's
edge and catches behind the flange; the nut jams half turned, its hooked lip is levered back out of the hole, and when
the left gripper lets go the nut drops out of the hole. The run scores 0.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine: about 2 minutes for the scripted demo while rendering 1280 x 720
video (508 steps, 25.4 s of simulated time; 127 s), and about 6.5 minutes for a Dream-1 trial of 2,400 steps (395 s on
the clock, with Dream answering in 0.88 s on average). Isaac Sim's first start on a new machine adds a few minutes while
it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

A rack's vertical mounting rail (any EIA-310 square-hole rail, 2 mm steel) held upright in a bench vise or bolted to an
angle, a hole marked with a paint pen, an M6 cage nut for 9.5 mm holes and 1.8-2.6 mm panels, and a small flat-blade
screwdriver. A trial is scored by looking at the rail's front (both lips sprung out round the marked hole), checking
that the nut sits flat against the back and that a light tug does not pull it out, and by the overhead and wrist videos
(whether the nut ever fell, and whether a lip was bent).

[oracle]: https://docs.oracle.com/cd/E55211_01/html/E55213/grbue.html
[ibm]: https://www.ibm.com/docs/POWER5/iphbf_p5/cagenuts.htm
[lenovo]: https://pubs.lenovo.com/hdc_rackcabinet/install_nuts_48u
[rs]: https://au.rs-online.com/web/p/cage-nuts/0523137
[panduit]: https://media.masterelectronics.com/productfiles/mf-pn/c-rkcnwsm6-c--eng.pdf
