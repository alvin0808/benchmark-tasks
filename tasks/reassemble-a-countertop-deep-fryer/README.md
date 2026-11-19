# Put a washed countertop deep fryer back together: pot into the base, control assembly down its two tracks

A countertop immersion-element deep fryer (built after the Presto Stainless Steel ProFry, 3 qt of oil) has been taken
apart for washing, as its manual says to do after every use. Its stainless base stands on the table with its back
toward the robot; the washed, empty enamelled pot stands on the table in front of the robot; the control assembly
(the control unit with its thermostat knob and signal light, and the heating element that hangs from it) stands on the
table to the robot's left. The instruction asks for the fryer to be put back together so that it is ready to use; it
does not say how.

What the task tests, all of it visible in the scene:

- The order is forced by the parts. The element hangs 11 cm down inside the pot and runs along its floor, so once
  the control assembly is on the base the pot can no longer go in under it: the pot goes in first.
- The pot needs both hands. It is 27 cm long, wider than a gripper opens, and fits its base with 3 mm to spare all
  round, so it has to go down level; the manual lifts it "grasping two opposite corners". Pinched by one end wall it
  tips, slides round in the fingers and falls.
- The control assembly is seated by sliding it straight down two tracks on the back of the base. It is 0.75 kg and
  14.5 cm tall with a 12 x 11 cm element loop at its foot; it has to be lifted until the loop is above the pot's rim,
  carried over the back of the base, lowered with the loop going into the pot without knocking it (the element's
  tubes pass 1.2 to 1.6 cm inside the pot's back wall), and its two guides (5 mm wide) brought into the two tracks (8 mm
  channels with a flared mouth) and slid down until they stand on the stops at the tracks' feet. That is the fryer's
  interlock: "The appliance will only operate when both guides are properly positioned in the tracks."

The task is the task file
[`reassemble_a_countertop_deep_fryer.py`](reassemble_a_countertop_deep_fryer.py) for the `yam_arena` embodiment
(scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Why it matters

Countertop deep fryers are a common kitchen appliance, and taking one apart to wash it and putting it back together is
part of every use: the manuals tell owners to lift out the control assembly and the pot after frying, wash the pot and
basket, and reassemble before the next use. Anyone watching a clip knows what is being done.

The procedure modelled is the one in the Presto Stainless Steel ProFry (model 05461) instructions
([gopresto.com, 05461 instructions][presto]), "How To Use for Frying": "2. Place removable pot in base. 3. Attach the
control assembly to the base by aligning the two guides on the front of the control unit with the two tracks on the
back of the base (Fig. F). Slide the control assembly down (Fig. G) as far as it goes making sure both guides are in
the tracks. Note: The appliance will only operate when both guides are properly positioned in the tracks." Care and
Cleaning: "After unit and oil have cooled completely, carefully lift control assembly out of fryer ... Then, grasping
two opposite corners of the removable pot, carefully lift the pot out of the deep fryer." Troubleshooting ("If unit
fails to preheat"): "Check to see that both guides on the front of the control unit are properly positioned in the
tracks on the back of the base." The same seating rule appears in other makers' manuals (Westinghouse WST3034: "The
Deep Fryer will not operate if the Control Panel/Heater Assembly is not properly seated"; Cuisinart CDF-200: "The
immersion heating element will not heat up unless the control panel is correctly installed").

This is not the plug-in coil element of an electric range (a burner coil whose two terminals push sideways into a
receptacle under the cooktop): here a pot must go in first, the control box carries an interlock that closes only
when it is slid fully down two tracks, and the element swings down into a deep pot that it must not strike.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library and uses no
third-party assets: the base, the pot and the control assembly are built in these files from measurements.

`yam_arena` task files place single boxes, cylinders and spheres. The base (walls, handles, feet, the two tracks with
their rails, flared mouths and stops), the pot (drawn walls and a flange) and the control assembly (control unit,
head, guides and the element's tubes) are each one rigid body built from several shapes with `bodykit.py`, which
hooks `yam_arena.scene` when the simulator starts, without changing yam_arena. This copy of `bodykit.py` also lets a
body set its centre of mass and turn on PhysX contact reporting, so the task file reads the contact force between the
control assembly and the pot and between the control assembly and the base.

## The scene

Positions are in cm from the midpoint between the two arm bases: x forward (away from the arms), y to the robot's
left, z up from the table top. The layout is the same in every trial.

- Fryer base, brushed stainless, fixed to the table (a fryer stands on its feet and does not slide under these
  forces): outer 20.5 cm (x) x 30.0 cm (y), its rim 16.0 cm above the table, walls 1.2 cm, on four 1 cm feet; x 30.0 to
  50.5, y -15.0 to 15.0, its back (x 30.0) toward the robot; black handles on both ends (y +-15.0 to +-17.6). On the
  outside of the back wall, two vertical tracks at y +2.5 and -2.5: each an 8 mm channel between two rails (5 mm wide,
  standing 6 mm off the wall), from 6.5 to 15.0 cm above the table, the rails' top 1 cm leaning out at 45 deg as a
  funnel, closed at the foot by a red stop. The back wall's top outer edge is chamfered 7 mm.
- Pot, enamelled steel, loose, 0.65 kg: 17.5 x 27.0 cm at the top, drawn to 16.5 x 26.0 cm at the bottom, 12.5 cm
  deep, walls 4 mm, with a 9 mm flange round the top; MIN and MAX marks inside the back wall. It starts on the table
  centred at x 13.5, y 0. Seated, it hangs on its flange on the base's rim, centred at x 40.25, y 0, with 3 mm between
  its walls and the base's walls.
- Control assembly, loose, 0.75 kg (centre of mass just in front of the guides): a black control unit 5.8 cm (x) x
  7.0 cm (y) x 11.6 cm, and above it a head 11.2 x 7.0 x 2.9 cm that reaches 4.6 cm forward over the rim, with the
  thermostat knob and the signal light on top; on the unit's front face two guides (5 mm wide, 7.5 mm deep, 5 cm long,
  their lowest 1 cm narrowed to 3 mm); from the head's underside two 9 mm element tubes run down and join a flat loop
  (12.4 x 10.9 cm outside, 9 mm tube) that lies 8 mm above the pot's floor when seated. It starts standing upright on
  the table on its loop and the foot of its unit, the element pointing forward, the unit's back face at x 20.0,
  centred at y 26.0.
- Both arms start at the ready pose, grippers open.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. Every
control step the task file reads the pot's and the control assembly's poses, the contact forces between the control
assembly and the pot and between the control assembly and the base (PhysX pair-wise contact forces), and the points of
each finger's collision hulls.

- 0.25: the pot seated in the base: its flange on the rim (within 2 mm of its seated height), within 3 mm of its place
  across, tilted less than 2 deg.
- 0.5: and both guides of the control assembly in their tracks (each guide's foot inside its channel, below the rails'
  tops).
- 0.75: and the control assembly fully down: both guides within 2 mm of their stops and the assembly resting on the
  base with 3 N or more (its weight is 7.4 N): the interlock closed; with the whole element loop inside the pot.
- 1.0 (success, held 1 s): all of that, with both grippers let go and clear (no finger point within 1 cm of the pot or
  the control assembly) and both still (moving less than 1 mm over 1 s).
- Caps, for the rest of the run: if the control assembly ever pressed on the pot with more than 5 N (the element
  knocking the pot's rim or floor, or the assembly dropped onto it), the score is at most 0.75; if the control
  assembly was dropped (falling faster than 0.7 m/s) or tipped over (more than 45 deg), at most 0.5.

Doing nothing scores 0; the control assembly put on before the pot scores 0 (the pot cannot then be seated).

A real trial is scored from the fryer itself and the video: the pot's flange lies flat on the rim all round (a photo,
no gap at any corner); both guides are in their tracks and down on the stops (a photo of the back); the element loop
is inside the pot. The fryer's own interlock gives the last check: plug it in and turn the thermostat up for a moment
(the manual: "While oil is heating, the signal light will be illuminated"); the signal light comes on only when both
guides are fully down in their tracks. Turn it straight back down and unplug, since an empty pot must not be heated.
A knock of the element against the pot (heard, or seen to move the pot) caps the score at 0.75, a dropped or tipped
assembly at 0.5, both judged from the video.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=reassemble_a_countertop_deep_fryer.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Put the deep fryer back together after washing so that it is ready to use." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/fryer-dream
```

Our Dream-1 run scored 0.0: the right gripper pushed the pot across the table and tipped it, the left gripper hovered
near the control assembly, and nothing was put into the base.

The scripted demo, which reaches full score (open loop from the known layout): both arms take the pot by its two end
walls (grips pitched 45 deg down, fingers on either side of each wall), lift it, carry it up and forward over the
base's back wall and lower it level into the base, aimed 1 cm past its seat (the pot slips back a little in the
fingers and the arms lag under load). The fingers outside the end walls cannot follow the pot below the base's rim, so
they let go with its flange 4 cm above the rim and the pot drops the rest of the way, its drawn walls centring it. The
fingers move out from the walls and up, and the right arm moves aside. The left arm takes the control assembly by the
sides of its head (above the guides, close to its centre of mass), lifts it until the element loop is about 2 cm above
the pot's rim (the arm sags about 3 cm under the load at that height), tips the hand back 6.5 deg so the assembly
hangs level, carries it over the base's back, lowers it straight down until the guides are 8 mm above the tracks'
mouths (the loop is then inside the pot), then slides it down the tracks 6 mm past the seat (the guides stop on the
stops), lets go, and draws the fingers back along themselves and up. It is aimed 9 mm forward of and 4 mm left of the
seat, because the loaded arm stops short and the unit sits slightly off-centre in the grip; the tracks' flared mouths
take up the rest.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=reassemble_a_countertop_deep_fryer.py -E object_jitter=0 -E control_hz=20 \
  --instruction "scripted demo" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/fryer-demo
```

Videos: `inspect-robots video <log>.json` in the log folder.

`FRYER_DEBUG=1` prints a `DBG` line every 5 control steps (`FRYER_DEBUG_EVERY=n` to change): the pot's offset from
its seat, the guides' feet, whether they are engaged and down, the contact forces with the pot and the base, the
caps and both TCPs.

Wrong ways, each scoring 0 for a physical reason:

- `FRYER_WRONG=ctrl_first`: the control assembly is put on first, then both arms lower the pot as in the demo. The
  control assembly seats (its guides down on the stops), but its head and element now fill the back of the base:
  carried forward, the pot hits them, is knocked out of the fingers and falls back onto the table (the element's
  contact with the pot peaks at 83 N). Score 0 (the pot is never seated).
- `FRYER_WRONG=one_arm`: the right arm alone carries the pot by its right end wall along the demo's path. Pinched by
  one end wall, the pot swings and tips as it is lifted, slides round in the fingers and falls onto the table beside
  the base. Score 0.

## Runtime per trial

On one NVIDIA RTX PRO 6000 Blackwell shared with other simulation jobs, the scripted demo while rendering 1280 x 720
video took 203 s (1,062 steps, 53.1 s of simulated time, ending on success); a Dream-1 trial of 2,400 steps at
320 x 176 took 388 s (0.16 s per step, 1.2 s per policy call). Debug runs of the demo without video took 3 to 5
minutes. Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## What is modelled and what is not

Rigid bodies, 20 Hz control. The base is fixed to the table. The pot's walls are 4 mm (a real enamelled pot is about
1 mm steel) so that they collide reliably. The control unit, its head and the element are one rigid body; the
element's round tubes are cylinders and the loop is drawn as a rectangle of four tubes (a real element is one bent
tube with rounded corners). The interlock switch is not modelled as a part: the scorer reads it as both guides standing
on their stops with the assembly's weight on the base, which is where the guides sit when the real switch closes. No
oil, no basket, no lid, no cord. Masses and dimensions are estimates for this class of fryer.

## Building it for real

A Presto Stainless Steel ProFry 05461 (about 60 USD) or any immersion-element fryer whose control assembly slides down
tracks on the back of its base; washed, dry and cold, the cord unplugged. Set the base on the table, back toward the
robot, its back wall's outer face 30 cm in front of the line between the arm bases and centred between them; stand the
pot on the table centred 13.5 cm in front of that line; stand the control assembly on the table on its element loop,
element pointing away from the robot, its back face 20 cm in front of the line and 26 cm to the robot's left. To reset,
lift the control assembly out (straight up the tracks) and stand it back, then lift the pot out by two opposite
corners and stand it back.

[presto]: https://www.gopresto.com/uploads/downloads/instructions/05461.pdf
