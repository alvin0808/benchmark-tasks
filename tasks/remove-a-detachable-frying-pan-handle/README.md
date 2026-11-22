# Squeeze both side releases to remove a frying pan's detachable handle

The pan stays on the counter while the handle comes off. Squeeze the two opposed release buttons together with the
lower lever free, then lift the opened handle away from the rim and put it beside the pan. This is the release
procedure in the [Tefal Ingenio cookware handle instructions][handle]. The grip stays secured if only one release
is pressed. Separating the handle makes the pan ready for stacking.

## Setup

A cool 206 mm rolled-rim frying pan sits on the horizontal counter. The aluminum pan is 53 mm high and 530 g,
with a 160 mm diameter foot and a dark non-stick interior. Its mouth is open and empty. The removable grip is
about 198 mm long and 210 g, with a molded polymer body, steel jaws, two red 16.5 × 14 mm side buttons, and a
spring-return lever underneath. The two buttons are opposite each other 49 mm behind the jaw. The handle is
already locked around the near rim, pointing back toward the robot. The rim and the pan both bear the normal load.

Coordinates in metres are relative to the midpoint between the robot bases, with x away from the robot and positive
y toward the left arm. The pan centre starts at (0.438, 0.091), the handle jaw starts at (0.335, 0.091), and the
bench is horizontal. Both grippers start at their ready pose. For reset, fit the handle over the same rim and
lock the lower lever, with both buttons sprung outward.

## Real trial and scoring

Use overhead and side video, a ruler and an angle gauge. The pan must stay supported until the grip is clear.
The two buttons must overlap in time at least 0.10 s with at least 3.5 mm of travel on each. If the pan lifts over
7 mm or tilts over 9° before separation, the trial earns zero.

The lower lever opening at least 19° with at least 10 mm of jaw retraction earns 0.25. Clearing the pan by at
least 45 mm with the handle separated earns 0.65. Full score requires both pieces at rest, supported by the
counter and both hands clear for 1.5 s. In simulation, the closest handle underside is within 4 mm of the bench and
the hands are at least 65 mm from the handle centreline and 36 mm from the pan envelope. The handle and pan remain
loose on the counter. The lower lever can rest against the counter once the detached grip is placed down.

The buttons, lower lever, and moving steel jaw have separate joints. The gripper contacts the buttons directly.
Simultaneous inward travel releases the internal catch, so springs drive the lever and jaw. The steel jaw's
undercut and the continuous ring of pan rim colliders carry contact loads. The small linkages inside the grip
are represented by the latch and spring drives.

## Reproduce

Install [yam-isaac-lab-arena][arena] on a GPU. All parts are built from the task files. The trial uses a fixed reset
with no random seed required.

```bash
task=remove_a_detachable_frying_pan_handle.py
instruction="Squeeze both side buttons to release the handle while the pan stays on the counter. Let the lower lever open, lift the handle off the rim, and set it down beside the pan."

inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task="$task" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames --log-dir ./handle-demo

inspect-robots run --embodiment yam_arena --policy dream \
  -E task="$task" -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$instruction" --scorer task_score --max-steps 1200 --store-frames --log-dir ./handle-dream
```

`PAN_HANDLE_SINGLE=1` selects a control that presses only one button. `PAN_HANDLE_DEBUG=1` logs the two button
positions, jaw and lever travel, pan lift and scorer state. To render the recordings, run `inspect-robots video`
on the result JSON. A scripted trial takes about a minute on an RTX PRO 6000. Dream-1 uses configured credentials
and runs up to 60 simulated seconds.

[handle]: https://www.tefal.com/medias/2010665625-IFU-handle-Ingenio-TEF-fr-en-nl-de-es-pl.pdf?context=bWFzdGVyfHJvb3R8MTgzMTYzMHxhcHBsaWNhdGlvbi9wZGZ8aDUzL2gzYy8xNDQ1NTU3MDAzODgxNC5wZGZ8ODRhMjQyMmE3MjUwMzdkZjgwMmVhZWIxZjkwYTkzMGI0YzE3OWFiMDVlMWFjM2NlYTY1NzI3Y2Q2ZjNiZGM4MQ
[arena]: https://github.com/kairosiann/yam-isaac-lab-arena
