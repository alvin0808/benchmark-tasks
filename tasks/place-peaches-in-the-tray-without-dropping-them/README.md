# Place the peaches in the packing tray, never dropping them

The table top stands for a packing table at a fruit farm. Four ripe peaches, just picked, rest in foam rings, two on
each side, and a cardboard packing tray with four cups lies between them. The instruction asks for the peaches to be
packed into the tray; it does not say how.

The rule the task checks is the one fruit growers train their pickers and packers in: a ripe peach bruises when it is
dropped, even a short way, so fruit is placed, not dropped. University of Maryland Extension, [Maximizing Apple and
Peach Profits: Preventative Bruising
Practices](https://extension.umd.edu/resource/maximizing-apple-and-peach-profits-preventative-bruising-practices):
peaches are placed "gently into the basket", and apples handled "like an egg". The quick way is to let each peach go
over its cup and let it fall in, which breaks the rule; lowering it until it rests in the cup, and only then letting go,
keeps it. The task is the task file
[`place_peaches_in_the_tray_without_dropping_them.py`](place_peaches_in_the_tray_without_dropping_them.py) for the
`yam_arena` embodiment (scene, scoring and the scripted demo), with [`bodykit.py`](bodykit.py) next to it.

## Requirements

[yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) (tested at commit `0059d02`) on Isaac Lab-Arena
0.3.1 / Isaac Sim 6.1, on Linux with an NVIDIA RTX GPU. Its `setup.sh`, or the program's Brev launchable, installs
everything, including the Dream-1 policy. The task needs no models from Arena's object library.

`yam_arena` task files place single boxes, cylinders and spheres, upright, and cannot join them. The tray (a card with
four rings for cups), the foam rings and the peaches (each with its blush and stem scar) are bodies built from several
shapes, so `bodykit.py` adds such bodies: importing it hooks `yam_arena.scene` when the simulator starts, without
changing yam_arena, and gives `score()` each body's pose.

What is modelled and what is not: the peaches are rigid spheres and nothing bruises in the simulation. The rule is
scored from how far each peach fell after it was let go: the height at which it last moved with a gripper (held, or
resting under it), against the height at which it came to rest.

## Run and score a trial

The scorer is `task_score`: 1.0 when `success()` has held for 1 s, otherwise the score at the end of the run. A peach
moves with a gripper while its centre is within 4.5 cm of the gripper's centre and keeps the same offset from it (to 2
mm a step); it was let go at the height where it last moved with a gripper, and comes to rest when it has moved less
than 3 mm in 0.15 s. A peach coming to rest 1.5 cm or more below the height at which it was let go (dropped, or rolled
off and fell) sets the score to 0 for the rest of the run. Otherwise the score is the best stage reached: 0.25 one peach
resting in a cup; 0.5 two; 0.75 three. Full score: all four resting in the tray's cups, both grippers 6 cm or more from
every peach, still for 1 s. `-E object_jitter=0` keeps every object exactly where the task file puts it, so every trial
starts from the same layout.

Dream-1 (needs a key in `~/.dreamlabs/credentials`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=place_peaches_in_the_tray_without_dropping_them.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Pack the peaches into the tray." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/peach-dream
```

The scripted demo, which reaches full score (the left gripper takes its two peaches one at a time across the middle
from above, carries each over a cup on its side of the tray, lowers it until it rests in the cup and only then opens;
the right gripper does the same with its two once the left has moved away):

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=place_peaches_in_the_tray_without_dropping_them.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Pack the peaches into the tray." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/peach-demo
```

Videos, one MP4 per camera next to each run's frames:

```bash
for log in ~/yam-eval/peach-dream/*.json; do inspect-robots video "$log"; done
```

`PEACH_DEBUG=1` prints, every 10 steps, where each peach is, whether a gripper holds it, the height at which it was let
go, how far it fell, whether it rests in a cup, where each gripper is, and the stage reached.

`PEACH_DROP=1` runs the demo the wrong way: each peach is let go 6 cm above its cup; the run scores 0 as the first one
comes to rest.

## Runtime per trial

On one RTX PRO 6000 (Blackwell), on a shared machine under heavy load: about 4 minutes for the scripted demo while
rendering 1280 x 720 video (666 steps, 33 s of simulated time; 241 s), and about 14 minutes for a Dream-1 trial of 2,400
steps (831 s). Isaac Sim's first start on a new machine adds a few minutes while it compiles its shaders.

## Seeds

None needed: with `-E object_jitter=0` the layout is fixed and the scripted demo is deterministic.

## Building it for real

Four ripe peaches (or soft fruit of a similar size), a packing tray with cups 6.4 cm across, and four foam rings.
How far each peach fell when it was let go is judged from the overhead and wrist videos.
