# Lint-roll a dark chair cushion

Use the adhesive lint roller to clean the loose white fibers off the dark chair cushion, then set the roller down beside it. The roller has to stay in light, rolling contact as it crosses the cushion. A hovering or skidding roller leaves fibers behind. The necessary actions are visible in the motion of the roller and the before-and-after contrast.

## Setup

Set an ordinary 30 × 30 × 3.8 cm removable charcoal chair pad flat on a clean countertop. Use woven polyester or cotton upholstery over foam with a grippy backing, around 420 g. Center it 32.5 cm forward of the robot base midpoint and 6.2 cm to the robot's left. Prepare fifteen white cotton-thread tufts, each with two strands about 12–16 mm long. Lay them loosely on the fabric without pressing them in. Relative to the cushion center, place a tuft at each combination of x = −7.3, −2.7 and +3.4 cm and y = −6.7, −2.8, +0.9, +5.2 and +7.5 cm. Keep x forward and y to the robot's left. A marked grid on a see-through plastic sheet can be held just over the fabric while placing the tufts and removed before the trial.

Use a standard [3M Scotch-Brite 4-inch adhesive lint roller](https://www.scotch-brite.com/3M/en_US/p/d/b00019857/), with a 10.2 cm drum about 5.4 cm in diameter and a 10 cm plastic handle in line with its axle. Expose a fresh adhesive sheet. Lay the roller sideways on a 9.5 × 22 cm coated silicone mat beside the cushion. Its handle midpoint starts 26.5 cm forward and 25 cm to the robot's right, with the drum pointing toward the left. Use the mat to keep the sticky sleeve from picking up dirt from the table before the trial. Start with both grippers open at the ready pose.

## Score

Take an overhead image of the cushion at the beginning and end. The video shows whether the adhesive roller actually picked up the tufts. Count the original fifteen sites, crediting each once when the tuft transfers from the cushion to the rolling drum. Count a removed tuft only when the video shows the roller cleaning its site. Sweeping cotton to another location, blowing it away, or picking it up by hand earns no credit.

- Partial credit is `0.9 × (number transferred / 15)`, from zero for no transfer to 0.9 for all fifteen.
- Full credit is 1.0 for all fifteen, with the cushion center within 3 cm of its start and the roller released and resting on the countertop beside it for one second. Measure the shift from two removable corner marks and confirm the release on video.

The simulation uses the real handle and sleeve dimensions, a free-running cylindrical joint, and PhysX contacts with the cushion. A tuft transfers only when the sleeve passes over it while supporting at least 0.35 N of normal contact and its rim travel agrees with its horizontal rolling speed. The visual tuft moves to the sleeve after contact. Foam compression and individual adhesive fibers are approximated by rigid contact and this explicit transfer rule.

## Run

Use [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) with Isaac Lab-Arena. The task uses no external geometry or textures. Run from this folder. Switch `--policy yam_scripted` to `--policy dream` for the model trial. Use `LINT_HOVER=1` to run the no-contact control.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=lint_roll_a_dark_chair_cushion.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Use the adhesive lint roller to clean all the white fibers off the cushion. Leave the cushion in place and set the roller down beside it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lint-roller
```

Reset with fresh adhesive and fifteen new tufts. With `object_jitter=0`, the starting scene is deterministic and needs no seed.

The scripted trial transferred all fifteen tufts in two passes and scored 1.0 at step 410 (20.5 seconds of simulation, about a minute elapsed on an NVIDIA RTX PRO 6000). The same passes above the fabric with `LINT_HOVER=1` scored 0.0. Dream-1 scored 0.0 at the 1,200-step limit.
