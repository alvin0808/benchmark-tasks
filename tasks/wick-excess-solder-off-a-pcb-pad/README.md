# Wick excess solder off a robot motor-driver pad

Clean the tinned power or motor land before soldering a replacement lead. The braid begins lying across the marked deposit. The robot applies the iron to the braid, waits for the copper braid to take up the solder, then removes the iron and braid together. Finish with the iron resting in its holder and the reel and braid on the open bench, clear of the work mat.

The [Chemtronics application guide](https://www.chemtronics.com/desoldering-wick-application-guide) describes this sequence and explains how taking the iron away first can solder the braid to the land. The earlier task [#1206](https://github.com/robocurve/benchmark-tasks/pull/1206) uses a pump. This trial requires two simultaneous physical contacts to transfer heat through the braid, then a coupled lift to release the heat source and the wetted braid before the joint freezes.

## Real trial

Use a plain single-sided FR-4 practice coupon at the motor-driver land size. Start with a 96 × 117 mm board, one exposed 10 × 9 mm copper land with a raised 8 × 6.4 mm deposit, a roughly 5–6 mm braid on a plastic reel, an insulated iron with a 3.2 mm chisel tip and an iron holder. Position the board, reel and iron as in `task.json`. The reel rests in a 70 × 66 mm silicone mat pocket with a flat 11 mm bed and six shallow 4 mm retaining tabs. The pocket keeps the reel from rolling across the bench while it is grasped. Its open top lets the reel lift free. Pull a fresh 88 mm tail beyond the reel edge and lay it over the deposit. Keep the workpiece disconnected and give a cabled iron enough slack to reach its stand.

Use a close overhead or side view that shows the silver deposit, braid, iron tip and attached copper land. Score 0.25 when silver transfers to the braid and 0.6 after the transfer and a clean lift. The full outcome also requires the iron parked in its holder, the reel and braid clear of the work mat, and hands off with both tools still for one second. Direct hot-tip contact with the exposed pad or board, freezing the braid onto the pad, or lifting the pad zeros the trial. A fresh deposit and fresh braid reset the real trial.

## Simulation

The iron and pad have separate collision bodies. A sprung joint gives the braid some give. Pairwise PhysX contacts track the tip against the braid and the braid against the solder. The scorer accumulates 1.3 simulated seconds of dwell with physical contact at both interfaces. Solder disappearing from the target and staining the braid makes the transfer visible. With heat removed, leaving the braid in the solder-film band for 0.25 s represents freezing. The real timing depends on the tip, solder and temperature. The simulation approximates molten flow, heating and adhesion. The scene geometry is generated in code.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=wick_excess_solder_off_a_pcb_pad.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Remove excess solder from the marked motor-driver pad using the copper braid and soldering iron. Lift the braid and iron together, then put the iron back in its holder and leave the reel on the bench." \
  --scorer task_score --max-steps 1200 --store-frames
```

Use `--policy dream` for the independent model run under the same instruction and fixed layout. The trial limit is 60 simulated seconds. The scripted demo takes about 15 simulated seconds, with Isaac RTX rendering taking a few wall-clock minutes. `WICK_LATE=1` raises the iron before lifting the reel.
