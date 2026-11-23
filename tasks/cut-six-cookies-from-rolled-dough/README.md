# Cut six complete cookies from one sheet of rolled dough

Stamp six 52 mm circles that each go through the entire sheet. Leave at least 1 mm between the outlines and between each outline and the slab edge. Leave the shapes in the sheet and set the cutter back on the bench.

## Physical setup

Use a smooth wooden pastry board, a 52 mm round stainless steel cutter, a rolling pin with 4.2 mm guides, a ruler, a probe thermometer and a thin pastry turner. Fit removable washable silicone fingertip covers on shared grippers and handle the cutter by its rolled upper lip. Photograph the setup from above with a ruler visible.

A 300 g batch of the study’s control short dough uses 150 g soft wheat flour (11% protein), 90 g baking shortening (the study used St. Auvent), 45 g sugar, 0.90 g milk powder, 0.30 g salt, 0.30 g sodium bicarbonate, 0.18 g ammonium hydrogen carbonate and 13.32 g water. Mix the shortening on low for 4 min. Add the sugar and mix for 2 min at medium speed, then dissolve the milk powder in the water and mix it in on low for 2 min. Add the flour, salt and leaveners and mix on low for 2 min. Roll to 4.2 mm, trim to 19.8 × 13.7 cm and chill flat for 30 min at 4 °C. Start at 4 ± 1 °C, checked by inserting a probe horizontally into a same-thickness scrap from the batch. Use a fresh sheet and clean, dry cutter for every trial. [Laguna et al. (2011), Sections 2.2 and 2.5.2](https://doi.org/10.1016/j.lwt.2010.05.034)

From overhead, check the number, edge clearance and spacing using the ruler. Once the robot stops, gently lift each circle with the thin turner. A circle with any attached section fails the through-cut criterion. Record the whole lift from above so incomplete cuts remain visible.

## Dough load and through-cut scoring

The published wire test used a 30 mm engaged length of straight wire at 1 mm/s after a 30 min rest at 4 °C. The six control specimens averaged 0.26 ± 0.03 N in the steady region. The simulation interpolates the first 4.2 mm of the representative trace, including a 0.25 mm approach before the reported 0.049 N trigger, and normalizes the trace to the mean plateau. At each rim sector, the modeled force scales with the engaged edge length following the cutting-force relation for thin blades. The vertical forces at the individual rim sectors also apply a net torque to the cutter. A level rim fully engaged to the board has a nominal modeled load of about 1.35 N. The wire experiment used thicker specimens. The ring's blade geometry, friction and speed can change the corresponding physical load. [Raw-dough wire test, Table 1 and Figure 1](https://riunet.upv.es/server/api/core/bitstreams/b5e1c376-68e5-4d1b-a885-04b018d5d02c/content#page=76) · [Cutting-force relation, Equations 1–2](https://ciccotti.espci.fr/sites/ciccotti.espci.fr/IMG/pdf/lechanaultciccotti2022-eml-softcoring-final.pdf)

To check the transfer on a physical setup, use a vertical test stand with a tared force gauge or load cell (0.02 N resolution or finer). Lower the same cutter through a fresh recipe-matched sheet at 1 mm/s. Log force and travel until first board contact, identifiable from the steep rise in force and verified against an empty-board stroke. Repeat with at least six fresh sheets and use the turner to check each disc. Compare the average pre-contact force curve with the simulated load. The current model is based on the published straight-wire measurements.

In simulation, a circle is marked after every rim sector reaches the slab's lower surface in one continuous stamp with PhysX cutter–board contact. A 0.22 mm numerical tolerance accounts for the solver's contact clearance, with a maximum of 2.5 mm lateral drift within the stamp. Lifting clear resets partial rim coverage. A completed circle is displayed as a dark seam. The simulated force and seam approximate cutting while the slab remains visually rigid.

Each valid circle earns 0.125, up to 0.75. A through-cut that overlaps another or crosses the slab edge invalidates the trial. Full credit requires six cuts and the cutter parked on the bench for 1 s, at least 7 mm from the slab with both hands clear.

## Evaluation

The Python file contains the scene, scorer and scripted demonstration and runs with Isaac Lab and yam-isaac-lab-arena. All simulation geometry is built in code.

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=cut_six_cookies_from_rolled_dough.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Cut six complete cookies from the rolled dough with the round cutter. Keep each cut inside the dough and separate from the others. Set the cutter down beside the dough." \
  --scorer task_score --max-steps 1200 --store-frames
```

At seed 0, the successful scripted run took about 31 s of simulated time and 2 min 14 s on an RTX PRO 6000. For the independent policy use `--policy dream` with the same instruction and the 1,200-step limit (60 s at 20 Hz). Set `COOKIE_SHALLOW=1` to stop the cutter partway through the sheet, or `COOKIE_OVERLAP=1` to stamp a third circle across the second.
