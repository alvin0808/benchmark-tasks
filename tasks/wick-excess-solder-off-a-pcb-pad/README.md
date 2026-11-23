# Wick excess solder off a robot motor-driver pad

Clean the tinned power or motor land before soldering a replacement lead. The braid begins lying across the marked deposit. The robot applies the iron to the braid, waits for the copper braid to take up the solder, then removes the iron and braid together. Finish with the iron resting in its holder and the reel and braid on the open bench, clear of the work mat.

This trial requires two simultaneous physical contacts to transfer heat through the braid, then a coupled lift to release the heat source and the wetted braid before the joint freezes. The Chemtronics application guide describes this lift and explains how taking the iron away first can solder the braid to the land. [Chemtronics application guide](https://www.chemtronics.com/desoldering-wick-application-guide)

## Real trial

Use an unpowered 96 × 117 mm single-sided FR-4 practice coupon with a 10 × 9 mm exposed copper motor-driver land. Use a temperature-controlled iron with a 3.2 mm chisel tip and an insulated handle. Set the station to 350 °C, check 340–360 °C at the stabilized unloaded tip with a tip thermometer, and wipe and re-tin the tip with SAC305 before the trial. Use a 5–6 mm copper braid with factory-applied no-clean flux, such as Chemtronics 60-6-5 (5.1 mm). The braid's factory flux is sufficient for the trial. [HAKKO tip temperature guidance](https://www.hakko.com/english/support/maintenance/detail.php?seq=183) · [Chemtronics braid specifications](https://www.chemtronics.com/soder-wick-no-clean)

### Prepare the deposit

1. Start with a clean spare coupon at room temperature. Fix a flat stainless shim, 0.2–0.5 mm thick, to the coupon with a clamp. Use a 6.0 mm circular opening centered on the land to limit the wetted footprint. Wipe the copper with isopropyl alcohol and let it dry. Apply one thin pass of MG Chemicals 836LFNC-P lead-free no-clean flux to the copper inside the opening.
2. Weigh 0.22 ± 0.02 g of solid lead-free SAC305 wire (96.5% Sn, 3.0% Ag and 0.5% Cu) on a 0.01 g balance. For 0.80 mm diameter solid wire, a 59 mm piece is approximately 0.22 g. Touch the 350 °C chisel tip to the exposed copper and feed the premeasured piece into the heated land until it forms a single wetted mound. Withdraw the tip and allow the deposit to solidify before removing the shim.
3. After cooling, check that the central mound is within the marked 8 × 6.4 mm rectangle and 1.6–2.2 mm above the adjacent copper. Use a calibrated close side image or the difference of two caliper height readings from the underside of the coupon. A coupon with unwetted balls or excess spread is replaced with another prepared coupon. Pre-cut and weigh the wire before a session so the soldering, cooling and inspection can be done in a few minutes per coupon.

[Kester solid SAC305 wire](https://www.kester.com/products/product/solid-wire) · [Kapp SAC305 alloy density](https://www.kappalloy.com/pub/media/pdf/TDS%20280%20SAC305%20Tin%20Silver%20Copper.pdf) · [MG Chemicals flux pen](https://mgchemicals.com/products/soldering-supplies/flux-pens/lead-free-flux-pen/)

### Reset and score

Position the board, reel and iron as in `task.json`. The reel rests in a 70 × 66 mm silicone mat pocket with a flat 11 mm bed and six shallow 4 mm retaining tabs. The pocket keeps the reel from rolling across the bench while it is grasped. Its open top lets the reel lift free. Pull a fresh 88 mm tail beyond the reel edge and lay it over the deposit. Return the iron to its holder with enough cable slack to reach the coupon.

Use local fume extraction with its capture hood close to the coupon and outside the arm paths. Turn extraction on before preparing the mound or running a trial. Fit the iron gripper with secure removable woven silica heat-resistant covers rated above 350 °C. Mark the middle of the insulated handle as the grasp region and check with a cold iron that both pads stay on the handle and clear of the metal shaft and tip throughout the movement. Park the iron in its heat-resistant holder and allow the used braid to cool before trimming it. [HSE fume-control guidance](https://www.hse.gov.uk/pubns/guidance/oce4.pdf) · [Example silica fabric specification](https://mid-mountain.com/wp-content/uploads/2019/08/Siltex-18-UH-Amorphous-Silica-Cloth.pdf)

Use a close overhead or side view that shows the silver deposit, braid, iron tip and attached copper land. Score 0.25 when silver transfers to the braid and 0.6 after the transfer and a clean lift leaves the land exposed and the mound gone. A flat tinned finish on the exposed land is acceptable. The full outcome also requires the iron parked in its holder, the reel and braid clear of the work mat, and hands off with both tools still for one second. Direct hot-tip contact with the exposed pad or board, freezing the braid onto the pad, or lifting the pad zeros the trial.

## Simulation

The iron and pad have separate collision bodies. A sprung joint gives the braid some give. The deposit is an 8 × 6.4 × 2.1 mm rigid contact envelope around the crest of the rounded physical mound. Its box volume is independent of the solder charge specified for the physical reset. Pairwise PhysX contacts track the tip against the braid and the braid against the solder. The scorer accumulates 1.3 simulated seconds of dwell with physical contact at both interfaces. Solder disappearing from the target and staining the braid makes the transfer visible. With heat removed, leaving the braid in the solder-film band for 0.25 s represents freezing. Heat flow and capillary uptake are approximated, and actual timings depend on the solder, tip and temperature. The geometry is generated in code. The recorded runs exercise the simulator. The physical deposit and wicking procedure are specified for reproduction and await bench validation.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=wick_excess_solder_off_a_pcb_pad.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Remove excess solder from the marked motor-driver pad using the copper braid and soldering iron. Lift the braid and iron together, then put the iron back in its holder and leave the reel on the bench." \
  --scorer task_score --max-steps 1200 --store-frames
```

Use `--policy dream` for the independent model run under the same instruction and fixed layout. The trial limit is 60 simulated seconds. The scripted demo takes about 15 simulated seconds, with Isaac RTX rendering taking a few wall-clock minutes. `WICK_LATE=1` raises the iron before lifting the reel.
