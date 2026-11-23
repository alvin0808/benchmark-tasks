# Activate an instant cold pack without opening the outer pouch

The robot lifts the room-temperature pack, squeezes the inner pocket until it opens, mixes the contents and returns the cool pack upright to its holder. One hand can keep the pack from rolling or folding while the other squeezes the filled region. The broad flexible faces transmit compression to the inner pouch, and the outer seams retain both ingredients throughout.

## Physical trial

Use the 5 × 9 in Dynarex 4518 single-use instant cold pack. The named pack uses a urea-based coolant. Condition fresh packs at 22 ± 1 °C and measure the initial temperature at the center of both broad faces. Fix the holder by its base inside a shallow catch tray. The holder is 26.8 × 8.5 cm with two 25 cm lips, each 1.7 cm high and 7 mm thick, separated by a 4 cm slot. Position the pack upright as shown in the setup photograph, using its wide horizontal seam as the lower edge. Use clean silicone fingertip covers. The holder should support the pack upright while leaving its faces and the sides of the upper seam accessible. [Dynarex product family and item numbers](https://dynarex.com/products/disposable-medical-supplies/personal-care/4512-instant-cold-pack)

Before a series, characterize three packs from the same lot using 30 × 45 mm rubber-faced pads and a handheld force gauge. Squeeze the filled pocket through the outer pack. Record the force and pad-to-pad travel when the pocket pops. Confirm that the outer pouch stays dry after release. Mix the contents and record the change in temperature on both faces within 30 seconds of activation. The numerical model uses a nominal 12 N reaction at 16 mm of compression as its starting estimate. A different measured curve calls for updating the simulation parameters and rerunning its demonstrations. This project reports simulator runs. The lot test on physical packs remains to be performed. [Manufacturer squeeze-activation instructions](https://dynarex.com/products/disposable-medical-supplies/first-aid/z-instant_hotcold_packs-instant-hot-and-cold-packs)

In the scored trial, use overhead footage with a ruler to verify the clear lift and the final release. An audible pop provides a witness for the inner pouch opening. If unclear, check at the end whether the separate water pocket is still palpable. Measure both broad faces with the same contact thermometer, then inspect the outer seam and the dry catch tray. Use temperature strips matched to the room-temperature and cold ranges if visual feedback is needed during the run. Both faces must be at least 8 °C colder than their own baselines. The test ends with the pack still and the grippers clear. The trial stops at preparing the pack.

## Simulation

Rigid edge seams and spring-loaded faces approximate the outer film. Compression is calculated from the distance between the moving faces along the pack's own normal. The inner pouch's modeled burst occurs at 16 mm of compression while clear of the holder. Compression beyond 22 mm invalidates the trial. After activation, a mixing proxy counts direction reversals with at least 75 mm of travel. The cold indication derives from elapsed time and those strokes. It is a model of the outcome of shaking the pouch. There is no simulated liquid transfer or heat diffusion.

The first credited state is inner-pouch activation. Cooling earns 0.75. Full credit requires the pack returned upright within 14 mm crosswise and 16 mm lengthwise of its start in the holder, upright within 20 degrees, held for one second with both hands at least 35 mm clear. The half-width of each face and the pack's height are considered for hand clearance.

```sh
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=activate_an_instant_cold_pack.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Activate the instant cold pack and mix it until both sides are cold. Keep the outer pouch sealed and put it back in the holder." \
  --scorer task_score --max-steps 1200 --store-frames
```

Use seed 0 and fixed object poses. Replace `--policy yam_scripted` with `--policy dream` for the independent run. Set `COLD_PACK_GENTLE=1` for the control that lifts and shakes without generating the activation force. All geometry and colors are built in code.
