# Three-foot balance fixture

[Dimensioned drawings](fixture_drawing.pdf) give the plan, screw stack and wheel section. [wheel_45mm.stl](wheel_45mm.stl) is the wheel blank. Two wheels are required.

This mechanical fixture is for the leveling trial. Its load-bearing feet, two independent adjustments, analog bubble and unloaded pan reproduce the setup. A printed face provides the display. The trial measures leveling of the unloaded case. The [METTLER TOLEDO ML-T guide](https://www.mt.com/dam/P5/labtec/17_Miscellaneous/QG_ML-T_Precision_and_Analytical_Balances_EN.pdf), section 3.6, describes how the two front feet correct both axes. The parts below specify the particular fixture used here.

## Parts and dimensions

All dimensions are in millimeters. The rear pad's lower face is the assembly z = 0 datum. The plan origin is marked in the drawing. Positive x is toward the rear foot and positive y is toward the operator's left.

| Quantity | Part | Specification |
|---:|---|---|
| 1 | Body and dummy display | 221 × 179 housing with an internal 12 mm plywood or aluminum deck. Plan edges x = −91 and +130, y = ±89.5. Deck underside z = 33.5. Front display and vial belong on the top at z = 64. |
| 2 | Outboard screw carriers | Mirrored 12 mm thick aluminum plates. Locate the bores at (−116, +118) and (−116, −118). The outer hub is Ø34. Merge it into a 30 mm wide web ending on the deck at the two M5 holes on the drawing. Use two M5 through bolts per web. Keep the Ø45 wheel envelope open underneath. |
| 2 | Screw and matching nut | Ø12, single-start Tr12×3 left-hand, 60 mm working screw length. Example screw stock, igus PTGSG-12X3-01-L cut to length. Example cylindrical nut, igus JSLM-2618TR12X3. Measure the nut body and bore the carrier for a slip fit, nominally Ø26.2. |
| 2 | Nut captures | One M4 pinch bolt per carrier across a radial saw slit. Add a 2 mm dowel through the carrier wall into a 2 mm deep flat drilled on the nut outside diameter. The pinch bolt retains the nut axially and the dowel prevents rotation. Check that the screw still runs freely. |
| 2 | Handwheels | Ø45 maximum including ribs, 12.5 thick, Ø12.2 through hole. Use the supplied mesh at 1:1 in mm. Print in PETG or machine the profile in aluminum. Drill a radial Ø4.2 through hole to the stem bore. Counterbore from the rim to the insert maker’s diameter. Seat an M4 heat-set insert in that counterbore and fit a recessed M4 × 16 cup-point screw. Seat the screw on a small flat filed on the stem at the wheel center. |
| 2 | Front pad stacks | Ø24 × 7 total height. Bond a Ø24 × 3 nitrile disk to a Ø24 × 4 acetal cap. Place a thin greased steel thrust shim in the cap's central recess. A loose M3 shoulder screw into the drilled and tapped screw end retains the pad. Recess the screw head above the rubber sole. Give the rotating stem 0.2–0.5 axial endplay so the pad can stay planted. |
| 1 | Rear foot | Ø28 × 27 total height. Rigid spacer with a flat nonslip rubber sole, fixed to the housing at (+87, 0). The housing's local rear-foot boss ends at z = 27. |
| 1 | Level and pan | Circular analog vial with roughly Ø27 visible field, mounted at (−67, −44). Pan 144 × 153, centered at (+42, 0), with top z = 75. |
| As needed | Ballast | Bolt steel plates low and centered inside the housing to reach 2.20 ± 0.10 kg including hardware. With the geometry shown, all three pads must stay planted under the unloaded fixture. |

The [igus screw table](https://www.igus.com/info/resumen-de-los-husillos-trapezoidales-drylin?C=ES&L=es) gives the 3 mm lead and the left-hand stock designation. Keep the rotation direction shown on the drawing. Viewed from above, one anticlockwise revolution extends a front pad by 3 mm. All grip surfaces outside the screw axis are visible and accessible.

## Assembly and reset

1. Drill the deck and carrier plates using the coordinate table in the drawing. Set both carrier undersides at z = 33.5. Clamp the nuts with their axes normal to the deck. Slide each wheel onto its stem so its center plane is 19.0 above the assembled front pad's sole. Tighten the cup-point screw, then verify the rotation and travel by one hand turn.
2. Place the fixture upside-down on a padded assembly stand. Use a straightedge across the rear sole as the datum and a depth caliper to set the left sole 3.1 and right sole 5.4 closer to the housing than the rear sole. Mark the wheel index at these reset positions. In this position the left wheel center is z = 22.1 and the right is z = 24.4. Check both wheels turn freely through at least two anticlockwise revolutions and that the assembly gives at least 8 mm of outward stroke.
3. Seat the fixture at the bench marks in the README with all three feet touching. Use the bowl level's actual center. To check or apply the target, set the pan mounting surface horizontal with a machinist's level, shim the vial so its bubble is centered, and mark that center. Select a vial with approximately 1–1.5 mm of bubble travel for a 0.16° tilt. This can be checked with a 0.57 mm shim across a 203 mm span. The target ring has a 1.45 mm radial allowance beyond the measured bubble's edge. The drawing shows an 8.7 mm ID for a 5.8 mm bubble. Set the partial-credit ruler reference 3 mm from the center.
4. Empty the pan. Film the vial from above with a scale marker and all three feet from the side for the end check. Return both wheels to the starting index and verify reset heights before each trial.
