"""On the station table: a green felt card mat with four card places marked on it in white (two rows of two), and a
standard 52-card deck of playing cards lying face down on the mat to their left. Task (instruction): deal one card face
down onto each of the four places, then put the deck down on the mat. The instruction does not say how.

How it is done (pagat.com, Card Game Mechanics): "This is done by dealer holding the pack, face-down, in one hand";
"removing cards from the top of it with her other hand to distribute to the players"; "placing them face-down on the
table in front of the players to whom they are dealt"; "assume that the cards are dealt one at a time". A 0.3 mm card on
a deck cannot be pinched: it is moved by friction alone, a fingertip pressed on its back and drawn off the deck's edge,
and only the top card must move.

The hand skill: one hand picks the deck up by its long edges and holds it level, low over the mat, beside each place
in turn. The other hand comes down on the top card's back with a fingertip, presses lightly and pushes it out over the
deck's edge in short strokes (a fingertip pressed on the part of the card that already overhangs would lever it up
over the edge); once more than half of it is out, the card tips, its leading end onto the felt, its trailing end on the
deck. The holding hand then draws the deck back from under its trailing end, so it drops flat into its place. Pressing
too hard drags the next cards along.

World frame (m): x forward from the arm bases, y to the robot's left, z up; the table top is at z 0.7475.
- card mat: fixed (taped down), green felt 26 x 34 cm, 3 mm thick, centred at (0.21, -0.015). Four places, each the
  outline of a card (63.5 x 88.9 mm, long side along y): A (0.165, -0.090), B (0.255, -0.090), C (0.165, 0.010),
  D (0.255, 0.010) (centres).
- deck: loose, face down, long side along y, centred at (0.210, 0.115) on the mat: the lower 46 cards as one rigid
  body (63.5 x 88.9 x 9.6 mm, 82.8 g) and the top 6 cards as separate cards (59.5 x 88.9 x 1.0 mm, 1.8 g each).

Score (0.25 per place, from the poses; full score held 1 s):
- a card is in a place when half or more of its area is inside the outline and it is down on the mat (its lowest point
  within 8 mm of the felt, not on the deck held above). A place is dealt when exactly one card is in it, with at least
  80 % of its area inside the outline, face down (back up, tilted under 15 deg) and lying on the felt (its underside
  within 2 mm of it). Two or more cards in a place: 0 for that place. Each dealt place gives 0.25.
- while the deck is not set down and released, the score is at most 0.9.
- 1.0 (success, held 1 s): all four places dealt; the deck's lower body resting flat on the mat (underside within 3 mm
  of it, tilted under 10 deg), every fingertip 15 mm or more from the deck and from the dealt cards, and the deck and
  cards still (each under 1 mm in 1 s).
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bodykit import Body, BoxShape, allow_matrix_grips, as_numpy, on_reset, on_step, rotation  # noqa: E402
from yam_arena.demo import Waypoint, grip_rotation  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

allow_matrix_grips()

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
MM = 0.001

# -- the layout ----------------------------------------------------------------------------------------------------
MAT_T = 0.003
MAT_C = (0.21, -0.015)
MAT_SIZE = (0.26, 0.34)
MAT = TOP + MAT_T  # the felt surface
CARD_W, CARD_L = 0.0635, 0.0889  # Bicycle Standard (poker size): 2.5 x 3.5 in; short side along x, long side along y
CARD_T = 0.001  # the separate cards' collision thickness (a real card is 0.3 mm)
CARD_M = 0.0018
TOP_W = CARD_W - float(os.environ.get("DEAL_NARROW", "4")) * MM  # the separate cards are narrower, so the holding
# jaws squeeze only the lower deck
N_TOP = 6  # separate cards on top of the deck
LOWER_T = 0.0096  # the lower 46 cards (the whole deck stays 15.6 mm tall)
LOWER_M = 46 * CARD_M
DECK0 = (0.210, 0.115)
PLACES = {"A": (0.165, -0.090), "B": (0.255, -0.090), "C": (0.165, 0.010), "D": (0.255, 0.010)}

FELT, LINE, BACK, FACE, EDGE = (0.05, 0.33, 0.16), (0.92, 0.92, 0.88), (0.70, 0.10, 0.12), (0.96, 0.96, 0.94), (
    0.90, 0.89, 0.85)

# card friction (static, dynamic): coated card on coated card about 0.3; PhysX averages the two materials' values, so
# card on card is 0.30 / 0.25, a card on the felt (a smooth card-table cloth, 0.30 / 0.25) the same, the robot's
# fingertips (PhysX default 0.5) on a card 0.40 / 0.375
CARD_MU = (float(os.environ.get("DEAL_MU_S", "0.20")), float(os.environ.get("DEAL_MU_D", "0.16")))
DECK_MU = (0.60, 0.55)  # the lower deck's edges and the bottom card's face (the holding jaws grip its edges)
FELT_MU = (float(os.environ.get("DEAL_FELT_S", "0.30")), float(os.environ.get("DEAL_FELT_D", "0.25")))


def _mat() -> Body:
    sx, sy = MAT_SIZE
    sh = [BoxShape((0.0, 0.0, 0.0), (sx, sy, MAT_T), FELT)]
    lw = 0.002  # the printed outlines, 2 mm white lines just outside each place
    for (px, py) in PLACES.values():
        cx, cy = px - MAT_C[0], py - MAT_C[1]
        z = MAT_T / 2 + 0.0001
        for dy in (-1, 1):
            sh.append(BoxShape((cx, cy + dy * (CARD_L / 2 + lw / 2), z), (CARD_W + 2 * lw, lw, 0.0002), LINE,
                               collide=False))
        for dx in (-1, 1):
            sh.append(BoxShape((cx + dx * (CARD_W / 2 + lw / 2), cy, z), (lw, CARD_L, 0.0002), LINE, collide=False))
    return Body.build("mat", pos=(MAT_C[0], MAT_C[1], TOP + MAT_T / 2), mass=1.0, fixed=True, color=FELT,
                      friction=FELT_MU, shapes=sh, contact_offset=0.001)


def _deck() -> Body:
    """The lower 46 cards as one block: white card edges, a red back on top (hidden under the separate cards)."""
    sh = [BoxShape((0.0, 0.0, 0.0), (CARD_W, CARD_L, LOWER_T), EDGE),
          BoxShape((0.0, 0.0, LOWER_T / 2 - 0.0002), (CARD_W - 0.001, CARD_L - 0.001, 0.0005), BACK, collide=False)]
    return Body.build("deck", pos=(DECK0[0], DECK0[1], MAT + LOWER_T / 2 + 0.0003), mass=LOWER_M, color=EDGE,
                      friction=DECK_MU, shapes=sh, contact_offset=0.001, solver_iterations=32)


def _card(k: int) -> Body:
    """Separate card k (0: the top card), face down: red back up, white face down, with a white border on the back."""
    sh = [BoxShape((0.0, 0.0, 0.0), (TOP_W, CARD_L, CARD_T), FACE),
          BoxShape((0.0, 0.0, CARD_T / 2 - 0.00005), (TOP_W - 0.006, CARD_L - 0.006, 0.00012), BACK, collide=False)]
    z = MAT + LOWER_T + (N_TOP - 1 - k + 0.5) * CARD_T + 0.0003 + (N_TOP - k) * 0.0001
    return Body.build(f"card_{k}", pos=(DECK0[0], DECK0[1], z), mass=CARD_M, color=BACK, friction=CARD_MU,
                      shapes=sh, contact_offset=0.001, solver_iterations=32)


MAT_BODY = _mat()
DECK = _deck()
CARDS = [_card(k) for k in range(N_TOP)]
OBJECTS = [MAT_BODY, DECK] + CARDS

# -- tracking and scoring -------------------------------------------------------------------------------------------
IN_FRAC = 0.80  # a dealt card: this much of its area inside the outline
OWN_FRAC = 0.50  # a card "in" a place (two such cards: the place scores 0)
FLAT_DEG = 15.0
ON_FELT = 2.0 * MM
LYING = 8.0 * MM  # a card counts as lying in a place when it is down there (not on the held deck above it)
DECK_FLAT_DEG, DECK_ON = 10.0, 3.0 * MM
CLEAR = 0.015
STILL_STEPS, STILL_MM = 20, 1.0
_DEBUG = os.environ.get("DEAL_DEBUG") == "1"
_DEBUG_EVERY = int(os.environ.get("DEAL_DEBUG_EVERY", "10"))
_EP: dict = {}
_IDS: dict = {}
_U = np.linspace(-0.5, 0.5, 13)
_V = np.linspace(-0.5, 0.5, 17)
_GRID = np.array([[u * TOP_W, v * CARD_L, 0.0] for u in _U for v in _V])  # sample points over a card's area


@on_reset
def _reset() -> None:
    _EP.update(tick=0, step=0, counted=-1, hist=[], places={}, n_good=0, deck_down=False, clear=False, near=1.0,
               cards={}, deck=None, tcp={}, gaps={}, peak_press=0.0)


_reset()


def _finger_points() -> dict:
    import struct
    import xml.etree.ElementTree as ET

    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = pathlib.Path(YAM_URDF).parent
    out = {}
    for link in ET.parse(YAM_URDF).getroot().iter("link"):
        name = link.get("name")
        if name not in ("tip_left", "tip_right"):
            continue
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
            b = (root / col.find("geometry").find("mesh").get("filename")).read_bytes()
            n = struct.unpack("<I", b[80:84])[0]
            tri = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            v = tri["v"].reshape(-1, 3).astype(float)
            pts.append(np.unique(np.round(v, 4), axis=0) @ rpy_matrix(rpy).T + xyz)
        out[name] = np.concatenate(pts)
    return out


def _place_status(cards: dict) -> dict:
    """For each place: the cards with half or more of their area in it, and whether it is dealt."""
    out = {}
    for name, (px, py) in PLACES.items():
        owners, good = [], False
        for cname, (pos, rot) in cards.items():
            pts = pos + _GRID @ rot.T
            inside = (np.abs(pts[:, 0] - px) <= CARD_W / 2) & (np.abs(pts[:, 1] - py) <= CARD_L / 2)
            frac = float(inside.mean())
            low = float(pts[:, 2].min() - MAT)  # its centre plane's lowest point above the felt
            if frac >= OWN_FRAC and low <= LYING:
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, rot[2, 2]))))  # 0: back up (face down)
                under = float(pos[2] - MAT - CARD_T / 2 * abs(rot[2, 2]))
                owners.append(dict(card=cname, frac=frac, tilt=tilt, under=under))
        if len(owners) == 1:
            o = owners[0]
            good = o["frac"] >= IN_FRAC and o["tilt"] <= FLAT_DEG and abs(o["under"]) <= ON_FELT
        out[name] = dict(owners=owners, good=good)
    return out


@on_step
def _track(env) -> None:
    try:
        _track_inner(env)
    except Exception:
        import traceback

        print("DEAL_TRACEBACK\n" + traceback.format_exc(), flush=True)
        raise


def _track_inner(env) -> None:
    from scipy.spatial import cKDTree

    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    _EP["tick"] += 1
    base = getattr(env, "unwrapped", env)
    origin = as_numpy(base.scene.env_origins[0])
    if "fingers" not in _IDS:
        _IDS["fingers"] = _finger_points()
    fingers = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        if side not in _IDS:
            _IDS[side] = list(robot.find_joints(ARM_JOINTS, preserve_order=True)[0])
            _IDS[side + "_f"] = list(robot.find_joints(FINGER_JOINTS, preserve_order=True)[0])
            _IDS[side + "_tips"] = [robot.body_names.index(n) for n in ("tip_left", "tip_right")]
        q = as_numpy(robot.data.joint_pos)[0]
        _EP["tcp"][side] = np.asarray(ik.tcp_pose(q[_IDS[side]], station.BASES[side])[0], float)
        _EP["gaps"][side] = float(-q[_IDS[side + "_f"]].sum())
        poses = as_numpy(robot.data.body_link_pose_w)[0]
        fingers[side] = np.concatenate([_IDS["fingers"][n] @ rotation(poses[i][3:7]).T + poses[i][:3] - origin
                                        for n, i in zip(("tip_left", "tip_right"), _IDS[side + "_tips"])])
    cards = {}
    for c in CARDS:
        p = as_numpy(base.scene[c.name].data.root_link_pose_w)[0]
        cards[c.name] = (p[:3] - origin, rotation(p[3:7]))
    dp = as_numpy(base.scene["deck"].data.root_link_pose_w)[0]
    deck = (dp[:3] - origin, rotation(dp[3:7]))
    places = _place_status(cards)
    good_cards = [pl["owners"][0]["card"] for pl in places.values() if pl["good"]]
    # fingertips' distance from the deck block and from the dealt cards (sampled over their areas)
    pts = [deck[0] + np.array([[u * CARD_W, v * CARD_L, w * LOWER_T] for u in _U[::3] for v in _V[::4]
                               for w in (-0.5, 0.5)]) @ deck[1].T]
    pts += [cards[c][0] + _GRID[::5] @ cards[c][1].T for c in good_cards]
    tree = cKDTree(np.concatenate(pts))
    near = min(float(tree.query(fp, k=1)[0].min()) for fp in fingers.values())
    dtilt = math.degrees(math.acos(max(-1.0, min(1.0, deck[1][2, 2]))))
    dunder = float(deck[0][2] - LOWER_T / 2 - MAT)
    deck_down = dtilt <= DECK_FLAT_DEG and abs(dunder) <= DECK_ON
    _EP.update(places=places, n_good=len(good_cards), deck_down=deck_down, near=near, cards=cards, deck=deck,
               dtilt=dtilt, dunder=dunder, good_cards=good_cards)


def _tally() -> dict:
    hist = _EP["hist"]
    if _EP["counted"] != _EP["tick"] and _EP["deck"] is not None:
        _EP["counted"] = _EP["tick"]
        snap = np.concatenate([_EP["deck"][0]] + [_EP["cards"][c][0] for c in _EP.get("good_cards", [])])
        hist.append(snap)
        del hist[:-STILL_STEPS]
    still = (len(hist) == STILL_STEPS and all(h.shape == hist[-1].shape for h in hist)
             and all(np.abs(h - hist[-1]).max() * 1000 < STILL_MM for h in hist))
    done = bool(_EP["n_good"] == 4 and _EP["deck_down"] and _EP["near"] >= CLEAR and still)
    return dict(still=still, done=done)


def _debug(t) -> None:
    _EP["step"] += 1
    if (_EP["step"] - 1) % _DEBUG_EVERY or _EP["deck"] is None:
        return
    d = _EP["deck"][0]
    cs = " ".join(f"{k[5:]}=({p[0]:.3f},{p[1]:.3f},{(p[2] - MAT) * 1000:.1f}mm,"
                  f"t{math.degrees(math.acos(max(-1, min(1, r[2, 2])))):.0f})" for k, (p, r) in _EP["cards"].items())
    pl = " ".join(f"{n}:{len(v['owners'])}{'G' if v['good'] else ''}" + (
        f"[{v['owners'][0]['frac']:.2f}]" if v["owners"] else "") for n, v in _EP["places"].items())
    print(f"DBG step={_EP['step']} deck=({d[0]:.3f},{d[1]:.3f},{_EP['dunder'] * 1000:.1f}mm,t{_EP['dtilt']:.0f}) "
          f"down={int(_EP['deck_down'])} near={_EP['near'] * 1000:.0f}mm still={int(t['still'])} places {pl} | "
          f"cards {cs} | " + " ".join(f"{k} tcp=({v[0]:.3f},{v[1]:.3f},{(v[2] - MAT) * 1000:.1f}mm) "
                                      f"gap={_EP['gaps'][k] * 1000:.0f}" for k, v in _EP["tcp"].items()), flush=True)


def score(objects) -> float:
    """0.25 per dealt place; at most 0.9 until the deck is set down and released. Full score is success()."""
    t = _tally()
    if _DEBUG:
        _debug(t)
    s = 0.25 * _EP["n_good"]
    if not (_EP["deck_down"] and _EP["near"] >= CLEAR):
        s = min(s, 0.9)
    return s


def success(objects) -> bool:
    return bool(_tally()["done"])


# -- scripted demo --------------------------------------------------------------------------------------------------
# Open loop, from the known layout. The left hand comes down straight over the deck's left end (fingers closing across
# the deck, along x), its pads overlapping the deck's last 50 mm, closes on the long edges (told 6 mm inside each),
# lifts the deck 6 mm off the mat and carries it, level, to beside each place in turn, its right edge 39 mm left of the
# place's middle (a tipped card's middle lands about that far past the edge) and 3 mm forward of it, each place with a
# small correction found by trial (AIM: up to 14 mm along y and 9 mm along x): A and B first (the
# deck held over the empty places C and D), then C and D. At each, the right hand, tilted 55 deg (its wrist out to the
# right, away from the holding hand), deals: three strokes, each coming down on the top card's back 22 mm inside the
# deck's edge already moving right, pressing 2.5 mm into it and pushing it 20 mm right (the card slips a little under
# the fingertip). During the third the card, more than half out, tips over the edge, its leading end onto the felt. The
# fingertip rises clear and the left hand draws the deck 50 mm back (left) and 2 mm down from under the card's trailing
# end, so it drops flat. Then the left hand sets the deck down where it was and lets go, and both hands rise clear.
# DEAL_MODE=press presses 14 mm into the card instead of 2.5 mm (the wrong way: too hard).
OPEN_C = -0.047
MODE = os.environ.get("DEAL_MODE", "demo")  # demo | press (wrong way: presses too hard)
PRESS = float(os.environ.get("DEAL_PRESS", "14.0" if MODE == "press" else "2.5")) * MM
SAG = float(os.environ.get("DEAL_SAG", "2.5")) * MM  # the held deck sits this much lower than commanded (arm's give)
OVERLAP = float(os.environ.get("DEAL_OVERLAP", "50")) * MM  # how much of the deck's end the pads cover
HOLD_H = float(os.environ.get("DEAL_HOLD_H", "6")) * MM  # the held deck's underside above the felt
SQUEEZE = float(os.environ.get("DEAL_SQUEEZE", "6")) * MM  # the jaws told to close this far inside each long edge
PAD_LOW = 0.0128  # the pads reach this far below the TCP (straight down)
PICK_DZ = PAD_LOW + 0.0015  # the TCP above the felt when taking the deck (pad bottoms 1.5 mm up)
TIP_DROP, TIP_DY = 0.0127, 0.0051  # the right fingertip's lowest point: below the TCP, and to its left (tilted 55 deg)
TILT_R = float(os.environ.get("DEAL_TILT", "55"))
PRESS_AT = float(os.environ.get("DEAL_PRESS_AT", "22")) * MM  # the fingertip this far in from the card's right end
CARRY_S = float(os.environ.get("DEAL_CARRY_S", "4.0"))
STROKE = float(os.environ.get("DEAL_STROKE", "20")) * MM  # strokes 1 and 2 push the card this far
SWIPE = float(os.environ.get("DEAL_SWIPE", "6")) * MM  # the fingertip lands moving right over this distance
WD_S = float(os.environ.get("DEAL_WD_S", "2.5"))  # seconds to draw the deck back
WD_DZ = float(os.environ.get("DEAL_WD_DZ", "-2")) * MM  # the deck also rises (or sinks) this much as it is drawn back
WITHDRAW = float(os.environ.get("DEAL_WITHDRAW", "50")) * MM
LAND = float(os.environ.get("DEAL_LAND", "39")) * MM  # the tipped card's middle lies this far past the deck's edge
X_AIM = float(os.environ.get("DEAL_X_AIM", "3")) * MM  # deck and fingertip this far forward (+x) of the place
# per-place corrections (m) to the deck's position, found by trial: how far the tipped card lands from LAND and X_AIM
# differs between the places (the arms' give differs with reach, and each card starts a little differently on the deck)
AIM = {"A": (0.002, -0.011), "B": (0.009, -0.008), "C": (0.007, 0.014), "D": (-0.003, 0.016)}
NPUSH = int(os.environ.get("DEAL_NPUSH", "3"))  # short strokes before the last, long one
STROKE_S = float(os.environ.get("DEAL_STROKE_S", "1.2"))  # seconds per 20 mm of stroke
SQ = -(CARD_W / 2 - SQUEEZE)  # finger value for the squeeze


def _rot_axis(axis, a: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


R_LEFT = grip_rotation(-math.pi / 2)  # straight down, fingers closing along x
R_RIGHT = _rot_axis((1.0, 0.0, 0.0), math.radians(TILT_R)) @ grip_rotation(math.pi / 2)  # wrist out to -y


def _good_ik(arm: str, pos, rot):
    from yam_arena import ik, station

    best = None
    b = station.BASES[arm]
    j1 = math.atan2(float(pos[1]) - b[1], float(pos[0]) - b[0])
    for sd in itertools.product((j1,), (0.8, 1.3), (0.6, 1.2), (-0.8, 0.8), (0.0,), (-1.8, -0.9, 0.0, 0.9, 1.8)):
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), np.asarray(sd, float),
                             base_pos=station.BASES[arm], iters=150)
        if pe < 0.001 and re < math.radians(0.5):
            room = float(np.minimum(q - ik.LOWER, ik.UPPER - q).min())
            if best is None or room > best[1]:
                best = (q, room)
                if room > 0.3:
                    break
    if best is None:
        raise ValueError(f"no IK solution for the {arm} arm at {pos}")
    return best[0]


def _lead_in(arm: str, pos, rot, finger: float, label: str, seconds: float = 3.0, steps: int = 15) -> list:
    from yam_arena import ik, station
    from yam_arena.constants import READY

    q0, q1 = np.array(READY, float), _good_ik(arm, pos, rot)
    out = []
    for k in range(1, steps + 1):
        p, r = ik.tcp_pose(q0 + (q1 - q0) * k / steps, station.BASES[arm])
        out.append(Waypoint(seconds / steps, tuple(float(v) for v in p), finger, label, r))
    return out


def _w(t, p, finger, label, rot) -> Waypoint:
    return Waypoint(t, tuple(float(v) for v in p), finger, label, rot)


def _deck_at(place: str) -> np.ndarray:
    """The deck's centre (x, y) when dealing to ``place``: its right edge ``LAND`` left of the place's middle."""
    px, py = PLACES[place]
    ax, ay = AIM[place]
    return np.array([px + X_AIM + ax, py + LAND + ay + CARD_L / 2])


def _left_tcp(dxy, h: float) -> np.ndarray:
    """The left TCP holding the deck centred at ``dxy`` with its underside ``h`` above the felt."""
    return np.array([dxy[0], dxy[1] + CARD_L / 2 - OVERLAP + 0.034, MAT + h + PICK_DZ])


def _right_tip(x: float, y: float, dz: float) -> np.ndarray:
    """The right TCP with its fingertip's lowest point at (x, y), ``dz`` above the felt."""
    return np.array([x, y - TIP_DY, MAT + dz + TIP_DROP])


def demo():
    top0 = HOLD_H - SAG + LOWER_T + N_TOP * CARD_T  # the top card's back above the felt, while held
    left, right = [], []
    d0 = np.array(DECK0)
    # 1. the left hand takes the deck (the right hand waits at the ready pose)
    left += _lead_in("left", _left_tcp(d0, 0.04), R_LEFT, OPEN_C, "above the deck's left end, open")
    left += [_w(2.0, _left_tcp(d0, 0.0), OPEN_C, "down round the deck's left end", R_LEFT),
             _w(1.2, _left_tcp(d0, 0.0), SQ, "squeeze the deck's long edges", R_LEFT),
             _w(0.8, _left_tcp(d0, 0.0), SQ, "settle", R_LEFT),
             _w(1.0, _left_tcp(d0, HOLD_H), SQ, "lift it a little", R_LEFT),
             _w(1.0, _left_tcp(d0, 0.03), SQ, "lift it clear", R_LEFT)]
    from yam_arena import ik, station
    from yam_arena.constants import READY

    rp, rr = ik.tcp_pose(np.array(READY, float), station.BASES["right"])
    right += [Waypoint(9.0, tuple(float(v) for v in rp), 0.0, "wait, fingers closed", rr)]
    first = True
    for i, place in enumerate(("A", "B", "C", "D")):
        deck_top = top0 - i * CARD_T  # one card fewer on the deck each time
        dxy = _deck_at(place)
        # the left hand carries the deck there (over, then down to 6 mm above the felt), and holds it
        left += [_w(CARRY_S, _left_tcp(dxy, 0.03), SQ, f"carry the deck beside place {place}", R_LEFT),
                 _w(1.5, _left_tcp(dxy, HOLD_H), SQ, "lower it to just above the felt", R_LEFT)]
        edge = dxy[1] - CARD_L / 2  # the deck's right edge (y)
        x = dxy[0]
        y_start = edge + PRESS_AT  # every stroke starts with the fingertip this far inside the deck's edge
        seg = []
        above = _right_tip(x, y_start, deck_top + 0.03)
        if first:
            seg += _lead_in("right", above, R_RIGHT, 0.0, "over the top card, tilted", seconds=4.0)
            first = False
        else:
            seg += [_w(4.0, above, 0.0, "over the next top card", R_RIGHT)]
        # short strokes push the card out over the deck's edge (the fingertip stays over the deck); once more than half
        # of it is out it tips over the edge, its leading end onto the felt, its trailing end still on the deck
        for i in range(NPUSH):
            ya, yb = y_start, y_start - STROKE
            seg += [_w(1.0, _right_tip(x, ya + SWIPE, deck_top + 0.004), 0.0,
                       f"stroke {i + 1}: fingertip over the card's back", R_RIGHT),
                    _w(0.6, _right_tip(x, ya, deck_top - PRESS), 0.0, "come down on it already moving right",
                       R_RIGHT)]
            n = max(4, int(round((ya - yb) / 0.005)))
            for k in range(1, n + 1):
                seg.append(_w(STROKE_S / n, _right_tip(x, ya + (yb - ya) * k / n, deck_top - PRESS), 0.0,
                              f"stroke {i + 1}: push the top card out over the edge", R_RIGHT))
            seg.append(_w(0.6, _right_tip(x, yb, deck_top + 0.004), 0.0, "lift the fingertip", R_RIGHT))
        # the fingertip rises clear and the left hand draws the deck back (left) and down from under the tipped card's
        # trailing end, so it drops flat onto the felt (held down by a fingertip, the card would drag the next card off
        # the deck with it)
        seg += [_w(1.0, _right_tip(x, edge - 0.03, deck_top + 0.03), 0.0, "rise clear of the tipped card", R_RIGHT)]
        t_hold = sum(w.duration for w in seg)
        seg += [_w(WD_S + 0.5, _right_tip(x, edge - 0.03, deck_top + 0.03), 0.0, "wait while the deck is drawn back",
                   R_RIGHT)]
        t_seg = sum(w.duration for w in seg)
        # the right hand waits while the deck is carried; the left hand holds it still while the right hand deals
        right += [_w(CARRY_S + 1.5, right[-1].pos, 0.0, "wait while the deck is carried", right[-1].yaw)]
        right += seg
        back = dxy + np.array([0.0, WITHDRAW])
        left += [_w(t_hold, _left_tcp(dxy, HOLD_H), SQ, f"hold the deck still while place {place} is dealt", R_LEFT),
                 _w(WD_S, _left_tcp(back, HOLD_H + WD_DZ), SQ, "draw the deck back from under the card", R_LEFT),
                 _w(t_seg - t_hold - WD_S, _left_tcp(back, HOLD_H + WD_DZ), SQ, "hold", R_LEFT)]
    # set the deck down where it was, let go, rise clear; the right hand goes back out of the way
    left += [_w(2.5, _left_tcp(d0, 0.03), SQ, "carry the deck back to its spot", R_LEFT),
             _w(1.5, _left_tcp(d0, 0.0005), SQ, "set it down on the felt", R_LEFT),
             _w(1.0, _left_tcp(d0, 0.0005), OPEN_C, "let go", R_LEFT),
             _w(1.5, _left_tcp(d0, 0.06), OPEN_C, "rise clear", R_LEFT),
             _w(10.0, _left_tcp(d0, 0.06), OPEN_C, "stay clear", R_LEFT)]
    clear = np.asarray(right[-1].pos) + np.array([0.0, -0.03, 0.03])
    right += [_w(2.0, clear, 0.0, "move clear", right[-1].yaw), _w(14.0, clear, 0.0, "stay clear", right[-1].yaw)]
    return {"left": left, "right": right}
