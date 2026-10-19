"""Navigation constants: OD handover error and IMU error model.

IMU magnitudes are LITERATURE values for flown space-grade units, cited inline. The OD error is a
placeholder set by the user pending the parking-orbit OD study, which is the thing that will
actually produce it.
"""
from __future__ import annotations

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX_DIR = os.path.abspath(os.path.join(HERE, "..", "SCvx"))

G0 = 9.80665
UG = 1e-6 * G0                      # one micro-g in m/s^2

# ---- ORBIT DETERMINATION AT HANDOVER -------------------------------------------------------------
# An IMU measures specific force. Integrating twice gives position CHANGE, so both constants of
# integration come from outside: the OD solution accumulated over the parking orbit and carried
# through braking + TCM. These are the numbers the descent filter STARTS from, and without a
# position-fixing sensor they are a floor it can only move away from.
OD_POS_SIGMA = 10.0                 # m,   1-sigma
OD_VEL_SIGMA = 0.01                 # m/s, 1-sigma

# ---- IMU -----------------------------------------------------------------------------------------
# NORTHROP GRUMMAN LN-200S. Datasheet CS-23015, Approved for Public Release; Distribution
# Unlimited; #23-1041; dated 07/11/23. Every number below is quoted from its accelerometer column.
#
# WHY THIS UNIT. The datasheet's own heritage list names Hayabusa2 and Muses-C (= Hayabusa 1), so
# it is the flown IMU on BOTH asteroid sample-return missions, alongside Clementine, Deep Space 1,
# LADEE and every Mars rover; and it states the hermetic seal is "advantageous for planetary and
# asteroid probes". The Honeywell MIMU flown on OSIRIS-REx has NO public sensor-level
# specification (its datasheet is an image-only PDF), so it cannot be cited at all.
#
#   Bias Repeatability      300 ug, 1 sigma     -> 2.9420e-03 m/s^2
#   Noise                    35 ug/sqrt(Hz)    -> 3.4323e-04 (m/s^2)/sqrt(Hz)
#   Scale Factor Accuracy   300 ppm, 1 sigma
#   Input Axis Alignment      0.1 mrad
#   Bandwidth               200 Hz @ 400 Hz data rate
#
# WHY GRADE MATTERS MORE HERE THAN ON A PLANETARY LANDER. Thrust acceleration on this vehicle is
# 0.0083-0.0334 m/s^2 and Eros surface gravity is 0.0055 m/s^2, so the 300 ug bias is 8.8% of full
# thrust, 35% of minimum thrust and 53% of surface gravity. The same unit on a Mars lander sees
# 3.7 m/s^2 and the bias is 0.008% of it.
#
# WHY THE BIAS IS THE WHOLE ERROR BUDGET. Subtracting truth from the propagation gives
#     dr(t) = dr0 + dv0*t - 0.5*db*t^2 + O(N*sqrt(t^3))
# and at t = 2052 s with these numbers:
#     bias  0.5*b*t^2                        6195 m     <-- dominates everything
#     OD velocity  dv0*t (0.01 m/s)            20.5 m
#     scale factor 0.5*s*a*t^2 (300 ppm)       21 m
#     white noise  N*sqrt(t^3/3)               11 m
#     OD position  dr0 (10 m)                  10 m
# That is why the filter carries 3 bias states, and why the scale factor and the 0.1 mrad
# misalignment are LEFT OUT of the measurement model: at 1.0e-05 and 3.3e-06 m/s^2 they are 300x
# below the bias. Reinstating the scale factor is one term in IMU.measure.
#
# BIAS REPEATABILITY IS NOT BIAS DRIFT. "Repeatability" is a turn-on to turn-on figure: one draw
# per power-up, then CONSTANT for the run. The datasheet gives no in-run stability and no
# accelerometer random-walk number, so there is nothing to justify a drift term -- hence
# IMU_BIAS_TAU_S = inf, which makes the bias a random constant. It also feeds EKF._f, EKF._F and
# EKF._Q correctly with no change to those: -b/inf = 0, -I/inf = 0, and 1 - exp(-dt/inf)^2 = 0.
IMU_RATE_HZ = 100.0                 # simulation sampling rate. The unit runs at 400 Hz; because
                                    # sigma_a = N/sqrt(dt) this sets fidelity, not error magnitude.
IMU_GRADES = {
    # LN-200S, cited above. bias_inst_ug = 0 because the bias is a random constant.
    "ln200s":     dict(bias_ug=300.0, bias_inst_ug=0.0, vrw_ug_rthz=35.0, sf_ppm=300.0),
    # NOT FROM LITERATURE. Kept only so the earlier runs at this grade stay reproducible; it was
    # invented in an earlier session and is 12x better than the flown unit on bias.
    "navigation": dict(bias_ug=25.0,  bias_inst_ug=5.0,  vrw_ug_rthz=8.0,  sf_ppm=100.0),
}
IMU_GRADE = "ln200s"
IMU_BIAS_TAU_S = np.inf             # random constant; see the note above

# THE FRAME THE BIAS IS CONSTANT IN. Physically the bias is fixed in the INSTRUMENT, which is
# body-fixed. In 3DoF there is no attitude state, but the thrust direction gives one body axis, and
# it sweeps 129.5 deg over this descent (ang_act 130.94 -> 1.43 deg). A 300 ug bias held fixed in
# that frame changes in inertial axes by 2*|b_perp|*sin(dtheta/2), where b_perp is the part
# perpendicular to the slew axis. At most that is 543 ug = 5.3e-03 m/s^2 -- LARGER than the bias
# itself -- so the choice of frame is not cosmetic. A typical draw sees roughly half of it.
#   "thrust"   : constant in a thrust-aligned frame, carried by parallel transport. Physical.
#   "inertial" : constant in ACI. What the previous version did; kept for comparison.
IMU_BIAS_FRAME = "thrust"

# ---- LIDAR ---------------------------------------------------------------------------------------
# ONE beam, boresight along -thrust, i.e. the same panel as the aft camera. That is the flown
# arrangement: Hayabusa2 carries its LIDAR and ONC-W1 on the -z panel and thrusts along +z.
#
# INSTRUMENT -- Hayabusa2 LIDAR, Mizuno et al. (2017), Space Science Reviews 208:33,
# "Development of the Laser Altimeter (LIDAR) for Hayabusa2":
#     measurement range      30 m -- 25 km
#     range resolution       0.5 m      (300 MHz sampling)
#     accuracy               +/-1 m at 30 m ,  +/-5.5 m at 25 km
#     measurement rate       1 Hz
#     laser                  Nd:YAG 1064 nm, 15 mJ, 7 ns ;  3.52 kg, 18 W
# It is the only single flown instrument spanning this whole 12 km descent. OSIRIS-REx OLA is far
# more accurate (HELT +/-0.31 m, LELT +/-0.06 m, Daly et al. 2017 SSRv 212:899 Table 3) but its
# high-energy channel tops out at 9.0 km, below our handover altitude.
#
# NOISE: the two quoted accuracies are the only two points given, so sigma is interpolated linearly
# between them and clamped outside. Quantisation is carried separately: a 0.5 m resolution adds a
# uniform step whose variance is q^2/12, which is folded into R rather than left out.
#
# THE 30 m FLOOR IS REAL, NOT A MODELLING CHOICE. Hayabusa2 hands over to its Laser Range Finder
# below about 25 m -- JAXA TD1-R3 briefing, 8 Nov 2018: "Begin LRF altitude control from ~25 m
# altitude". With a single LIDAR and no LRF the beam simply goes dark for the last ~30 m of this
# descent, and the plots show it.
LIDAR_MIN_RANGE_M = 30.0
LIDAR_MAX_RANGE_M = 25000.0
LIDAR_SIGMA_NEAR_M = 1.0            # at LIDAR_MIN_RANGE_M
LIDAR_SIGMA_FAR_M = 5.5             # at LIDAR_MAX_RANGE_M
LIDAR_QUANT_M = 0.5                 # range resolution -> uniform step, variance q^2/12
LIDAR_RATE_HZ = 1.0

# INCIDENCE-ANGLE GATE. A laser altimeter returns nothing useful off a surface too oblique to it.
# Two independent arguments put the limit in the same place and both are worth stating:
#
#   LINK BUDGET. OLA's performance is specified against a 3 % Lambertian surface with 3 dB of margin
#   at maximum range (Daly et al. 2017, Table 1). Lambertian return scales as cos(i), so cos(i)=0.5,
#   i.e. i = 60 deg, exhausts that margin at max range and the limit relaxes as the range closes.
#   The 70 deg used here costs 10*log10(1/cos 70) = 4.7 dB, past OLA's margin at maximum range but
#   comfortable well inside it -- which is where this instrument spends the descent.
#
#   NAVIGATION. For a locally planar surface the range Jacobian is H = -n_face^T / (n_face . u), so
#   |H| = 1/cos(i). At 70 deg that is 2.92: a metre of position error moves the measured range by
#   2.92 m. Past about 80 deg (|H| = 5.8) the measurement is so sensitive to position that a small
#   linearisation error in the filter dominates the instrument noise. 70 deg keeps |H| under 3.
LIDAR_INCIDENCE_MAX_DEG = 70.0

# ---- CAMERAS -------------------------------------------------------------------------------------
# Two cameras, boresights along +thrust and -thrust, same field of view. In 3DoF the thrust
# direction is the only attitude we have, so this is what "body-mounted" means here. The flown
# precedent is exact: Hayabusa2 carries ONC-W1 on the -z panel, the same panel its main engine
# thrusts from, and that is the camera it tracked the target marker with all the way to touchdown.
#
# FIELD OF VIEW -- Hayabusa2 ONC-W1 / ONC-W2, quoted from the USGS ISIS hyb2onc2isis documentation,
# which cites Kameda et al. (2017), Space Science Reviews 208:17, doi 10.1007/s11214-015-0227-y:
#     "square 65.24 x 65.24 deg. field of view (FOV)"   focal length 10.35 mm (W1), 10.44 mm (W2)
# The detector is 1024 x 1000 px at 13 um pitch. That last figure is search-snippet level, but it
# checks out independently: 65.24 deg at f = 10.35 mm over 1024 px implies a 12.94 um pitch, 0.5 %
# from the quoted 13 um. Pixel scale is therefore 65.24/1024 = 1.1120 mrad/px.
#
# MEASUREMENT -- a landmark-based POSITION FIX, which is what the OSIRIS-REx NFT processor and
# Hayabusa2's target-marker tracking actually delivered to their filters. It is NOT what a single
# image gives (that is a bearing, rank 2), so this assumes the on-board landmark database and the
# matching work as intended. Flagged rather than hidden.
#
# NOISE -- there are two defensible anchors and they differ by two orders of magnitude:
#
#   (a) IMAGE-PROCESSING FLOOR. NASA autoNGC (Shoemaker et al., AIAA SciTech 2022, NTRS 20210024481)
#       sets its TRN measurement noise at "0.1 pixels 1-sigma in each image pixel column/row
#       direction", and measures (0.025, 0.169) px against ray-traced synthetic imagery, calling it
#       "a lower-bound on the achievable measurement accuracy". On ONC-W1 that is
#       0.1 px x 1.1120 mrad/px = 111.2 urad, i.e. 1.31 m at 11.8 km and 0.014 m at 125 m.
#
#   (b) FLOWN END-TO-END. OSIRIS-REx NFT during the TAG event (doi 10.3847/PSJ/ac5183, sec 4.2,
#       Fig 24) reports a 1-sigma position uncertainty of about 1.2 m at Checkpoint. Checkpoint sits
#       at roughly 125 m altitude in the TAG sequence -- that altitude is mission-design knowledge,
#       NOT quoted from the paper -- which makes the delivered accuracy ~0.96 % of range.
#
# (b) is 86x worse than (a) at Checkpoint, and the difference is NOT pixels: it is landmark-map
# error, attitude knowledge and lighting. That is the whole reason the two must be combined rather
# than one of them chosen:
#
#     sigma(rho) = sqrt( (k_img * rho)^2 + sigma_floor^2 )
#         k_img       = 111.2 urad   -- anchor (a), scales with range because GSD does
#         sigma_floor =   1.2 m      -- anchor (b), does NOT scale with range
#
# A landmark whose mapped position is wrong by a metre puts a metre into the fix whether you are
# 100 m or 10 km away, so the map/attitude term is a FLOOR, not a slope. Treating (b) as a slope --
# "0.96 % of range", which is what a linear extrapolation of the single Checkpoint data point gives
# -- is unjustified and was wrong by 70x at 15 km. The two terms cross at rho = 10.8 km: pixels
# dominate above it, the floor below. At Checkpoint the model returns 1.200 m, reproducing the
# flown figure by construction.
#
# CAM_SIGMA_MODEL selects: "twoterm" (the above, default), "floor" (pixel term alone, i.e. assume a
# perfect on-board map), or "flown" (the superseded linear model, kept so the difference can be
# shown).
CAM_FOV_DEG = 65.24           # full field of view, ONC-W1/W2
CAM_PIX = 1024                # detector, ONC
CAM_PIXEL_SCALE = 1.1120e-3   # rad/px, = FOV/CAM_PIX
CAM_BEARING_SIGMA = 1.1120e-4 # rad, 0.1 px 1-sigma (autoNGC) -- the floor, anchor (a)
CAM_SIGMA_MODEL = "twoterm"   # "twoterm" | "floor" | "flown"
CAM_SIGMA_FLOOR_M = 1.2       # m, range-independent floor: map + attitude + processing, anchor (b)
CAM_SIGMA_FRAC = 9.6e-3       # superseded linear model, kept for comparison only
CAM_RATE_HZ = 1.0 / 60.0      # autoNGC: "one surface image per minute, with 3 to 5 landmarks"
CAM_MAX_RANGE_M = 25000.0     # beyond this the body is too small to resolve mapped landmarks
CAM_MIN_RANGE_M = 1.0         # below this too little surface is in frame to fix a position

# ---- DEFERRED, DELIBERATELY ----------------------------------------------------------------------
# The near-term goal is an end-to-end chain good enough to generate the SCvx Monte Carlo set that
# trains the guidance PINN. The navigation block is deepened when that PINN is under test, not
# before. Three things are therefore KNOWN to be missing rather than overlooked:
#
#   1. ACTIVATION THRESHOLD. Any sliver of surface in frame currently yields a full rank-3 position
#      fix -- at t = 700 s of the nominal descent that is a 9.3 % crescent in one corner. The real
#      criterion is a count of USABLE landmarks (autoNGC flies 3 to 5 per image) with an
#      incidence-angle limit, plus a conditioning check, because landmarks clustered in one corner
#      leave the solution poorly determined along that cluster. Expect the fixes before ~930 s to
#      be optimistic until this lands. The incidence-angle machinery the altimeter needs is the
#      same machinery, so build it once for both.
#
#   2. THE MAP ERROR IS DRAWN AS WHITE NOISE. CAM_SIGMA_FLOOR_M is landmark-map plus attitude
#      error: nearly the SAME error frame to frame, not an independent sample. Resampling it every
#      fix lets the filter average it down as 1/sqrt(N) -- about sqrt(20) of false confidence over
#      this descent. It belongs as a correlated state or an inflated R. Note also that the floor is
#      an ABSOLUTE error against the map frame, and the landing site is defined in that same frame,
#      so part of it cancels for relative navigation -- which is exactly what Hayabusa2's target
#      marker exploited to land at 0.59 m on much worse absolute knowledge.
#
#   3. NO ILLUMINATION MODEL. A camera needs lit terrain; there is no Sun direction and no phase
#      angle here, so every in-FOV landmark counts as visible. Adding it can only narrow the
#      windows further, never widen them.

# ACTIVATION is geometric and evaluated against the REAL 49 152-facet shape, not a sphere. Two
# tests, OR'd, because neither alone is right at both ends of the descent:
#   * the boresight RAY hits the surface -- exact, and the only test that works close in;
#   * some visible surface VERTEX lies inside the FOV cone -- catches the limb at long range.
# The vertex test alone fails near the ground: mean vertex spacing on this mesh is 211 m, so below
# roughly 12 km range there can be no vertex near the boresight even while the surface fills the
# frame. The ray test alone fails far out, where the body is a small disc off to one side. Using a
# sphere for either understates visibility badly -- Eros is 34 km end to end against a 6.5 km site
# radius, and a spherical approximation puts the blind window at 670 s where the real mesh gives 99.

# ---- FILTER ---------------------------------------------------------------------------------------
# 9 states: position(3), velocity(3), accelerometer bias(3). The bias is a state because at the
# cited 300 ug it is 2.9e-3 m/s^2, which unestimated integrates to 6195 m over a 2052 s descent --
# larger than everything else in the problem put together. No attitude states: this is 3DoF, there
# is no gyro.
# DYNAMIC MODEL COMPENSATION. Three extra states carrying the acceleration the dynamics model does
# not know about, so the filter ESTIMATES the deficiency instead of being dragged by it.
#
# WHY IT IS NEEDED, measured. The filter propagates on PINN-GM-III while the world runs on the 49k
# polyhedral. Those agree to 0.2% above 1 km and disagree by 2.1% at 200-500 m altitude, and the
# disagreement is SIGNED: the PINN pulls down harder by up to 1.06e-4 m/s^2. Integrated along a real
# descent that is 2.07 m of unopposed downward drift by the time the vehicle reaches 2 m. The
# sensors fight it -- which is why the lidar innovations sat at +0.24 to +0.31 m below 500 m instead
# of zero mean -- and what leaks through is a -0.43 +/- 0.47 m altitude bias at contact.
#
# RANDOM WALK, NOT GAUSS-MARKOV. The usual DMC formulation decays the estimate toward zero with a
# finite correlation time. That is wrong here: the lidar goes dark below its 30 m floor, and the
# error is still near its peak there, so the estimate must be HELD through the blind descent rather
# than decayed. tau = infinity does that.
#
# BODY-FIXED, NOT INERTIAL. The error is a function of position, so it is fixed in ACAF and rotates
# in ACI at w. That also separates it from the accelerometer bias, which is constant in the
# THRUST-ALIGNED frame: over the descent the boresight sweeps ~130 deg while ACAF turns ~38 deg, and
# that difference is the only thing making the two observable apart from each other.
#
# SIGMA from the measurement: the error reaches 1e-4 m/s^2 over roughly 300 s of descent through the
# band where it matters, so a random walk that can cover that is 1e-4/sqrt(300) = 5.8e-6.
# DEFAULT OFF, and the A/B that decided it. DMC was built and measured against the control on four
# cases. It did reduce the altitude bias, but it OVERSHOT -- the sign flipped in three of four -- and
# the miss got ~1 m worse in all four, for two reasons that are specific to this problem:
#   1. it traded against the accelerometer bias state (|b_err| went 1.2e-5 -> 5.8e-5), exactly the
#      degeneracy a 130 deg boresight sweep against 38 deg of ACAF rotation cannot resolve;
#   2. a random walk HOLDS what it learned, but the true gravity error PEAKS at 300 m altitude and
#      shrinks 7x by the surface, so holding the value learned at 300 m through the blind final
#      30 m over-corrects. |a_dmc| ended at 1.6-1.9e-4 where the truth was ~1.5e-5.
# The error is deterministic and known, not a random disturbance, so it is corrected in the gravity
# model (see config.PINN_NPZ) rather than estimated here. The states and their Jacobians are kept
# and tested: set this non-zero to absorb a model error that is NOT known in advance.
EKF_DMC_SIGMA = 0.0           # m/s^2 per sqrt(s); 0 freezes the states at zero (= no DMC)
EKF_DMC_P0 = 1.0e-4           # m/s^2, initial 1-sigma on each component
EKF_DMC_FRAME = "body"        # "body" = constant in ACAF, rotated into ACI; "inertial" = ACI-fixed

NX, IR, IV, IB, IA = 12, slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12)

# THE FILTER RUNS IN ACI. The IMU senses there, so no rotation is applied to the measurement at
# all; what the inertial frame costs instead is that GRAVITY has to be rotated, because the field
# is defined body-fixed:  g_I(r,t) = Rz(wt) g_body(Rz(-wt) r). That trade is worth taking -- the
# rotating frame's Coriolis and centrifugal terms are exact linear functions of the state, but they
# are also 5.1x gravity at handover on a body that turns 39.8 deg during the descent, and carrying
# them means every sensor and every state lives in a frame no instrument works in.
#
# THE BIAS FRAME must match how the truth holds it, or not, deliberately:
#   "thrust"   : b is constant in the thrust-aligned frame and enters as R_TAF->ACI(t) b. This
#                MATCHES the IMU device (config_nav.IMU_BIAS_FRAME = "thrust"), so the filter is
#                estimating a genuinely constant quantity.
#   "inertial" : b is constant in ACI. This MISMATCHES the truth, which swings by up to
#                2|b_perp|sin(dtheta/2) over the descent, and is the pessimistic case worth
#                running to show what the frame assumption is worth.
# CHI-SQUARE INNOVATION GATE, per degree of freedom.  y' S^-1 y is chi-square with dim(z) dof when
# the filter is consistent, so a per-dof threshold serves the scalar lidar range and the 3-vector
# camera fix with one number.  13.0 per dof is well beyond the 99.9th percentile for 1 dof (10.83)
# and for 3 dof (16.27/3 = 5.42), i.e. deliberately loose: the gate is here to stop a measurement
# that is wrong by three orders of magnitude, not to trim the tails of a healthy distribution.
# A filter that rejects good data hides its own inconsistency, which is the worse failure.
# Measured need: a lidar return of 11 112 m against a predicted 0.33 m at sigma 3 m was accepted
# and moved the state 414 m.  NIS there was ~1.4e7.
EKF_GATE_NIS = 13.0
EKF_BIAS_FRAME = "thrust"

# The PINN's gravity gradient is identically zero by construction (see gravity.PinnField), so the
# G term in F[v, r] is zero whenever the on-board field is the network. It enters only alongside
# the identity in F[r, v] at O(1e-6) and an inexact Jacobian moves the covariance, not the mean.
EKF_USE_GRAD = False


_WZ = None


def w_body_z():
    """Eros spin rate about +z [rad/s], read once from the SCvx config so there is one definition."""
    global _WZ
    if _WZ is None:
        import sys
        sys.path.insert(0, SCVX_DIR)
        import config as C
        _WZ = float(C.W_BODY[2])
    return _WZ


def imu_params(grade=None, rate=None, bias_frame=None):
    """Datasheet numbers converted to SI. Consumed by both the IMU device and the EKF."""
    g = IMU_GRADES[grade or IMU_GRADE]
    return dict(bias_sigma=g["bias_ug"] * UG, bias_inst=g["bias_inst_ug"] * UG,
                vrw=g["vrw_ug_rthz"] * UG, sf=g["sf_ppm"] * 1e-6,
                tau=IMU_BIAS_TAU_S, rate=rate or IMU_RATE_HZ,
                bias_frame=bias_frame or IMU_BIAS_FRAME)
