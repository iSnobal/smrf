import math
from typing import Tuple

import numpy as np

# define some constants
MAXV = 1.0              # vis albedo when gsize = 0
MAXIR = 0.85447         # IR albedo when gsize = 0
IRFAC = -0.02123        # IR decay factor
VFAC = 500.0            # visible decay factor
VZRG = 1.375e-3         # vis zenith increase range factor
IRZRG = 2.0e-3          # ir zenith increase range factor
IRZ0 = 0.1              # ir zenith increase range, gsize=0
BOIL = 373.15           # boiling temperature K
GRAVITY = 9.80665       # gravity (m/s^2)


# TODO - Need to raise here instead of silently fail to make the user aware
def isint(x):
    try:
        return int(x)
    except ValueError:
        return False


def growth(t):
    """
    Calculate grain size growth
    From IPW albedo > growth
    """

    a = 4.0
    b = 3.0
    c = 2.0
    d = 1.0

    factor = (a + (b * t) + (t * t)) / (c + (d * t) + (t * t)) - 1.0

    return 1.0 - factor


def albedo(telapsed, cosz, gsize, maxgsz, dirt=2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the albedo, adapted from IPW function albedo

    Args:
        telapsed: Time since last snow storm (decimal days)
        cosz:     Cosine local solar illumination angle matrix
        gsize:    Grain size is effective grain radius of snow after last storm (mu m)
        maxgsz:   Max grain size is maximum grain radius expected from grain growth (mu m)
        dirt:     Dirt is effective contamination for adjustment to visible albedo
                  (usually between 1.5-3.0)

    Returns:
        Tuple
            alb_v - albedo for visible spectrum
            alb_ir -  albedo for ir spectrum

    Modified July 23, 2015 - take image of cosz and calculate albedo for
        one time step
    Scott Havens

    """
    # check inputs
    if gsize <= 0 or gsize > 500:
        raise Exception("unrealistic input: gsize=%i", gsize)

    if maxgsz <= gsize or maxgsz > 2000:
        raise Exception("unrealistic input: maxgsz=%i", maxgsz)

    if 1 >= dirt >= 10:
        raise Exception("unrealistic input: dirt=%i", dirt)

    # set initial grain radii for vis and ir
    radius_ir = math.sqrt(gsize)
    range_ir = math.sqrt(maxgsz) - radius_ir
    radius_v = math.sqrt(dirt * gsize)
    range_v = math.sqrt(dirt * maxgsz) - radius_v

    # calc grain growth decay factor
    growth_factor = growth(telapsed + 1.0)

    # calc effective grain size for vis & ir
    gv = radius_v + (range_v * growth_factor)
    gir = radius_ir + (range_ir * growth_factor)

    # calc albedo for cos(z)=1
    alb_v_1 = MAXV - (gv / VFAC)
    alb_ir_1 = MAXIR * np.exp(IRFAC * gir)

    # calculate effect of cos(z)<1

    # adjust diurnal increase range
    dzv = gv * VZRG
    dzir = (gir * IRZRG) + IRZ0

    # calculate albedo
    alb_v = alb_v_1
    alb_ir = alb_ir_1

    # correct if the sun is up
    ind = cosz > 0.0
    alb_v[ind] += dzv[ind] * (1.0 - cosz[ind])
    alb_ir[ind] += dzir[ind] * (1.0 - cosz[ind])

    return alb_v, alb_ir


def decay_alb_power(
    veg: dict,
    veg_type: np.ndarray,
    current_hours: float,
    decay_hours: float,
    pwr: float,
    alb_v: np.ndarray,
    alb_ir: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find a decrease in albedo due to litter accumulation. Decay is based on
    max decay, decay power, and start and end dates. No litter decay occurs
    before start_date. Fore times between start and end of decay,

    .. math::
      \\alpha = \\alpha - (dec_{max}^{\\frac{1.0}{pwr}} \\times
      \\frac{t-start}{end-start})^{pwr}

    Where :math:`\\alpha` is albedo, :math:`dec_{max}` is the maximum decay
    for albedo, :math:`pwr` is the decay power, :math:`t`, :math:`start`,
    and :math:`end` are the current, start, and end times for the litter decay.

    Args:
        veg: Vegetation specific decay values
        veg_type: Array of vegetation type from the topo
        current_hours: Time delta in hours of current time minus decay start time
        decay_hours: Time delta in hours of decay start time to end time
        pwr: power for power law decay
        alb_v: numpy array of albedo for visible spectrum
        alb_ir: numpy array of albedo for IR spectrum

    Returns: Tuple
        alb_v_d, alb_ir_d : numpy arrays of decayed albedo

    """
    alb_dec = np.zeros_like(alb_v)

    if current_hours <= 0:
        return alb_v, alb_ir
    # Use max decay if after start
    elif current_hours > decay_hours:
        # Use default
        alb_dec = alb_dec + veg["default"]
        # Decay based on veg type
        for k, v in veg.items():
            if isint(k):
                alb_dec[veg_type == int(k)] = v

    # Power function decay if during decay period
    else:
        # Use defaults
        max_dec = veg["default"]
        tao = decay_hours / (max_dec ** (1.0 / pwr))

        # Add default decay to array of zeros
        alb_dec = alb_dec + (current_hours / tao) ** pwr

        # Decay based on veg type
        for k, v in veg.items():
            max_dec = v
            tao = decay_hours / (max_dec ** (1.0 / pwr))

            # Set albedo decay at correct veg types
            if isint(k):
                alb_dec[veg_type == int(k)] = (current_hours / tao) ** pwr

    alb_v_d = alb_v - alb_dec
    alb_ir_d = alb_ir - alb_dec

    return alb_v_d, alb_ir_d


def decay_alb_hardy(litter, veg_type, storm_day, alb_v, alb_ir):
    """
    Find a decrease in albedo due to litter accumulation
    using method from :cite:`Hardy:2000` with storm_day as input.

    .. math::
        lc = 1.0 - (1.0 - lr)^{day}

    Where :math:`lc` is the fractional litter coverage and :math:`lr` is
    the daily litter rate of the forest. The new albedo is a weighted
    average of the calculated albedo for the clean snow and the albedo
    of the litter.

    Note: uses input of l_rate (litter rate) from config
    which is based on veg type. This is decimal percent litter
    coverage per day

    Args:
        litter: A dictionary of values for default,albedo,41,42,43 veg types
        veg_type: An image of the basin's NLCD veg type
        storm_day: numpy array of decimal day since last storm
        alb_v: numpy array of albedo for visible spectrum
        alb_ir: numpy array of albedo for IR spectrum
        alb_litter: albedo of pure litter

    Returns:
        tuple:
        Returns a tuple containing the corrected albedo arrays
        based on date, veg type
        - **alb_v** (*numpy.array*) - albedo for visible specturm

        - **alb_ir** (*numpy.array*) -  albedo for ir spectrum

    Created July 19, 2017
    Micah Sandusky

    """
    # array for decimal percent snow coverage
    sc = np.zeros_like(alb_v)
    # calculate snow coverage default veg type
    l_rate = litter['default']
    alb_litter = litter['albedo']

    sc = sc + (1.0-l_rate)**(storm_day)

    # calculate snow coverage based on veg type
    for k, v in litter.items():

        l_rate = litter[k]
        if isint(k):
            sc[veg_type == int(k)] = (
                1.0 - l_rate)**(storm_day[veg_type == int(k)])

    # calculate litter coverage
    lc = np.ones_like(alb_v) - sc

    # weighted average to find decayed albedo
    alb_v_d = alb_v*sc + alb_litter*lc
    alb_ir_d = alb_ir*sc + alb_litter*lc

    return alb_v_d, alb_ir_d

def decay_burned(
    alb_v: np.ndarray,
    alb_ir: np.ndarray,
    last_snow: np.ndarray,
    burn_mask: np.ndarray,
    k_burned: float,
    k_unburned: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply exponential albedo decay as a function of time since last snowfall.
    Different decay rates are applied for burned and unburned pixels.

    Args:
        alb_v:      Visible albedo
        alb_ir:     Infrared albedo
        last_snow:  Time since last snow storm (decimal days)
        burn_mask:  Mask of burned area
        k_burned:   Decay rate for burned area
        k_unburned: Decay rate unburned area

    Returns:
        alb_v_d, alb_ir_d : numpy arrays of decayed albedo
    """
    # initialize output
    alb_v_d = np.empty_like(alb_v)
    alb_ir_d = np.empty_like(alb_ir)

    # masks
    burned = burn_mask == 1
    unburned = burn_mask == 0

    # exponential decay factors depending on burn condition
    burned_exp = np.exp(-k_burned * last_snow)
    unburned_exp = np.exp(-k_unburned * last_snow)

    # apply decay rates to vis and infrared albedo
    alb_v_d[burned] = alb_v[burned] * burned_exp[burned]
    alb_ir_d[burned] = alb_ir[burned] * burned_exp[burned]

    alb_v_d[unburned] = alb_v[unburned] * unburned_exp[unburned]
    alb_ir_d[unburned] = alb_ir[unburned] * unburned_exp[unburned]

    return alb_v_d, alb_ir_d
