# Albedo related
IR_FACTOR = -0.02123    # IR decay factor
IR_MAX = 2.8
IR_MAX_0 = 0.85447      # IR albedo when grain_size = 0
IR_MIN = .7
IR_Z_RF = 2.0e-3        # IR zenith increase range factor
IR_Z_0 = 0.1            # IR zenith increase range, grain_size = 0
VIS_FACTOR = 500.0      # Visible decay factor
VIS_MAX = .7
VIS_MAX_0 = 1.0         # Visible albedo when grain_size = 0
VIS_MIN = .28
VIS_Z_RF = 1.375e-3     # Visible zenith increase range factor

# Visible solar irradiance wavelengths
VISIBLE_WAVELENGTHS = [VIS_MIN, VIS_MAX]
# Infrared solar irradiance wavelengths
IR_WAVELENGTHS = [IR_MIN, IR_MAX]

BOIL = 373.15            # Boiling temperature K
EMISS_TERRAIN = 0.98     # Emissivity of the terrain
EMISS_VEG = 0.96         # Emissivity of the vegetation
FREEZE = 273.16          # Freezing temperature K
GRAVITY = 9.80665        # Gravity (m/s^2)
MOL_AIR = 28.9644        # Molecular weight of air (kg / kmole)
RGAS = 8.31432e3         # Gas constant (J / kmole / deg)
SEA_LEVEL = 1.013246e5   # Sea level pressure
SOLAR_CONSTANT = 1368.0  # Solar constant in W/m**2
STD_AIRTMP = 2.88e2
STD_LAPSE = -6.5         # Lapse rate (K/km)
STD_LAPSE_M = -0.0065    # Lapse rate (K/m)
STEF_BOLTZ = 5.6697e-8   # Stefan Boltzmann constant
