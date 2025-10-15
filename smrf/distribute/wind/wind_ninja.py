import os

import numpy as np
import pytz
from scipy.interpolate import interp1d

from smrf.utils import utils


class WindNinjaModel:
    """
    The `WindNinjaModel` loads data from a WindNinja simulation.
    The WindNinja is ran externally to SMRF and the configuration
    points to the location of the output ascii files. SMRF takes the
    files and interpolates to the model domain.
    """

    MODEL_TYPE = "wind_ninja"
    VARIABLE = "wind"
    WN_DATE_FORMAT = "%m-%d-%Y_%H%M"
    DATE_FORMAT = "%Y%m%d"

    def __init__(self, wind_distribution):
        """
        Initialize the WinstralWindModel

        Arguments:
            wind_distribution: Instance of :mod:`smrf.distribute.wind.wind`

        Raises:
            IOError: if maxus file does not match topo size
        """
        self.wind_distribution = wind_distribution

        # wind ninja parameters
        self.wind_ninja_dir = self.config["wind_ninja_dir"]
        self.wind_ninja_dxy = self.config["wind_ninja_dxdy"]
        self.wind_ninja_pref = self.config["wind_ninja_pref"]
        if self.config["wind_ninja_tz"] is not None:
            self.wind_ninja_tz = pytz.timezone(self.config["wind_ninja_tz"].title())

        self.init_interp = True
        self.flatwind = None
        self.dir_round_cell = None
        self.cellmaxus = None

    @property
    def config(self):
        return self.wind_distribution.config

    @property
    def topo(self):
        return self.wind_distribution.topo

    def wind_ninja_path(self, dt, file_type):
        """Generate the path to the wind ninja data and ensure
        it exists.

        Arguments:
            file_type {str} -- type of file to get
        """

        # convert the SMRF date time to the WindNinja time
        t_file = dt.astimezone(self.wind_ninja_tz)

        f_path = os.path.join(
            self.wind_ninja_dir,
            'data{}'.format(dt.strftime(self.DATE_FORMAT)),
            'wind_ninja_data',
            '{}_{}_{:d}m_{}.asc'.format(
                self.wind_ninja_pref,
                t_file.strftime(self.WN_DATE_FORMAT),
                self.wind_ninja_dxy,
                file_type
            ))

        if not os.path.isfile(f_path):
            raise ValueError(
                'WindNinja file does not exist: {}!'.format(f_path))

        return f_path

    def initialize(self):
        """
        Initialize the model with data

        Arguments:
            topo {topo class} -- Topo class
            data {None} -- Not used but needs to be there
        """
        # meshgrid points
        self.X = self.topo.X
        self.Y = self.topo.Y

        self.model_dxdy = np.mean(np.diff(self.topo.x))

        # WindNinja output height in meters
        self.wind_height = float(self.config['wind_ninja_height'])

        # set roughness that was used in WindNinja simulation
        # WindNinja uses 0.01m for grass, 0.43 for shrubs, and 1.0 for forest
        self.wn_roughness = float(self.config['wind_ninja_roughness']) * \
            np.ones_like(self.topo.dem)

        # get our effective veg surface roughness
        # to use in log law scaling of WindNinja data
        # using the relationship in
        # https://www.jstage.jst.go.jp/article/jmsj1965/53/1/53_1_96/_pdf
        self.veg_roughness = self.topo.veg_height / 7.39

        # make sure roughness stays reasonable using bounds from
        # http://www.iawe.org/Proceedings/11ACWE/11ACWE-Cataldo3.pdf
        self.veg_roughness[self.veg_roughness < 0.01] = 0.01
        self.veg_roughness[np.isnan(self.veg_roughness)] = 0.01
        self.veg_roughness[self.veg_roughness > 1.6] = 1.6

        # precalculate scale arrays so we don't do it every timestep
        self.ln_wind_scale = np.log(
            (self.veg_roughness + self.wind_height) / self.veg_roughness
        ) / np.log(
            (self.wn_roughness + self.wind_height) / self.wn_roughness
        )

    def initialize_interp(self, t):
        """Initialize the interpolation weights

        Arguments:
            t {datetime} -- initialize with this file
        """

        # do this first to speedup the interpolation later
        # find vertices and weights to speedup interpolation fro ascii file
        fp_vel = self.wind_ninja_path(t, 'vel')

        # get wind ninja topo stats
        ts2 = utils.get_asc_stats(fp_vel)
        self.windninja_x = ts2['x'][:]
        self.windninja_y = ts2['y'][:]

        XW, YW = np.meshgrid(self.windninja_x, self.windninja_y)
        self.wn_mx = XW.flatten()
        self.wn_my = YW.flatten()

        xy = np.zeros([XW.shape[0]*XW.shape[1], 2])
        xy[:, 1] = self.wn_my
        xy[:, 0] = self.wn_mx
        uv = np.zeros([self.X.shape[0]*self.X.shape[1], 2])
        uv[:, 1] = self.Y.flatten()
        uv[:, 0] = self.X.flatten()

        self.vtx, self.wts = utils.interp_weights(xy, uv, d=2)

        self.init_interp = False

    def distribute(self, data_speed, data_direction):
        """Distribute the wind for the model

        Arguments:
            data_speed {DataFrame} -- wind speed data frame
            data_direction {DataFrame} -- wind direction data frame
        """

        t = data_speed.name

        if self.init_interp:
            self.initialize_interp(t)

        wind_speed, wind_direction = self.convert_wind_ninja(t)
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    def convert_wind_ninja(self, t):
        """
        Convert the WindNinja ascii grids back to the SMRF grids and into the
        SMRF data.

        Args:
            t: datetime of timestep

        Returns:
            ws: wind speed numpy array
            wd: wind direction numpy array

        """

        # get the ascii files that need converted
        fp_vel = self.wind_ninja_path(t, 'vel')
        data_vel = np.loadtxt(fp_vel, skiprows=6)
        data_vel_int = data_vel.flatten()

        # interpolate to the SMRF grid from the WindNinja grid
        g_vel = utils.grid_interpolate(
            data_vel_int, self.vtx,
            self.wts, self.X.shape)

        # There will be NaN's around the edge, handle those first
        if self.model_dxdy != self.wind_ninja_dxy:
            self.wind_distribution._logger.debug(
                "Wind speed from WindNinja has NaN, filling"
            )

            g_vel = self.fill_data(g_vel)

        # log law scale
        g_vel = g_vel * self.ln_wind_scale

        # wind direction from angle, split into u,v components then interpolate
        fp_ang = self.wind_ninja_path(t, 'ang')
        data_ang = np.loadtxt(fp_ang, skiprows=6)

        u = np.sin(data_ang * np.pi / 180)
        v = np.cos(data_ang * np.pi / 180)

        ui = utils.grid_interpolate(
            u.flatten(), self.vtx, self.wts, self.X.shape)
        vi = utils.grid_interpolate(
            v.flatten(), self.vtx, self.wts, self.X.shape)

        uf = self.fill_data(ui)
        vf = self.fill_data(vi)

        g_ang = np.arctan2(uf, vf) * 180 / np.pi
        g_ang[g_ang < 0] = g_ang[g_ang < 0] + 360

        return g_vel, g_ang

    def fill_data(self, grid_values):
        """
        Fill missing values around the edges after the interpolation
        from the WindNinja grid to the configured model grid.

        Parameters
        ----------
        grid_values : np.array
            Interpolated grid data

        Returns
        -------
        np.array
            Grid with filled out edges

        Raises
        ------
        ValueError
            Unsuccessful attempt to fill in the edges
        """
        # Fill in the Y-direction
        grid_values = np.apply_along_axis(
            self.fill_nan, arr=grid_values, axis=1, x=self.X[0, :]
        )
        # Fill in the X-direction
        grid_values = np.apply_along_axis(
            self.fill_nan, arr=grid_values, axis=0, x=self.Y[:, 0]
        )

        if np.any(np.isnan(grid_values)):
            raise ValueError('WindNinja data still has NaN values')

        return grid_values

    @staticmethod
    def fill_nan(data, x):
        """
        Function to use with np.apply_along_axis to fill NaN values in a
        1-d array. Uses scipy interp1d to fill the missing values.

        Parameters
        ----------
        data : 1-d numpy array
            Array passed by np.apply_along_axis
        x : 1-d numpy array
            Values along the x-Axis

        Returns
        -------
        np.array
            New array with NaN filled
        """
        nan_mask = np.isnan(data)

        # Skip rows where all values are NaN
        if np.sum(nan_mask) == data.shape[0]:
            return data

        values = data[~nan_mask]
        x_values = x[~nan_mask]
        func = interp1d(x_values, values, fill_value='extrapolate')
        return func(x)
