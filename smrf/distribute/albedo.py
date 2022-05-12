import numpy as np
import netCDF4 as nc
import os

from smrf.distribute import image_data
from smrf.envphys import albedo
from smrf.utils import utils


class Albedo(image_data.image_data):
    """
    The :mod:`~smrf.distribute.albedo.Albedo` class allows for variable
    specific distributions that go beyond the base class.

    The visible (280-700nm) and infrared (700-2800nm) albedo follows the
    relationships described in Marks et al. (1992) :cite:`Marks&al:1992`. The
    albedo is a function of the time since last storm, the solar zenith angle,
    and grain size. The time since last storm is tracked on a pixel by pixel
    basis and is based on where there is significant accumulated distributed
    precipitation. This allows for storms to only affect a small part of the
    basin and have the albedo decay at different rates for each pixel.

    Args:
        albedoConfig: The [albedo] section of the configuration file

    Attributes:
        albedo_vis: numpy array of the visible albedo
        albedo_ir: numpy array of the infrared albedo
        config: configuration from [albedo] section
        min: minimum value of albedo is 0
        max: maximum value of albedo is 1
        stations: stations to be used in alphabetical order
    """

    variable = 'albedo'

    # these are variables that can be output
    OUTPUT_VARIABLES = {
        'albedo_vis': {
            'units': 'None',
            'standard_name': 'visible_albedo',
            'long_name': 'Visible wavelength albedo'
        },
        'albedo_ir': {
            'units': 'None',
            'standard_name': 'infrared_albedo',
            'long_name': 'Infrared wavelength albedo'
        }
    }
    # these are variables that are operate at the end only and do not need to
    # be written during main distribute loop
    post_process_variables = {}

    BASE_THREAD_VARIABLES = frozenset([
        'albedo_vis',
        'albedo_ir'
    ])

    def __init__(self, albedoConfig, out_location, start_date):
        """
        Initialize albedo()

        Args:
            albedoConfig: configuration from [albedo] section
        """
        self.start_date = start_date
        # extend the base class
        image_data.image_data.__init__(self, self.variable)

        # Get the veg values for the decay methods. Date method uses self.veg
        # Hardy2000 uses self.litter
        for d in ['veg', 'litter']:
            v = {}

            matching = [s for s in albedoConfig.keys()
                        if "{0}_".format(d) in s]
            for m in matching:
                ms = m.split('_')
                v[ms[-1]] = albedoConfig[m]

            # Create self.litter,self.veg
            setattr(self, d, v)

        self.getConfig(albedoConfig)
        self.out_location = out_location

        self._logger.debug('Created distribute.albedo')

        # MODIS pixel calibration added by DR
        if self.config['modpix_calibration']:

            self._logger.debug('loading .nc with per-pixel decay coefficients')
            
            f = nc.Dataset('/uufs/chpc.utah.edu/common/home/cryosphere/dragar/MODIS_pixel_coeffs/decay_parameters_20y_winter.nc', 'r')
            f.set_always_mask(False)

            # winter params

            # VIS 0
            VFAC_image = f.variables['band_data'][:,:,0]
            self.VFAC_image = VFAC_image[:]
            # VIS 1
            vis_gsmin_image = f.variables['band_data'][:,:,1]
            self.vis_gsmin_image = vis_gsmin_image[:]
            # VIS 2
            vis_gsmax_image = f.variables['band_data'][:,:,2]
            self.vis_gsmax_image = vis_gsmax_image[:]
            # VIS 3
            dirt_image = f.variables['band_data'][:,:,3]
            self.dirt_image = dirt_image[:]
            # IR 0
            IRFAC_image = f.variables['band_data'][:,:,4]
            self.IRFAC_image = IRFAC_image[:]
            # IR 1
            ir_gsmin_image = f.variables['band_data'][:,:,5]
            self.ir_gsmin_image = ir_gsmin_image[:]
            # IR 2
            ir_gsmax_image = f.variables['band_data'][:,:,6]
            self.ir_gsmax_image = ir_gsmax_image[:]

            f.close()

            # spring params
            f = nc.Dataset('/uufs/chpc.utah.edu/common/home/cryosphere/dragar/MODIS_pixel_coeffs/decay_parameters_20y_spring_filtered4.nc', 'r')
            f.set_always_mask(False)

            # VIS 0
            VFAC_image_spring = f.variables['band_data'][:,:,0]
            self.VFAC_image_spring = VFAC_image_spring[:]
            # VIS 1
            vis_gsmin_image_spring = f.variables['band_data'][:,:,1]
            self.vis_gsmin_image_spring = vis_gsmin_image_spring[:]
            # VIS 2
            vis_gsmax_image_spring = f.variables['band_data'][:,:,2]
            self.vis_gsmax_image_spring = vis_gsmax_image_spring[:]
            # VIS 3
            dirt_image_spring = f.variables['band_data'][:,:,3]
            self.dirt_image_spring = dirt_image_spring[:]
            # IR 0
            IRFAC_image_spring = f.variables['band_data'][:,:,4]
            self.IRFAC_image_spring = IRFAC_image_spring[:]
            # IR 1
            ir_gsmin_image_spring = f.variables['band_data'][:,:,5]
            self.ir_gsmin_image_spring = ir_gsmin_image_spring[:]
            # IR 2
            ir_gsmax_image_spring = f.variables['band_data'][:,:,6]
            self.ir_gsmax_image_spring = ir_gsmax_image_spring[:]

            f.close()

        # optical thickness added by DR
        # if it works, should probably be added to topo
        if self.config['optical_thickness']:

            self._logger.debug('loading land surface albedo .nc')
            # load landcover albedo image
            f = nc.Dataset(os.path.join(   
                 self.out_location, 'landcover_albedo.nc'), 'r')
            f.set_always_mask(False)
            surface_albedo = f.variables['Albedo_BSA_vis']
            self.landcover_albedo = surface_albedo[:]
            f.close()

    def initialize(self, topo, data, date_time=None):
        """
        Initialize the distribution, calls image_data.image_data._initialize()

        Args:
            topo: smrf.data.loadTopo.Topo instance contain topo data/info
            data: data dataframe containing the station data

        """

        self._logger.debug('Initializing distribute.albedo')

        #print(f"testing albedo.initialize. date_time: {self.start_date}")


        ### load thicknesss for optical depth - dillon ragar
        # open thickness file using date
        if self.config['optical_thickness']:
            
            previous_day = (self.start_date.to_pydatetime() - 
                timedelta(days=1)).strftime('%Y%m%d')
            
            f = nc.Dataset(os.path.join(   
                 self.previous_day, 'snow.nc'), 'r')
            f.set_always_mask(False)
            
            if 'thickness' in f.variables:
                t = f.variables['time']
                #t_max = t[:].max()
                thickness = f.variables['thickness']
                #time = nc.num2date(
                #    t_max,
                #    t.getncattr('units'),
                #    t.getncattr('calendar'),
                #    only_use_cftime_datetimes=False,
                #    only_use_python_datetimes=True
                #)
                
                #assign np.array for all hours of day
                self.thickness = thickness
                
                f.close()
            
        ### END Ragar thickness test

        # load mod2smrf processing result image daily
        if self.config['modis_albedo']:
            
            self._logger.info(f'testing modis. date: {self.start_date}')
            self._logger.info(f'testing path. path: {self.out_location}')
            # load visible image
            f = nc.Dataset(os.path.join(   
                 self.out_location, 'albedo_vis_modis.nc'), 'r')
            f.set_always_mask(False)
            albedo_vis = f.variables['albedo_vis']
            
            self.albedo_vis_modis = albedo_vis[:]
            self._logger.info(f"modis netcdf vis: {albedo_vis}")
            
            f.close()

            # load ir image
            r = nc.Dataset(os.path.join(   
                 self.out_location, 'albedo_ir_modis.nc'), 'r')
            r.set_always_mask(False)
            albedo_ir = r.variables['albedo_ir']
            
            self.albedo_ir_modis = albedo_ir[:]
            self._logger.info(f"modis netcdf ir: {albedo_ir}")
            
            r.close()
            
        # daily load modis gs file
        if self.config['modis_grainsize']: 
            
            f = nc.Dataset(os.path.join(   
                 self.out_location, 'modis_gs.nc'), 'r')
            f.set_always_mask(False)
            modis_gs = f.variables['grain_size']
            
            self.modis_gs = modis_gs[:]
            #self._logger.info(f"modis netcdf: {albedo_vis}")
            
            f.close()
            

        self.veg_type = topo.veg_type
        self.date_time = date_time
        self._initialize(topo, data.metadata)

        if self.config["decay_method"] is None:
            self._logger.warning("No decay method is set!")

    def distribute(self, current_time_step, cosz, storm_day):
        """
        Distribute air temperature given a Panda's dataframe for a single time
        step. Calls :mod:`smrf.distribute.image_data.image_data._distribute`.

        Args:
            current_time_step: Current time step in datetime object
            cosz: numpy array of the illumination angle for the current time
                step
            storm_day: numpy array of the decimal days since it last
                snowed at a grid cell

        """

        self._logger.debug('%s Distributing albedo' % current_time_step)

        # only need to calculate albedo if the sun is up
        if cosz is not None:

            # use mod2smrf processing
            if self.config['modis_albedo']:
                self._logger.debug(f"index: {current_time_step.hour}")
                self._logger.debug(f"alb ncdf: {self.albedo_vis_modis}")
                #select time 
                alb_v = self.albedo_vis_modis[current_time_step.hour]
                alb_ir = self.albedo_ir_modis[current_time_step.hour]
                
            #use just grain size
            elif self.config['modis_grainsize']:
                alb_v, alb_ir = albedo.albedo_modis(
                    self.modis_gs[current_time_step.hour], 
                    cosz,
                    self.config['max_grain'],
                    self.config['dirt'])

            # use arrays of decay parameters derived from MODIS
            elif self.config['modpix_calibration']:

                print(f"cosz shape: {cosz.shape}", flush=True)

                alb_v, alb_ir = albedo.albedo_modpix_calibration(
                    storm_day,
                    cosz,
                    self.VFAC_image,
                    self.VFAC_image_spring,
                    self.IRFAC_image,
                    self.IRFAC_image_spring,
                    self.vis_gsmin_image,
                    self.vis_gsmin_image_spring,  
                    self.vis_gsmax_image,
                    self.vis_gsmax_image_spring, 
                    self.dirt_image,
                    self.dirt_image_spring,
                    self.ir_gsmin_image,
                    self.ir_gsmin_image_spring,
                    self.ir_gsmax_image,
                    self.ir_gsmax_image_spring,
                    t_curr=current_time_step
                    )

            # use station albedo calibration
            elif self.config['station_calibration']:
                alb_v, alb_ir = albedo.albedo_calibration(
                    storm_day,
                    cosz,
                    current_time_step)


            # smrf existing
            else:
                alb_v, alb_ir = albedo.albedo(
                    storm_day, cosz,
                    self.config['grain_size'],
                    self.config['max_grain'],
                    self.config['dirt'])
        
            

                # Perform litter decay
                if self.config['decay_method'] == 'date_method':
                    alb_v_d, alb_ir_d = albedo.decay_alb_power(
                        self.veg,
                        self.veg_type,
                        self.config['date_method_start_decay'],
                        self.config['date_method_end_decay'],
                        current_time_step,
                        self.config['date_method_decay_power'],
                        alb_v, alb_ir)

                    alb_v = alb_v_d
                    alb_ir = alb_ir_d

                
                elif self.config['decay_method'] == 'hardy2000':
                    alb_v_d, alb_ir_d = albedo.decay_alb_hardy(
                        self.litter,
                        self.veg_type,
                        storm_day,
                        alb_v,
                        alb_ir)

                    alb_v = alb_v_d
                    alb_ir = alb_ir_d
                

                elif self.config['optical_thickness']:
                    # optical thickness decay   
                    albv, alb_ir = albedo.optical_thickness(
                        self.thickness,
                        self.surface_albedo,
                        time_step) 


            self.albedo_vis = utils.set_min_max(alb_v, self.min, self.max)
            self.albedo_ir = utils.set_min_max(alb_ir, self.min, self.max)

        else:
            self.albedo_vis = np.zeros(storm_day.shape)
            self.albedo_ir = np.zeros(storm_day.shape)

    def distribute_thread(self, smrf_queue, data_queue=None):
        """
        Distribute the data using threading and queue

        Args:
            queue: queue dict for all variables
            date: dates to loop over

        Output:
            Changes the queue albedo_vis, albedo_ir
                for the given date
        """
        self._logger.info("Distributing {}".format(self.variable))

        for date_time in self.date_time:

            illum_ang = smrf_queue['illum_ang'].get(date_time)
            storm_day = smrf_queue['storm_days'].get(date_time)

            self.distribute(date_time, illum_ang, storm_day)

            smrf_queue['albedo_vis'].put([date_time, self.albedo_vis])
            smrf_queue['albedo_ir'].put([date_time, self.albedo_ir])
