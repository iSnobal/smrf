############################################
Weather Research and Forecasting (WRF) input
############################################

SMRF reads WRF output in netCDF format, either as a single file or as a
folder of hourly files. The data set is detected from the variables in the
file.

Data set detection
==================

A file is read as WUS-D3 when it contains all of these variables:
``T2``, ``Q2``, ``PSFC``, ``RAINNC``, ``U10``, ``V10``, ``SWDNB``,
``SWDNBC`` and ``LWDNB``. Otherwise it is read as default WRF output, which
needs ``T2``, ``DWPT``, ``RAINNC``, ``UGRD``, ``VGRD`` and ``CLDFRA``. A
file that has neither set of variables raises an error.

Configuration
=============

Set ``data_type`` to ``wrf`` in the ``[gridded]`` section. The options
are the same for both data sets:

``wrf_path``
   A single WRF file, or a folder of hourly files. Used for both data sets.

``wrf_prefix``
   Required when ``wrf_path`` is a folder. Each hour is read from
   ``{wrf_prefix}_YYYY-MM-DD_HH:00:00.nc``. Every hour from ``start_date``
   to ``end_date`` needs a file, and a missing file stops the run.

``wrf_metadata``
   File that holds the grid fields ``XLAT``, ``XLONG`` and ``HGT``. Without
   it, the grid fields are read from ``wrf_path``. The WUS-D3 hourly files
   do not have the grid fields, so WUS-D3 input needs this file. It must be
   on the same grid as the files in ``wrf_path``.

Example for WUS-D3 hourly files downloaded with the `model_setup`_ scripts:

.. code-block:: ini

  [gridded]
  data_type:     wrf
  wrf_path:      /path/to/WUS-D3/cesm2_r11i1p1f1_ssp370_bc
  wrf_prefix:    auxhist_d01
  wrf_metadata:  /path/to/WUS-D3_domain.nc

Variables
=========

================  ============================  ============================
Quantity          Default WRF                   WUS-D3
================  ============================  ============================
Times             ``Times``                     ``Times``
Grid fields       ``XLAT``, ``XLONG``, ``HGT``  ``XLAT``, ``XLONG``, ``HGT``
Air temperature   ``T2``                        ``T2``
Vapor pressure    ``DWPT``                      ``Q2``, ``PSFC``
Precipitation     ``RAINNC``                    ``RAINNC``, ``RAINC``
Wind components   ``UGRD``, ``VGRD``            ``U10``, ``V10``
Cloud factor      ``CLDFRA``                    ``SWDNB``, ``SWDNBC``
Longwave          none                          ``LWDNB``
================  ============================  ============================

Variable Notes
--------------

WUS-D3
^^^^^^
In the hourly WUS-D3 output files, ``Q2`` is the water vapor mixing ratio.
The `WUS-D3 technical document`_ lists the 2 m field as "2-m specific humidity"
in kg kg-1. Hourly files are specified to be in their WRF-native form. However,
native WRF output has ``QVAPOR`` defined as the water vapor mixing ratio
and ``Q2`` is described as "QV at 2 M".

Precipitation
=============

Precipitation is a running total, so hourly precipitation is the difference
between consecutive hours. For WUS-D3, ``RAINC`` is added to ``RAINNC``
when the file has it. The first hour needs the hour before
``start_date``. In folder mode, that hour's file must be in the folder
under ``wrf_prefix``. If it isn't there, the first hour's precipitation is
set to 0.

WUS-D3 data
===========

The `model_setup`_ repository has scripts to download the WUS-D3 hourly
files and to create one ``wrf_metadata`` file for the full WUS-D3 domain.
That file covers every basin in the domain and works with hourly files that
cover the full domain. SMRF reads only the grid cells a basin needs.

Run dates
---------

WUS-D3 simulates each year separately, from September 1 to September 1 of
the next year, so the precipitation totals of one year do not continue into
the next. Keep ``start_date`` and ``end_date`` within October 1 to August 31
of one water year, so the precipitation differences come from one
simulation.

Thermal radiation
-----------------

Add ``hrrr_thermal`` to the ``[output]`` variables to use the WUS-D3
longwave, ``LWDNB``. SMRF interpolates it to the topo grid and corrects it
for the sky view. With ``thermal``, SMRF computes the longwave from the
cloud factor instead.

Grid cells
----------

The WUS-D3 grid cells are 9 km, so a basin covers few of them. SMRF uses the
cells within the topo extent plus a 0.1 degree buffer. With ``log_level``
set to ``debug`` in the ``[system]`` section, the log shows the number of
grid cells within the model domain. Keep ``grid_local_n`` below that
number, for example 9 instead of the default 25.

Outputs
=======

* ``net_solar`` and ``thermal`` work with both data sets.
* ``hrrr_thermal`` uses ``LWDNB`` directly, so it works only with WUS-D3
  files.

References
==========

.. _model_setup: https://github.com/iSnobal/model_setup
.. _WUS-D3 technical document: https://www.researchgate.net/publication/374504614_Data_tier_descriptions_directory_structure_and_data_access_of_the_Western_US_Dynamically_Downscaled_Dataset_WUS-D3_version_1

.. target-notes::
