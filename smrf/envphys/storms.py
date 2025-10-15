import numpy as np
import pandas as pd


def time_since_storm(
    precipitation,
    percent_snow_precipitation,
    storm_days,
    storm_precip,
    time_step,
    mass_threshold=1.0,
    percent_snow_threshold=0.5
):
    """
    Increase or reset the given storm days and storm precipitation based of
    given percent snow and mass threshold.

    Steps:
     - Look for pixels where percent snow precipitation is above the threshold.
     - Reset storm days if the precipitation at the pixel is above the mass
         threshold or increase the counter if it is below.
     - Add the precipitation amount to the storm precipitation.

    *NOTE*: Storm day precipitation is initialized and set to 0 with each
        processed day. Hence, no cross-day storm tracking is possible if the
        period between the days falls below the threshold.

    Args:
        precipitation: Precipitation values
        percent_snow_precipitation: Percent of precipitation that was snow
        storm_days: Storm days to keep track of
        storm_precip: Keeps track of the total storm precip
        time_step: Step in days of the model run
        (Optional)
        mass_threshold: Minimum amount of precipitation required to be a storm
                        (snow mass). Default: 0.5
        percent_snow_threshold: Minimum fraction for values in
                                `percent_snow_precipitation` to be considered
                                a snow event. Default: 0.5 (50%)

    Returns:
        tuple:
        - **stormDays** - Updated storm days
        - **stormPrecip** - Updated storm precipitation

    Created January 5, 2016
    @author: Scott Havens

    Updated: February 07, 2022
    @author: Joachim Meyer, Dillon Ragar
    """

    # Step 1: Pixels above snow percent threshold
    location_index = (percent_snow_precipitation >= percent_snow_threshold)
    storm_precip[location_index] += precipitation[location_index]

    # Step 2: Reset locations above mass threshold or increase counter when
    # below
    location_index = (storm_precip >= mass_threshold)
    storm_days[location_index] = 0
    storm_days[~location_index] += time_step

    # Step 3: Increase the storm precipitation total for the day
    storm_precip[~location_index] = 0

    return storm_days, storm_precip


def time_since_storm_pixel(precipitation, dpt, perc_snow, storming,
                           time_step=1/24, stormDays=None, mass=1.0,
                           ps_thresh=0.5):
    """
    Calculate the decimal days since the last storm given a precip time series

     - Will assign decimal days since last storm to every pixel

    Args:
        precipitation: Precipitation values
        dpt: dew point values
        perc_snow: percent_snow values
        storming: if it is stomring
        time_step: step in days of the model run
        stormDays: unifrom days since last storm on pixel basis
        mass: Threshold for the mass to start a new storm
        ps_thresh: Threshold for percent_snow

    Returns:
        stormDays: days since last storm on pixel basis

    Created October 16, 2017
    @author: Micah Sandusky
    """
    # either preallocate or use the input
    if stormDays is None:
        stormDays = np.zeros(precipitation.shape)

    # add timestep
    stormDays += time_step

    # only reset if stomring and not overly warm
    if storming and dpt.min() < 2.0:
        # determine location where there is enough mass
        idx_mass = precipitation >= mass
        # determine locations where it has snowed
        idx = perc_snow >= ps_thresh

        # reset the stormDays to zero where the storm is present
        stormDays[(idx_mass & idx)] = 0

    return stormDays


def tracking_by_station(precip, mass_thresh=0.01, steps_thresh=3):
    """
    Processes the vector station data prior to the data being distributed

    Args:
        precipitation: precipitation values
        time: Time step that smrf is on
        time_steps_since_precip: time steps since the last precipitation
        storm_lst: list that store the storm cycles in order. A storm is
                    recorded by its start and its end. The list
                    is passed by reference and modified internally.
                    Each storm entry should be in the format of:
                    [{start:Storm Start, end:Storm End}]

                    e.g.
                    [
                    {start:date_time1,end:date_time2,'BOG1':100, 'ATL1':85},
                    {start:date_time3,end:date_time4,'BOG1':50, 'ATL1':45},
                    ]

                    would be a two storms at stations BOG1 and ATL1

        mass_thresh: mass amount that constitutes a real precip event,
            default = 0.01.

        steps_thresh: Number of time steps that constitutes the end of a precip
            event, default = 2 steps (typically 2 hours)

    Returns:
        tuple:
            - **storms** - A list of dictionaries containing storm start,stop,
                mass accumulated, of given storm.

            - **storm_count** - A total number of storms found

    Created April 24, 2017
    @author: Micah Johnson
    """

    storm_columns = ['start', 'end']
    stations = list(precip)
    storm_columns += stations

    storms = []

    stations = list(precip)
    is_storming = False
    time_steps_since_precip = 0

    for i, row in precip.iterrows():
        time = pd.Timestamp(i)

        # Storm Idenificiation
        if row.max() > mass_thresh:
            # Start a new storm
            if not is_storming:
                new_storm = {}
                new_storm['start'] = time
                for sta, p in row.iteritems():
                    new_storm[sta] = 0
                # Create a new row
                is_storming = True

            time_steps_since_precip = 0
            # Always add the latest end date to avoid unclosed storms
            new_storm['end'] = time

            # Accumulate precip for storm total
            for sta, mass in row.iteritems():
                new_storm[sta] += mass

        elif is_storming and time_steps_since_precip < steps_thresh:
            new_storm['end'] = time

            time_steps_since_precip += 1

        if time_steps_since_precip >= steps_thresh and is_storming:
            is_storming = False
            storms.append(new_storm)
            # print "=="*10 + "> not storming!"

    # Append the last storm if we ended during a storm
    if is_storming:
        storms.append(new_storm)

    storm_count = len(storms)

    # Make sure we have storms
    if storm_count == 0:
        empty_data = {}
        for col in storm_columns:
            empty_data[col] = []
        storms = pd.DataFrame(empty_data)
    else:
        storms = pd.DataFrame(storms)

    return storms, storm_count


def tracking_by_basin(precipitation, time, storm_lst, time_steps_since_precip,
                      is_storming, mass_thresh=0.01, steps_thresh=2):
    """
    Args:
        precipitation: precipitation values
        time: Time step that smrf is on
        time_steps_since_precip: time steps since the last precipitation
        storm_lst: list that store the storm cycles in order. A storm is
                    recorded by its start and its end. The list
                    is passed by reference and modified internally.
                    Each storm entry should be in the format of:
                    [{start:Storm Start, end:Storm End}]

                    e.g.
                         [
                         {start:date_time1,end:date_time2},
                         {start:date_time3,end:date_time4},
                         ]

                         #would be a two storms

        mass_thresh: mass amount that constitutes a real precip
                    event, default = 0.0.
        steps_thresh: Number of time steps that constitutes the end of
                        a precip event, default = 2 steps (default 2 hours)

    Returns:
        tuple:
            storm_lst - updated storm_lst
            time_steps_since_precip - updated time_steps_since_precip
            is_storming - True or False whether the storm is ongoing or not

    Created March 3, 2017
    @author: Micah Johnson
    """
    # print  "--"*10 +"> Max precip = {0}".format(precipitation.max())
    if precipitation.max() > mass_thresh:
        # Start a new storm
        if len(storm_lst) == 0 or not is_storming:
            storm_lst.append({'start': time, 'end': None})
            is_storming = True

        # always append the most recent timestep to avoid unended storms
        storm_lst[-1]['end'] = time
        time_steps_since_precip = 0

    elif is_storming and time_steps_since_precip < steps_thresh:
        time_steps_since_precip += 1

    if time_steps_since_precip >= steps_thresh:
        is_storming = False
        # print "--"*10 + "> not storming!"

    return storm_lst, time_steps_since_precip, is_storming


def clip_and_correct(precip, storms, stations=[]):
    """
    Meant to go along with the storm tracking, we correct the data here by
    adding in the precip we would miss by ignoring it. This is mostly because
    will get rain on snow events when there is snow because of the storm
    definitions and still try to distribute precip data.

    Args:
        precip: Vector station data representing the measured precipitation
        storms: Storm list with dictionaries as defined in
                :func:`~smrf.envphys.storms.tracking_by_station`
        stations: Desired stations that are being used for clipping. If
                  stations is not passed, then use all in the dataframe


    Returns:
        The correct precip that ensures there is no precip outside of the
        defined storms with the clipped amount of precip proportionally added
        back to storms.

    Created May 3, 2017
    @author: Micah Johnson
    """

    # Specify zeros where were not storming
    precip_clipped = precip.copy()
    precip_clipped[:] = 0

    for j, storm in storms.iterrows():

        storm_start = storm['start']
        storm_end = storm['end']
        my_slice = precip.loc[storm_start:storm_end]
        precip_clipped.loc[storm_start:storm_end] = my_slice

    correction = {}

    if len(stations) == 0:
        stations = precip.columns

    # Correct the precip to be equal to the sum.
    for station in stations:
        original = precip[station].sum()
        clipped = precip_clipped[station].sum()

        if original == 0:
            c = 1.0
        elif clipped == 0:
            c = 0
        else:
            c = original/clipped

        correction[station] = c

    return precip_clipped.mul(pd.Series(correction), axis=1)
