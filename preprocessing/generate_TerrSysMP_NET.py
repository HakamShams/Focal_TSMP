# ------------------------------------------------------------
# Generate TerrSysMP_NET dataset for deep learning
# Hard codded. The script needs optimization
# Tested for TSMP (1989-2019) and NOAA (1989-2019)
# Contact person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
# ------------------------------------------------------------

import numpy as np
import xarray as xr
from cartopy import crs as ccrs

import os
import argparse
import time
import datetime
from tqdm import tqdm

np.set_printoptions(suppress=True)

# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lon', type=int, default=423, help='longitudal number of grid boxes [default: 423]')
    parser.add_argument('--n_lat', type=int, default=411, help='latitudal number of grid boxes [default: 411]')
    parser.add_argument('--d_lon', type=float, default=0.11, help='longitudal resolution (degrees) [default: 0.11]')
    parser.add_argument('--d_lat', type=float, default=0.11, help='latitudal resolution (degrees) [default: 0.11]')
    parser.add_argument('--ll_lon', type=float, default=-28.48, help='lower left longitude (degrees) [default: -28.48]')
    parser.add_argument('--ll_lat', type=float, default=-23.48, help='lower left latitude (degrees) [default: -23.48]')
    parser.add_argument('--pol_lon', type=float, default=-162.0, help='meta pole longitude (degrees) [default: -162.0]')
    parser.add_argument('--pol_lat', type=float, default=39.25, help='meta pole latitude (degrees) [default: 39.25]')

    parser.add_argument('--start_year', default=1989, type=int, help='starting year [default: 1989]')
    parser.add_argument('--end_year', default=2019, type=int, help='ending year [default: 2019]')

    parser.add_argument('--input_path_NOAA', type=str, default=r'../data/NOAA/',
                        help='directory to NOAA dataset [default: ../data/NOAA/]')
    parser.add_argument('--input_path_TerrSysMP', type=str, default='../data/TerrSysMP/',
                        help='directory to TerrSysMP dataset [default: ../data/TerrSysMP/]')
    parser.add_argument('--output_path', type=str, default='../data/TerrSysMP_NET/',
                        help='directory to save generated the dataset [default: ../data/TerrSysMP_NET/]')
    args = parser.parse_args()
    return args

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def create_xarray(n_lon=423, n_lat=411, d_lon=0.11, d_lat=0.11, ll_lon=-28.48, ll_lat=-23.48, pol_lon=-162.0,
                  pol_lat=39.25, days=None):
    """
        Create domain dataset from grid information in the geographic projection (plate carrée projection).
        this function is based on py-cordex implementation see: https://py-cordex.readthedocs.io/en/stable/index.html

        Parameters
        ----------
        n_lon : int
            longitudal number of grid boxes (default is 424)
        n_lat : int
            latitudal number of grid boxes (default is 412)
        d_lon : float
            longitudal resolution (degrees) (default is 0.11)
        d_lat : float
            latitudal resolution (degrees) (default is 0.11)
        ll_lon : float
            lower left longitude (degrees) (default is -28.48)
        ll_lat : float
            lower left latitude (degrees) (default is -23.48)
        pol_lon : float
            meta pole longitude (degrees) (default is -162.0)
        pol_lat : float
            meta pole latitude (degrees) (default is 39.25)
        days : numpy array
            days of the years (time) (default is None)
        Returns
        ----------
        domain:  xarray
    """

    domain = xr.Dataset(
        {
            "rlon": (["rlon"], np.array([round(ll_lon + i * d_lon, 14) for i in range(n_lon)], dtype=np.float32),
                     {"units": "degrees", "standard_name": "grid_longitude",
                      "long_name": "longitude in rotated pole grid", "axis": "X"}),
            "rlat": (["rlat"], np.array([round(ll_lat + i * d_lat, 14) for i in range(n_lat)], dtype=np.float32),
                     {"units": "degrees", "standard_name": "grid_latitude",
                      "long_name": "latitude in rotated pole grid", "axis": "Y"}),
            "time": (["time"], np.array(days, dtype=object),
                     {"standard_name": "time", "long_name": "time", "axis": "T"}),
        }
    )

    pole = ccrs.RotatedPole(pol_lon, pol_lat)
    projection = ccrs.PlateCarree()

    lat_2d, lon_2d = xr.broadcast(domain.rlat, domain.rlon)
    projected = projection.transform_points(pole, lon_2d.values, lat_2d.values)
    lon, lat = projected[:, :, 0], projected[:, :, 1]

    domain = domain.assign_coords(lon=(["rlat", "rlon"], np.ascontiguousarray(lon).astype(np.float32)),
                                  lat=(["rlat", "rlon"], np.ascontiguousarray(lat).astype(np.float32)))

    return domain

def is_leap_year(year):
    """
    Check if the year is a leap year

    https://en.wikipedia.org/wiki/Leap_year
    https://stackoverflow.com/questions/11621740/how-to-determine-whether-a-year-is-a-leap-year

    Parameters
    ----------
    year: int
        the input year to be checked
    Returns
    ----------
    Whether the year is leap or not: Bool
    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_from_week(year, week_n):
    """
    generate days from the corresponding year and week number
    Parameters
    ----------
    year: str
        the input year to generate the days for
    week_n: str
        the input week to generate the days for
    Returns
    ----------
    days: list
        days corresponding to the year and week
    """
    d = datetime.datetime(int(year), 1, 1) + datetime.timedelta(7 * int(week_n) - 7)
    days = []

    for w in range(7):
        days.append((d + datetime.timedelta(days=w)).strftime('%Y-%m-%d'))
    return days


def generate_dataset(args):

    # extract parameters
    args = parse_args()
    start_year = args.start_year
    end_year = args.end_year
    dir_in_NOAA = args.input_path_NOAA
    dir_in_TerrSysMP = args.input_path_TerrSysMP
    dir_out = args.output_path

    # create local directory to store data files
    os.makedirs(dir_out, exist_ok=True)

    # prepare NOAA files
    years = os.listdir(dir_in_NOAA)
    years.sort()

    years = [year for year in years if int(year) in range(start_year, end_year+1, 1)]

    # prepare TerrSysMP files
    terrSysMP_files = os.listdir(dir_in_TerrSysMP)
    terrSysMP_files.sort()

    files = []

    rlat_last = np.round(args.ll_lat + args.d_lat * (args.n_lat - 1), 14)
    rlon_last = np.round(args.ll_lon + args.d_lon * (args.n_lon - 1), 14)
    # variables with different vertical levels (3D)
    feature_layer = ["sgw", "pgw"]

    # variables in TSMP dataset
    variables_CLM = ['evspsbl', 'gh', 'hfls', 'hfss', 'lf', 'prsn', 'prso', 'rlds', 'rs', 'sr', 'tas',
                     'trspsbl']
    variables_PF = ['pgw', 'sgw', 'wtd']
    variables_COSMO = ['awt', 'capec', 'capeml', 'ceiling', 'cli', 'clt', 'clw', 'hudiv', 'hur2', 'hur200',
                       'hur500', 'hur850', 'hus2', 'hus200', 'hus500', 'hus850', 'incml', 'pr', 'prc', 'prg',
                       'prt', 'ps', 'psl', 'snt', 'ta200', 'ta500', 'ta850', 'tch', 'td2', 'ua200', 'ua500',
                       'ua850', 'uas', 'va200', 'va500', 'va850', 'vas', 'zg200', 'zg500', 'zg850', 'zmla']

    for terrSysMP_file in terrSysMP_files:
        dir_file = os.path.join(dir_in_TerrSysMP, terrSysMP_file)
        files_t = os.listdir(dir_file)
        files_t.sort()

        files_t = [file for file in files_t if file.endswith(".nc")]
        files.append(files_t)

    files_dates = []

    for file in files:
        files_t = []
        for file_t in file:
            ind = file_t.rfind("_")
            d1 = datetime.datetime(int(file_t[ind+1:ind+1+4]), int(file_t[ind+1+4:ind+1+4+2]),
                                   int(file_t[ind+1+4+2:ind+1+4+2+2])).strftime('%Y-%m-%d')
            d2 = datetime.datetime(int(file_t[ind+1+4+2+2+1:ind+1+4+2+2+1+4]),
                                   int(file_t[ind+1+4+2+2+1+4:ind+1+4+2+2+1+4+2]),
                                   int(file_t[ind+1+4+2+2+1+4+2:ind+1+4+2+2+1+4+2+2])).strftime('%Y-%m-%d')
            files_t.append((d1, d2))

        files_dates.append(files_t)

    # preprocessing
    print('Generating TerrSysMP_NET Data set...')
    time.sleep(1)

    for year in years:
        dir_year = os.path.join(dir_in_NOAA, year)

        # create local directory to store data files
        dir_out_year = os.path.join(dir_out, year)
        os.makedirs(dir_out_year, exist_ok=True)

        weeks = os.listdir(dir_year)
        weeks = [week for week in weeks if week.endswith('nc')]
        weeks.sort()

        weeks_n = [week[-9:-6] for week in weeks]
        weeks_n_unique = np.unique(weeks_n)
        pbar_w = tqdm(enumerate(weeks_n_unique), total=len(weeks_n_unique), leave=False)

        for w, week_n in pbar_w:

            # the TSMP dataset ends here
            if year == '2019' and week_n == '036':
                break

            pbar_w.set_description('year %s week %s' % (year, week_n), refresh=True)

            # prepare days
            days = days_from_week(year, week_n)

            # create cordex domain
            data_all = create_xarray(args.n_lon, args.n_lat, args.d_lon, args.d_lat,
                                     args.ll_lon, args.ll_lat, args.pol_lon, args.pol_lat, days)

            for f in range(len(terrSysMP_files)):

                feature = terrSysMP_files[f]

                if feature in variables_CLM:
                    model = 'CLM'
                elif feature in variables_COSMO:
                    model = 'COSMO'
                elif feature in variables_PF:
                    model = 'ParFlow'
                else:
                    model = '-'

                dir_feature = os.path.join(dir_in_TerrSysMP, feature)

                file_date = files_dates[f]

                if feature in feature_layer:
                    data_feature_all = np.empty((len(days), 15, args.n_lat, args.n_lon))
                else:
                    data_feature_all = np.empty((len(days), args.n_lat, args.n_lon))

                for k, day in enumerate(days):

                    ind = -1
                    for j in range(len(file_date)):
                        is_between = file_date[j][0] <= day <= file_date[j][1]
                        if is_between:
                            ind = j
                            break

                    if ind >= 0:
                        dir_file = os.path.join(dir_feature, files[f][ind])
                        data_feature = xr.open_dataset(dir_file)

                        try:
                            data_feature.coords['rlon'] = data_feature.coords['rlon'].astype(np.float32)
                            data_feature.coords['rlat'] = data_feature.coords['rlat'].astype(np.float32)

                        except:
                            data_feature = data_feature.rename({'lon': 'rlon', 'lat': 'rlat'})
                            data_feature.coords['rlon'] = data_feature.coords['rlon'].astype(np.float32)
                            data_feature.coords['rlat'] = data_feature.coords['rlat'].astype(np.float32)

                        if day[-5:] == "02-29":
                            try:
                                data_feature = data_feature.sel(time=day, rlat=slice(args.ll_lat-0.02, rlat_last+0.02),
                                                                rlon=slice(args.ll_lon-0.02, rlon_last+0.02))
                            except:
                                data_feature_all[k, :, :] = np.nan
                                continue
                        else:
                            try:
                                data_feature = data_feature.sel(time=day, rlat=slice(args.ll_lat - 0.02, rlat_last + 0.02),
                                                                rlon=slice(args.ll_lon - 0.02, rlon_last + 0.02))
                            except:
                                data_feature_all[k, :, :] = np.nan
                                continue

                        if len(data_feature["time"]) < 1:
                            day_t = datetime.datetime(int(day[:4]), int(day[5:7]), int(day[-2:])) + datetime.timedelta(1)
                            day_t = day_t.strftime('%Y-%m-%d')

                            ind = -1
                            for j in range(len(file_date)):
                                is_between = file_date[j][0] <= day_t <= file_date[j][1]
                                if is_between:
                                    ind = j
                                    break
                            if ind >= 0:
                                dir_file = os.path.join(dir_feature, files[f][ind])
                                data_feature = xr.open_dataset(dir_file)
                                data_feature.coords['rlon'] = data_feature.coords['rlon'].astype(np.float32)
                                data_feature.coords['rlat'] = data_feature.coords['rlat'].astype(np.float32)

                                data_feature = data_feature.sel(time=day_t,
                                                                rlat=slice(args.ll_lat - 0.02, rlat_last + 0.02),
                                                                rlon=slice(args.ll_lon - 0.02, rlon_last + 0.02))
                                data_feature = data_feature.isel(time=0)
                            else:
                                data_feature_all[k, :, :] = np.nan
                                continue

                        elif len(data_feature["time"]) > 1:
                            data_feature = data_feature.isel(time=1)

                        data = data_feature[feature].values
                        data_feature_all[k, :, :] = data

                    else:
                        data_feature_all[k, :, :] = np.nan

                if feature == 'gh' or feature == 'lf' or feature == 'rs' or feature == 'sr':
                    if int(year) < 1996:
                        data_feature = xr.Dataset(attrs=dict(long_name='',
                                                             units='watt/m^2',
                                                             grid_mapping='rotated_pole',
                                                             cell_method='time: mean'))

                        data_feature[feature] = (["time", "rlat", "rlon"], data_feature_all, data_feature.attrs)
                    if feature == 'gh':
                        data_feature[feature] = data_feature[feature].assign_attrs(long_name='ground_heat_flux')
                    elif feature == 'lf':
                        data_feature[feature] = data_feature[feature].assign_attrs(long_name='net_longwave_radiation')
                    elif feature == 'rs':
                        data_feature[feature] = data_feature[feature].assign_attrs(long_name='reflected_shortwave_radiation')
                    elif feature == 'sr':
                        data_feature[feature] = data_feature[feature].assign_attrs(long_name='incoming_shortwave_radiation')

                data_feature_all = data_feature_all.astype(np.float32)

                if feature in feature_layer:
                    data_feature_all[data_feature_all < -1000] = np.nan

                    data_all[feature] = (["time", "lev", "rlat", "rlon"], data_feature_all, data_feature[feature].attrs)
                else:
                    if feature == 'clt':
                        data_feature_all[data_feature_all < 0] = 0.0

                    data_all[feature] = (["time", "rlat", "rlon"], data_feature_all, data_feature[feature].attrs)

                data_all[feature] = data_all[feature].assign_attrs(model=model)
                if feature == 'zmla':
                    data_all[feature] = data_all[feature].assign_attrs(long_name='height_of_boundary_layer')

            # open NOAA data
            indices = []
            for idx, value in enumerate(weeks_n):
                if value == week_n:
                    indices.append(idx)

            n_satellite = len(indices)

            # check if there is just one satellite

            if n_satellite == 1:

                week = weeks[indices[0]]

                dir_week = os.path.join(dir_year, week)

                data_noaa = xr.open_dataset(dir_week)

                SMN = data_noaa['SMN'].sel(rlat=slice(args.ll_lat - 0.02, rlat_last + 0.02),
                                           rlon=slice(args.ll_lon - 0.02, rlon_last + 0.02))
                SMT = data_noaa['SMT'].sel(rlat=slice(args.ll_lat - 0.02, rlat_last + 0.02),
                                           rlon=slice(args.ll_lon - 0.02, rlon_last + 0.02))

                mask_cold = data_noaa['cold_surface'].sel(rlat=slice(args.ll_lat - 0.02, rlat_last + 0.02),
                                                          rlon=slice(args.ll_lon - 0.02, rlon_last + 0.02))

                data_all['SMN'] = (["rlat", "rlon"], SMN.values.astype(np.float32), SMN.attrs)
                data_all['SMT'] = (["rlat", "rlon"], SMT.values.astype(np.float32), SMT.attrs)
                data_all['mask_cold_surface'] = (["rlat", "rlon"], mask_cold.values.astype(np.uint8), mask_cold.attrs)

            else:

                SMN = np.empty((n_satellite, args.n_lat, args.n_lon))
                SMT = np.empty((n_satellite, args.n_lat, args.n_lon))

                mask_cold = np.empty((n_satellite, args.n_lat, args.n_lon))
                SMN_attr, SMT_attr, mask_cold_attr = [], [], []

                for ww, week_ind in enumerate(indices):
                    week = weeks[week_ind]

                    dir_week = os.path.join(dir_year, week)

                    data_noaa = xr.open_dataset(dir_week)
                    SMN_ww = data_noaa['SMN'].sel(rlat=slice(args.ll_lat - 0.02, rlat_last + 0.02),
                                                  rlon=slice(args.ll_lon - 0.02, rlon_last + 0.02))
                    SMT_ww = data_noaa['SMT'].sel(rlat=slice(args.ll_lat - 0.02, rlat_last + 0.02),
                                                  rlon=slice(args.ll_lon - 0.02, rlon_last + 0.02))

                    mask_cold_ww = data_noaa['cold_surface'].sel(rlat=slice(args.ll_lat - 0.02, rlat_last + 0.02),
                                                                 rlon=slice(args.ll_lon - 0.02, rlon_last + 0.02))

                    SMN_attr.append(SMN_ww.attrs)
                    SMT_attr.append(SMN_ww.attrs)
                    mask_cold_attr.append(mask_cold_ww.attrs)

                    SMN[ww, :, :] = SMN_ww.values
                    SMT[ww, :, :] = SMT_ww.values
                    mask_cold[ww, :, :] = mask_cold_ww.values

                data_all['SMN'] = (["satellite", "rlat", "rlon"], SMN.astype(np.float32))
                data_all['SMT'] = (["satellite", "rlat", "rlon"], SMT.astype(np.float32))
                data_all['mask_cold_surface'] = (["satellite", "rlat", "rlon"], mask_cold.astype(np.uint8))

                for k in range(n_satellite):
                    data_all['SMN'].isel(satellite=k).attrs= SMN_attr[k]
                    data_all['SMT'].isel(satellite=k).attrs= SMT_attr[k]
                    data_all['mask_cold_surface'].isel(satellite=k).attrs= mask_cold_attr[k]


            data_all = data_all.assign_attrs(convention="CF-1.4",
                                             conventionsURL="http://www.cfconventions.org/",
                                             creation_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                             CORDEX_domain="EUR-11",
                                             TerrSysMP_frequency="daily",
                                             NOAA_frequency="weekly",
                                             temporal_extent=str(days[0]) + "-" + str(days[-1]),
                                             spatial_resolution="0.11 deg ~12.5 km",
                                             longitudinal_extent=str(round(args.ll_lon, 2)) + "-" + str(round(rlon_last,
                                                                                                              2)),
                                             latitudinal_extent=str(round(args.ll_lat, 2)) + "-" + str(round(rlat_last,
                                                                                                             2)),
                                             grid_mapping="rotated_rotated_pole",
                                             rotated_pole_latitude=39.25,
                                             rotated_pole_longitude=-162.0,
                                             TerrSysMP_provider="FZJ, Jülich Research Centre",
                                             VHI_data_provider="NOAA/NESDIS NOAA Center for Satellite Applications and Research",
                                             )

            dir_out_week = os.path.join(dir_out_year, year+week_n+'.nc')
            data_all.to_netcdf(dir_out_week)


if __name__ == '__main__':

    args = parse_args()
    generate_dataset(args)

