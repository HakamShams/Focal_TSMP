# ------------------------------------------------------------------
# Dataset class for TerrSysMP_NET
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>
# ------------------------------------------------------------------

import numpy as np
import xarray as xr
import os
import json
import torch
from torch.utils.data import Dataset
import warnings

np.set_printoptions(suppress=True)

# ------------------------------------------------------------------

class TerrSysMP_NET(Dataset):
    """
    Dataloader for TerrSysMP_NET Dataset

    Attributes
    ----------
    root : str
        directory of the dataset
    nan_fill : float
        value to replace missing values
    is_aug : bool
        option to use data augmentation
    is_shuffle : bool
        option to shuffle the data
    variables : list
        TSMP variable names wihtout vertical levels
    variables_tsmp : list
        TSMP variable names with vertical levels
    variables_static : list
        static variable names
    years : list
        year used to load the data
    n_lon : int
        longitudal number of grid boxes
    n_lat : int
        latitudal number of grid boxes
    cut_lateral_boundary : int
        number of pixels to be excluded from the lateral boundary relaxation zone
    n_lon_cut : int
        longitudal number of grid boxes after excluding the lateral boundary relaxation zone
    n_lat_cut : int
        latitudal number of grid boxes after excluding the lateral boundary relaxation zone
    var_n_tsmp : int
        number of all tsmp variables with vertical levels
    var_n_static : int
        number of static variables
    var_n : int
        number of all tsmp with vertical levels and static variables
    files : list
        files in the dataset

    Methods
    -------
    __get_var_n(variables_models)
        private method to get the number of variables
    __getpath(root)
        private method to get the files of the dataset inside the root directory

    __load_tsmp_statistic(root)
        private method to get the statistics of the TSMP dataset from the root directory
    __load_climatology_statistic(root)
        private method to load the climatology from the root directory
    __load_valid_pixels_mask(root)
        private method to load the land/sea and non vegetation masks from the root directory
    __load_static_variables(root)
        private method to load the static variables from the root directory

    __generate_mask(NetCDF_file)
        private method to load NOAA data from the file path
     __load_datacube(NetCDF_file)
        private method to load TSMP data from the file path

    min_max_scale(array, min_alt, max_alt, min_new, max_new)
        helper method to normalize an array between new minimum and maximum values
    get_datacube_year_week(index)
        helper method to get the year and week number from the file index

    __getitem__(index)
        method to load datacube by the file index
    __len__()
        method to get the number of files
    """

    def __init__(self, root: str, nan_fill: float = 0., is_aug: bool = False, is_shuffle: bool = False,
                 variables: list = None, variables_static: list = None, years: list = None,
                 n_lat: int = 411, n_lon: int = 423, cut_lateral_boundary: int = 7):

        super().__init__()

        """
        Parameters
        ----------
        root : str
            directory of the dataset. It should also contain the necessary files for static and climatology data
        nan_fill : float (default 0.)
            value to replace missing values
        is_aug : bool (default False)
            option to use data augmentation
        is_shuffle : bool (default False)
            option to shuffle the data
        variables : list (default None)
            list of TSMP variable names
        variables_static : list (default None)
            list of static variable names
        years : list (default None)
            years used to load the data
        n_lon : int (default 423)
            longitudal number of grid boxes
        n_lat : int (default 411)
            latitudal number of grid boxes
        cut_lateral_boundary : int (default 7)
            number of pixels to be excluded from the lateral boundary relaxation zone
        """

        self.root = root
        self.nan_fill = nan_fill
        self.is_aug = is_aug
        self.is_shuffle = is_shuffle
        self.variables = variables
        self.variables_static = variables_static

        self.years = years

        self.n_lon = n_lon
        self.n_lat = n_lat
        self.cut_lateral_boundary = cut_lateral_boundary

        self.n_lon_cut = n_lon - cut_lateral_boundary * 2
        self.n_lat_cut = n_lat - cut_lateral_boundary * 2

        # preprocessing for the dataset
        self.__get_path(root, years)
        self.__load_tsmp_statistic(root)
        self.__load_climatology_statistic(root)
        self.__load_valid_pixels_mask(root)
        self.__load_static_variables(root)
        self.__get_var_n(variables)

        if is_shuffle:
            np.random.shuffle(self.files)

    def __get_var_n(self, variables_models):
        """
        Private method to get the number of variables

        Parameters
        ----------
        variables_models : list
            list of TSMP variable names
        """
        self.var_n_tsmp = len(variables_models)

        if 'sgw' in variables_models:
            self.var_n_tsmp += 14
        if "pgw" in variables_models:
            self.var_n_tsmp += 14

        self.var_n = self.var_n_tsmp + self.var_n_static

    def __get_path(self, root_path, years):
        """
        Private method to get the files of the dataset inside the root directory

        Parameters
        ----------
        root_path : str
            directory of the dataset. It should also contain the necessary files for static and climatology data
        years : list
            years used to load the data
        """

        self.files = []

        years_files = os.listdir(root_path)
        years_files = [year_file for year_file in years_files if
                       not year_file.endswith('json') and not year_file.endswith('nc')]
        years_files = [year_file for year_file in years_files if year_file in years]
        years_files.sort()

        for year_file in years_files:
            year_file_path = os.path.join(root_path, year_file)
            files = os.listdir(year_file_path)
            files.sort()

            for file in files:
                if not file.endswith('nc'):
                    continue
                file_path = os.path.join(year_file_path, file)

                self.files.append(file_path)

    def __load_tsmp_statistic(self, root_path):
        """
        Private method to get the statistics of the TSMP dataset from the root directory

        Parameters
        ----------
        root_path : str
            directory of the dataset. It should also contain the necessary files for static and climatology data
        """

        with open(os.path.join(root_path, 'mean_std_train.json'), 'r') as file:

            dict = json.load(file)

            self.__min_variables, self.__max_variables, self.__mean_variables, self.__std_variables = [], [], [], []
            self.variable_tsmp = self.variables.copy()

            if 'pgw' in self.variable_tsmp:
                variables_layers = ['pgw_' + str(l + 1) for l in range(15)]
                ind = self.variable_tsmp.index('pgw')
                self.variable_tsmp.remove('pgw')
                for ind_v in range(15):
                    self.variable_tsmp.insert(ind + ind_v, variables_layers[ind_v])

            if 'sgw' in self.variable_tsmp:
                variables_layers = ['sgw_' + str(l + 1) for l in range(15)]
                ind = self.variable_tsmp.index('sgw')
                self.variable_tsmp.remove('sgw')
                for ind_v in range(15):
                    self.variable_tsmp.insert(ind + ind_v, variables_layers[ind_v])

            for v in self.variable_tsmp:
                self.__min_variables.append(float(dict['min'][v]))
                self.__max_variables.append(float(dict['max'][v]))
                self.__mean_variables.append(float(dict['mean'][v]))
                self.__std_variables.append(float(dict['std'][v]))

    def __load_climatology_statistic(self, root_path):
        """
        Private method to load the climatology from the root directory

        Parameters
        ----------
        root_path : str
            directory of the dataset. It should also contain the necessary files for static and climatology data
        """

        dataset_climatology = xr.load_dataset(os.path.join(root_path, "climatology_1989_2016.nc"))

        self.__datacube_climatology = np.concatenate((dataset_climatology['SMN'].values,
                                                      dataset_climatology['SMT'].values), axis=1)

        self.__datacube_climatology = self.__datacube_climatology[:, :,
                                    self.cut_lateral_boundary:-self.cut_lateral_boundary,
                                    self.cut_lateral_boundary:-self.cut_lateral_boundary]

        self.__datacube_climatology = np.flip(self.__datacube_climatology, axis=-2).astype(np.float32)

    def __load_valid_pixels_mask(self, root_path):
        """
        Private method to load the land/sea and non vegetation masks from the root directory

        Parameters
        ----------
        root_path : str
            directory of the dataset. It should also contain the necessary files for static and climatology data
        """

        dataset_valid_pixels = xr.load_dataset(os.path.join(root_path, "mask_valid_pixels.nc"))
        self.__mask_sea = dataset_valid_pixels['mask_sea'].values[
                        self.cut_lateral_boundary:-self.cut_lateral_boundary,
                        self.cut_lateral_boundary:-self.cut_lateral_boundary]
        self.__mask_no_vegetation = dataset_valid_pixels['mask_no_vegetation'].values[
                                  self.cut_lateral_boundary:-self.cut_lateral_boundary,
                                  self.cut_lateral_boundary:-self.cut_lateral_boundary]

        self.__mask_sea = np.flip(self.__mask_sea, axis=-2).astype(np.float32)
        self.__mask_no_vegetation = np.flip(self.__mask_no_vegetation, axis=-2).astype(np.float32)

    def __load_static_variables(self, root_path):
        """
        Private method to load the static variables from the root directory

        Parameters
        ----------
        root_path : str
            directory of the dataset. It should also contain the necessary files for static and climatology data
        """

        dataset_static = xr.load_dataset(os.path.join(root_path, "static_variables.nc"))[self.variables_static]

        #min_variables_static, max_variables_static = np.empty(len(self.variables_static)), \
        #                                                       np.empty(len(self.variables_static))
        mean_variables_static, std_variables_static = np.empty(len(self.variables_static)), \
                                                      np.empty(len(self.variables_static))

        for k, v in enumerate(self.variables_static):
            # min_variables_static[k] = dataset_static[v].min(skipna=True).values
            # max_variables_static[k] = dataset_static[v].max(skipna=True).values
            mean_variables_static[k] = dataset_static[v].mean(skipna=True).values
            std_variables_static[k] = dataset_static[v].std(skipna=True).values

        self.var_n_static = len(self.variables_static)
        self.__datacube_static = dataset_static.to_array().values

        self.__datacube_static = (self.__datacube_static - mean_variables_static[:, None, None])\
                                 / std_variables_static[:, None, None]

        self.__datacube_static = self.__datacube_static[:,
                                 self.cut_lateral_boundary:-self.cut_lateral_boundary,
                                 self.cut_lateral_boundary:-self.cut_lateral_boundary]

        self.__datacube_static[np.isnan(self.__datacube_static)] = self.nan_fill
        self.__datacube_static = np.flip(self.__datacube_static, axis=-2).astype(np.float32)

    def min_max_scale(self, array: np.array,
                      min_alt: float, max_alt: float,
                      min_new: float = 0., max_new: float = 1.):

        """
        Helper method to normalize an array between new minimum and maximum values

        Parameters
        ----------
        array : numpy array
            array to be normalized
        min_alt : float
            minimum value in array
        max_alt : float
            maximum value in array
        min_new : float
            minimum value after normalization
        max_new : float
            maximum value after normalization

        Returns
        ----------
        array : numpy array
            normalized numpy array
        """

        array = ((max_new - min_new) * (array - min_alt) / (max_alt - min_alt)) + min_new

        return array

    def get_datacube_year_week(self, index):
        """
        Helper method to get the year and week number from the file index

        Parameters
        ----------
        index : int
          index of the file in the dataset

        Returns
        ----------
        year : str
            corresponding year of the indexed file
        week : str
            corresponding week number of the indexed file
        """

        file_name = os.path.splitext(os.path.basename(os.path.normpath(self.files[index])))[0]

        year = file_name[:4]
        week = file_name[4:7]

        return year, week

    def __generate_mask(self, NetCDF_file):
        """
        Private method to load NOAA data from the file path

        Parameters
        ----------
        NetCDF_file : str
           the file path

        Returns
        ----------
        Netcdf_smn : numpy array [n_lat, n_lon]
            SMN (smoothed NDVI)
        Netcdf_smt : numpy array [n_lat, n_lon]
            SMT (smoothed BT)
        Netcdf_cold_surface : numpy array [n_lat, n_lon]
            mask of cold surfaces (i.e., snow)
        """
        Netcdf_smn = xr.load_dataset(NetCDF_file)['SMN'].values
        Netcdf_smt = xr.load_dataset(NetCDF_file)['SMT'].values
        Netcdf_cold_surface = xr.load_dataset(NetCDF_file)['mask_cold_surface'].values

        if Netcdf_smn.ndim > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Netcdf_smn = np.nanmean(Netcdf_smn, axis=0, keepdims=False)
                Netcdf_smt = np.nanmean(Netcdf_smt, axis=0, keepdims=False)

            Netcdf_cold_surface = Netcdf_cold_surface.sum(axis=0)
            Netcdf_cold_surface[Netcdf_cold_surface > 1] = 1

        return Netcdf_smn, Netcdf_smt, Netcdf_cold_surface


    def __load_datacube(self, NetCDF_file):
        """
        Private method to load TSMP data from the file path

        Parameters
        ----------
        NetCDF_file : str
         the file path

        Returns
        ----------
        datacube : numpy array [var_n_tsmp, n_lat, n_lon]
            TSMP data
        """

        if 'sgw' not in self.variables and 'pgw' not in self.variables:
            datacube = xr.load_dataset(NetCDF_file)[self.variables].to_array().values

        elif 'sgw' in self.variables and 'pgw' not in self.variables:
            datacube = xr.load_dataset(NetCDF_file)[self.variables].to_array().values
            if datacube.shape[2] != self.n_lat:
                datacube = np.moveaxis(datacube, 2, -1)
            datacube = np.concatenate((datacube[:self.variables.index('sgw'), :, :, :, 0],
                                       np.moveaxis(datacube[self.variables.index('sgw'), :, :, :, :], -1, 0),
                                       datacube[self.variables.index('sgw') + 1:, :, :, :, 0]), axis=0)

        elif 'pgw' in self.variables and 'sgw' not in self.variables:
            datacube = xr.load_dataset(NetCDF_file)[self.variables].to_array().values
            if datacube.shape[2] == self.n_lat:
                datacube = np.moveaxis(datacube, 2, -1)
            datacube = np.concatenate((datacube[:self.variables.index('pgw'), :, :, :, 0],
                                       np.moveaxis(datacube[self.variables.index('pgw'), :, :, :, :], -1, 0),
                                       datacube[self.variables.index('pgw') + 1:, :, :, :, 0]), axis=0)

        elif 'pgw' in self.variables and 'sgw' in self.variables:

            datacube = xr.load_dataset(NetCDF_file)[self.variables].to_array().values
            if datacube.shape[2] != self.n_lat:
                datacube = np.moveaxis(datacube, 2, -1)
            datacube = np.concatenate((datacube[:self.variables.index('pgw'), :, :, :, 0],
                                       np.moveaxis(datacube[self.variables.index('pgw'), :, :, :, :], -1, 0),
                                       datacube[self.variables.index('pgw') + 1:self.variables.index('sgw'), :, :, :, 0],
                                       np.moveaxis(datacube[self.variables.index('sgw'), :, :, :, :], -1, 0),
                                       datacube[self.variables.index('sgw') + 1:, :, :, :, 0]), axis=0)

        if self.is_aug:
            ind = np.random.choice(7, size=2, replace=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                datacube = np.nanmean(datacube[:, ind, :, :], axis=1, keepdims=False)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                datacube = np.nanmean(datacube, axis=1, keepdims=False)

        return datacube


    def __getitem__(self, index):
        """
        Method to load datacube by the file index

        Parameters
        ----------
        index : int
          The index of the file

        Returns
        ----------
        datacube : numpy array [var_n_tsmp + var_n_static, n_lat_cut, n_lon_cut]
            TSMP data + static data
        datacube_smn : numpy array [n_lat_cut, n_lon_cut]
            SMN (smoothed NDVI)
        datacube_smt : numpy array [n_lat_cut, n_lon_cut]
            SMT (smoothed BT)
        datacube_climatology : numpy array [10, n_lat_cut, n_lon_cut]
            climatology for SMN and SMT
        datacube_cold_surface : numpy array [n_lat_cut, n_lon_cut]
            mask of cold surfaces (i.e., snow)
        datacube_mask_sea : numpy array [n_lat_cut, n_lon_cut]
            mask for land/sea pixels
        datacube_mask_no_vegetation : numpy array [n_lat_cut, n_lon_cut]
            mask for non vegetation
        """

        # get the file to be loaded
        file = self.files[index]
        # get the year and week of the sample
        _, week = self.get_datacube_year_week(index)
        # load NOAA data from the file
        datacube_smn, datacube_smt, datacube_cold_surface = self.__generate_mask(file)
        # load the TSMP data from the file
        datacube = self.__load_datacube(file)
        # get climatology of the corresponding file
        datacube_climatology = self.__datacube_climatology[int(week) - 1, :, :, :]
        # get static variables
        datacube_static = self.__datacube_static.copy()
        # get land/sea and non_vegetation masks
        datacube_mask_sea, datacube_mask_no_vegetation = self.__mask_sea.copy(), self.__mask_no_vegetation.copy()

        # normalize TSMP data
        for v in range(self.var_n_tsmp):
            datacube[v, :, :] = (datacube[v, :, :] - self.__mean_variables[v]) / self.__std_variables[v]
        #  datacube[v, :, :] = self.min_max_scale(datacube[v, :, :],
        #                                              self.__min_variables[v],
        #                                              self.__max_variables[v],
        #                                              -1, 1)
        #  datacube[v, :, :] = np.log(datacube[v, :, :] + self.__min_variables[v] + 1)

        # fill in the missing data
        datacube[np.logical_or(np.isnan(datacube), np.isinf(datacube))] = self.nan_fill

        datacube = np.flip(datacube, axis=-2).astype(np.float32)[:,
                   self.cut_lateral_boundary:-self.cut_lateral_boundary,
                   self.cut_lateral_boundary:-self.cut_lateral_boundary]

        datacube = np.concatenate((datacube, datacube_static), axis=0)

        datacube_smn = np.flip(datacube_smn, axis=-2).astype(np.float32)[
                   self.cut_lateral_boundary:-self.cut_lateral_boundary,
                   self.cut_lateral_boundary:-self.cut_lateral_boundary]
        datacube_smt = np.flip(datacube_smt, axis=-2).astype(np.float32)[
                   self.cut_lateral_boundary:-self.cut_lateral_boundary,
                   self.cut_lateral_boundary:-self.cut_lateral_boundary]
        datacube_cold_surface = np.flip(datacube_cold_surface, axis=-2).astype(np.float32)[
                                self.cut_lateral_boundary:-self.cut_lateral_boundary,
                                self.cut_lateral_boundary:-self.cut_lateral_boundary]

        # augmentation
        if self.is_aug:
            is_rotate = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])
            if is_rotate:
                #k = np.random.randint(1, 4)
                k = 2
                datacube = np.rot90(datacube, k=k, axes=(-1, -2))
                datacube_smn = np.rot90(datacube_smn, k=k, axes=(-1, -2))
                datacube_smt = np.rot90(datacube_smt, k=k, axes=(-1, -2))
                datacube_cold_surface = np.rot90(datacube_cold_surface, k=k, axes=(-1, -2))
                datacube_climatology = np.rot90(datacube_climatology, k=k, axes=(-1, -2))
                datacube_mask_sea = np.rot90(datacube_mask_sea, k=k, axes=(-1, -2))
                datacube_mask_no_vegetation = np.rot90(datacube_mask_no_vegetation, k=k, axes=(-1, -2))

            is_flip = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])
            if is_flip:
                ax = np.random.randint(1, 2)
                datacube = np.flip(datacube, axis=-ax)
                datacube_smn = np.flip(datacube_smn, axis=-ax)
                datacube_smt = np.flip(datacube_smt, axis=-ax)
                datacube_cold_surface = np.flip(datacube_cold_surface, axis=-ax)
                datacube_climatology = np.flip(datacube_climatology, axis=-ax)
                datacube_mask_sea = np.flip(datacube_mask_sea, axis=-ax)
                datacube_mask_no_vegetation = np.flip(datacube_mask_no_vegetation, axis=-ax)

            is_noise = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])
            if is_noise:
                datacube = datacube + np.random.randn(self.var_n, self.n_lat_cut, self.n_lon_cut).astype(
                    np.float32) * 0.01  # 0.002

        return datacube.copy(), datacube_smn.copy(), datacube_smt.copy(), datacube_climatology.copy(), \
               datacube_cold_surface.copy(), datacube_mask_sea.copy(), datacube_mask_no_vegetation.copy()

    def __len__(self):
        """
        Method to get the number of files
        """
        return len(self.files)


if __name__ == '__main__':

    root = r'./data/TerrSysMP_NET'

    variables_static = ['FIS', 'FR_LAND', 'HSURF', 'ZBOT', 'DIS_SEA', 'ROUGHNESS', 'SLOPE']
    variables = [
        'awt',
        'capec',
        'capeml',
        'ceiling',
        'cli',
        'clt',
        'clw',
        'evspsbl',
        # 'gh',
        'hfls',
        'hfss',
        'hudiv',
        'hur2',
        'hur200',
        'hur500',
        'hur850',
        'hus2',
        'hus200',
        'hus500',
        'hus850',
        'incml',
        #  'lf',
        #  'pgw',
        'pr',
        'prc',
        'prg',
        'prsn',
        'prso',
        'prt',
        'ps',
        'psl',
        'rlds',
        #  'rs',
        'sgw',
        'snt',
        # 'sr',
        'ta200',
        'ta500',
        'ta850',
        'tas',
        'tch',
        'td2',
        'trspsbl',
        'ua200',
        'ua500',
        'ua850',
        'uas',
        'va200',
        'va500',
        'va850',
        'vas',
        'wtd',
        'zg200',
        'zg500',
        'zg850',
        'zmla',
    ]

    years_train = [str(year) for year in range(1989, 2019)]

    dataset = TerrSysMP_NET(root=root, nan_fill=0.0, is_aug=False, is_shuffle=False,
                            variables=variables, variables_static=variables_static, years=years_train,
                            n_lat=411, n_lon=423, cut_lateral_boundary=7)

    print('number of sampled data:', dataset.__len__())

    is_test_run = False
    is_test_plot = True

    if is_test_run:

        import time
        import random

        manual_seed = 0
        random.seed(manual_seed)

        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,
                                                   pin_memory=True, num_workers=1)

        end = time.time()

        for i, (data_d, data_smn, data_smt, data_climatology,
                data_cold_surface, data_sea, data_no_vegetation) in enumerate(train_loader):
            print("sample: ", i)
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()

    if is_test_plot:

        #import matplotlib
        #matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        variable_all = dataset.variable_tsmp + dataset.variables_static

        for i in range(len(dataset)):

            # i = np.random.choice(len(dataset), 1, replace=False)

            data, data_smn, data_smt, data_climatology, \
            data_cold, data_sea, data_no_vegetation = dataset[int(i)]

            for v in range(data.shape[0]):

                plt.imshow(data[v, :, :])
                plt.title(variable_all[v])
                plt.colorbar()
                plt.show()

            break

