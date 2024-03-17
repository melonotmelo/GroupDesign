import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from others.data_information import get_type_from_filename


class BaseDataset(Dataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        self.steps = [
            1,
        ]
        self.hdf5_file_path = hdf5_file_path
        self.num_t = num_t
        self.x_size = x_size
        self.split = split
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        self._init_data()
        self._make_info()
        self.len_per_N = [self.T - step * (self.num_t - 1) for step in self.steps if self.T > step * (self.num_t - 1)]
        self.len = self.len_N * sum(self.len_per_N)
        self.each_step_length = [self.len_N * step for step in self.len_per_N]
        self.step_offset = [0] + [sum(self.each_step_length[:i + 1]) for i in range(len(self.each_step_length))]
        print(f"dataset info: {self.hdf5_file_path}, total len:{self.len}")

    def _init_data(self):
        raise NotImplementedError()

    def _make_info(self):
        """
        make info about this dataset ,
        len_N is the length of N and N_start is which N to start
        Returns:

        """
        raise NotImplementedError()

    def _get_example(self, n, t, x_size):
        raise NotImplementedError()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        step_idx = np.searchsorted(self.step_offset, idx, side='right') - 1
        n = (idx - self.step_offset[step_idx]) // self.len_per_N[step_idx] + self.N_start
        t = (idx - self.step_offset[step_idx]) % self.len_per_N[step_idx]
        physics = []
        # 采样num_t个时间步的样本最后concat到一起，每两个时间步之间的时间单位在self.steps中指出
        for i in range(self.num_t):
            data = self._get_example(n, t + self.steps[step_idx] * i, self.x_size)
            physics.append(data)
        physics = np.stack(physics, axis=0)
        physics = torch.tensor(physics)
        return physics, get_type_from_filename(self.hdf5_file_path)


# for 1D advection and 1D diffusion reaction
class Advection_ReacDiff_Dataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        self.data = self.hdf5_file['tensor']

    def _make_info(self):
        # basic info
        self.dim = 1
        if len(self.data.shape) == 4:
            self.N = self.data.shape[0] * self.data.shape[1]
            self.T = self.data.shape[2]
            self.X = self.data.shape[3]
        else:
            self.N = self.data.shape[0]
            self.T = self.data.shape[1]
            self.X = self.data.shape[2]
        # length of N and which N to start
        if self.split == 'train':
            self.len_N = self.N - 1
            self.N_start = 0
        else:
            self.len_N = 1
            self.N_start = self.N - 1

    def _get_example(self, n, t, x_size):
        if x_size is None:
            data = np.zeros([11, self.X])
            if len(self.data.shape) == 4:
                data[4, :] = self.data[n % 2, n / 2, t, :]
            else:
                data[4, :] = self.data[n, t, :]
            return data
        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size])
        if len(self.data.shape) == 4:
            data[4, :] = self.data[n % 2, n / 2, t, ::(self.X // x_size)]
        else:
            data[4, :] = self.data[n, t, ::(self.X // x_size)]
        return data


class BurgersDataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        self.data = self.hdf5_file['tensor']

    def _make_info(self):
        # basic info
        self.dim = 1
        self.N = self.data.shape[0]
        self.T = self.data.shape[1]
        self.X = self.data.shape[2]
        # length of N and which N to start
        if self.split == 'train':
            self.len_N = self.N - 1
            self.N_start = 0
        else:
            self.len_N = 1
            self.N_start = self.N - 1

    def _get_example(self, n, t, x_size):
        if x_size is None:
            data = np.zeros([11, self.X])
            data[0, :] = self.data[n, t, :]
            return data
        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size])
        data[0, :] = self.data[n, t, ::(self.X // x_size)]
        return data


class CFD1D_Dataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        self.Vx = self.hdf5_file['Vx']
        self.density = self.hdf5_file['density']
        self.pressure = self.hdf5_file['pressure']

    def _make_info(self):
        # basic info
        self.dim = 1
        self.N = self.density.shape[0]
        self.T = self.density.shape[1]
        self.X = self.density.shape[2]
        # length of N and which N to start
        if self.split == 'train':
            self.len_N = self.N - 1
            self.N_start = 0
        else:
            self.len_N = 1
            self.N_start = self.N - 1

    def _get_example(self, n, t, x_size):
        vx = self.Vx[n, t, :]
        density = self.density[n, t, :]
        pressure = self.pressure[n, t, :]
        if x_size is None:
            data = np.zeros([11, self.X])
            data[0, :] = vx
            data[3, :] = pressure
            data[4, :] = density
            return data
        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size])
        data[0, :] = vx[::(self.X // x_size)]
        data[3, :] = pressure[::(self.X // x_size)]
        data[4, :] = density[::(self.X // x_size)]
        return data


class DiffSorpDataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        pass

    def _make_info(self):
        # basic info
        self.dim = 1
        self.N = len(self.hdf5_file.keys())
        self.T = self.hdf5_file[list(self.hdf5_file.keys())[0]]['data'].shape[0]
        self.X = self.hdf5_file[list(self.hdf5_file.keys())[0]]['data'].shape[1]
        # length of N and which N to start
        if self.split == 'train':
            self.len_N = self.N - 1
            self.N_start = 0
        else:
            self.len_N = 1
            self.N_start = self.N - 1

    def _get_example(self, n, t, x_size):
        key = list(self.hdf5_file.keys())[n]
        if x_size is None:
            data = np.zeros([11, self.X])
            data[4, :] = self.hdf5_file[key]['data'][t, ..., 0]
            return data
        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size])
        data[4, :] = self.hdf5_file[key]['data'][t, ..., 0][::(self.X // x_size)]
        return data


class Rdb2D_Dataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        pass

    def _make_info(self):
        # basic info
        self.dim = 2
        self.N = len(self.hdf5_file.keys())
        self.T = self.hdf5_file[list(self.hdf5_file.keys())[0]]['data'].shape[0]
        self.X = self.hdf5_file[list(self.hdf5_file.keys())[0]]['data'].shape[1]
        # length of N and which N to start
        if self.split == 'train':
            self.len_N = self.N - 1
            self.N_start = 0
        else:
            self.len_N = 1
            self.N_start = self.N - 1

    def _get_example(self, n, t, x_size):
        key = list(self.hdf5_file.keys())[n]
        if x_size is None:
            data = np.zeros([11, self.X, self.X])
            data[8, ...] = self.hdf5_file[key]['data'][t, ..., 0]
            return data
        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size, x_size])
        data[8, ...] = self.hdf5_file[key]['data'][t, ..., 0][::(self.X // x_size), ::(self.X // x_size)]
        return data


class DiffReac2D_Dataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        pass

    def _make_info(self):
        # basic info
        self.dim = 2
        self.N = len(self.hdf5_file.keys())
        self.T = self.hdf5_file[list(self.hdf5_file.keys())[0]]['data'].shape[0]
        self.X = self.hdf5_file[list(self.hdf5_file.keys())[0]]['data'].shape[1]
        # length of N and which N to start
        if self.split == 'train':
            self.len_N = self.N - 1
            self.N_start = 0
        else:
            self.len_N = 1
            self.N_start = self.N - 1

    def _get_example(self, n, t, x_size):
        key = list(self.hdf5_file.keys())[n]
        if x_size is None:
            data = np.zeros([11, self.X, self.X])
            data[6, ...] = self.hdf5_file[key]['data'][t, ..., 0]
            data[7, ...] = self.hdf5_file[key]['data'][t, ..., 1]
            return data
        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size, x_size])
        data[6, ...] = self.hdf5_file[key]['data'][t, ..., 0][::(self.X // x_size), ::(self.X // x_size)]
        data[7, ...] = self.hdf5_file[key]['data'][t, ..., 1][::(self.X // x_size), ::(self.X // x_size)]
        return data


class NS_Incomp_2D_Dataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        self.force = self.hdf5_file['force']
        self.particles = self.hdf5_file['particles']
        self.velocity = self.hdf5_file['velocity']

    def _make_info(self):
        # basic info
        self.dim = 2
        self.N = self.velocity.shape[0]
        self.T = self.velocity.shape[1]
        self.X = self.velocity.shape[2]
        # length of N and which N to start
        # NOTE: all together because there is lots of ns incompressible datafiles,
        #       therefore divide by file instead of splitting one file.
        self.len_N = self.N
        self.N_start = 0

    def _get_example(self, n, t, x_size):
        vx = self.velocity[n, t, :, :, 0]
        vy = self.velocity[n, t, :, :, 1]
        force_x = self.force[n, :, :, 0]  # the force is constant for each trajectory
        force_y = self.force[n, :, :, 1]
        particles = self.particles[n, t, :, :, 0]
        if x_size is None:
            data = np.zeros([11, self.X, self.X])
            data[0, ...] = vx
            data[1, ...] = vy
            data[3, ...] = particles
            data[9, ...] = force_x
            data[10, ...] = force_y
            return data
        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size, x_size])
        data[0, ...] = vx[::(self.X // x_size), ::(self.X // x_size)]
        data[1, ...] = vy[::(self.X // x_size), ::(self.X // x_size)]
        data[3, ...] = particles[::(self.X // x_size), ::(self.X // x_size)]
        data[9, ...] = force_x[::(self.X // x_size), ::(self.X // x_size)]
        data[10, ...] = force_y[::(self.X // x_size), ::(self.X // x_size)]
        return data


class CFD2D_Dataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        self.Vx = self.hdf5_file['Vx']
        self.Vy = self.hdf5_file['Vy']
        self.density = self.hdf5_file['density']
        self.pressure = self.hdf5_file['pressure']

    def _make_info(self):
        # basic info
        self.dim = 2
        self.N = self.Vx.shape[0]
        self.T = self.Vx.shape[1]
        self.X = self.Vx.shape[2]
        # length of N and which N to start
        if self.split == 'train':
            self.len_N = self.N - 1
            self.N_start = 0
        else:
            self.len_N = 1
            self.N_start = self.N - 1

    def _get_example(self, n, t, x_size):
        vx = self.Vx[n, t, :, :]
        vy = self.Vy[n, t, :, :]
        density = self.density[n, t, :, :]
        pressure = self.pressure[n, t, :, :]

        if x_size is None:
            data = np.zeros([11, self.X, self.X])
            data[0, ...] = vx
            data[1, ...] = vy
            data[3, ...] = pressure
            data[4, ...] = density
            return data

        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size, x_size])
        data[0, ...] = vx[::(self.X // x_size), ::(self.X // x_size)]
        data[1, ...] = vy[::(self.X // x_size), ::(self.X // x_size)]
        data[3, ...] = pressure[::(self.X // x_size), ::(self.X // x_size)]
        data[4, ...] = density[::(self.X // x_size), ::(self.X // x_size)]
        return data


class CFD3D_Dataset(BaseDataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        super().__init__(split, hdf5_file_path, num_t, x_size)

    def _init_data(self):
        self.Vx = self.hdf5_file['Vx']
        self.Vy = self.hdf5_file['Vy']
        self.Vz = self.hdf5_file['Vz']
        self.density = self.hdf5_file['density']
        self.pressure = self.hdf5_file['pressure']

    def _make_info(self):
        # basic info
        self.dim = 3
        self.N = self.Vx.shape[0]
        self.T = self.Vx.shape[1]
        self.X = self.Vx.shape[2]
        # length of N and which N to start
        if self.split == 'train':
            self.len_N = self.N - 1
            self.N_start = 0
        else:
            self.len_N = 1
            self.N_start = self.N - 1

    def _get_example(self, n, t, x_size):
        vx = self.Vx[n, t, :, :]
        vy = self.Vy[n, t, :, :]
        vz = self.Vz[n, t, :, :]
        density = self.density[n, t, :, :]
        pressure = self.pressure[n, t, :, :]

        if x_size is None:
            data = np.zeros([11, self.X, self.X, self.X])
            data[0, ...] = vx
            data[1, ...] = vy
            data[2, ...] = vz
            data[3, ...] = pressure
            data[4, ...] = density
            return data
        if self.X % x_size != 0:
            raise ValueError(f"x_size {x_size} must be a divisor of x_len {self.X}")
        data = np.zeros([11, x_size, x_size, x_size])
        data[0, ...] = vx[::(self.X // x_size), ::(self.X // x_size), ::(self.X // x_size)]
        data[1, ...] = vy[::(self.X // x_size), ::(self.X // x_size), ::(self.X // x_size)]
        data[2, ...] = vz[::(self.X // x_size), ::(self.X // x_size), ::(self.X // x_size)]
        data[3, ...] = pressure[::(self.X // x_size), ::(self.X // x_size), ::(self.X // x_size)]
        data[4, ...] = density[::(self.X // x_size), ::(self.X // x_size), ::(self.X // x_size)]
        return data


class OmniArchDataset(Dataset):
    def __init__(self, split, hdf5_file_path, num_t=10, x_size=None):
        self.dataset_classes = {
            # 1D
            "1D_Advection": Advection_ReacDiff_Dataset,
            "1D_CFD": CFD1D_Dataset,
            "1D_Burgers": BurgersDataset,
            "1D_diff-sorp": DiffSorpDataset,
            "ReacDiff": Advection_ReacDiff_Dataset,
            # 2D
            "2D_CFD": CFD2D_Dataset,
            "2D_diff-react": DiffReac2D_Dataset,
            "ns_incom": NS_Incomp_2D_Dataset,
            "2D_rdb": Rdb2D_Dataset,
            # below is unchecked 2d
            "KH_M": CFD2D_Dataset,
            "2D_shock": CFD2D_Dataset,

            # 3D
            "3D_CFD": CFD3D_Dataset,
            "Turb_M": CFD3D_Dataset,
            "BlastWave": CFD3D_Dataset,
        }
        dataset_class = None
        for key in self.dataset_classes.keys():
            if key in hdf5_file_path:
                dataset_class = self.dataset_classes[key]
                break
        if dataset_class is None:
            raise ValueError(f"dataset class not found for {hdf5_file_path}, ask LiYing 347073775@qq.com to implement")
        self.dataset = dataset_class(split, hdf5_file_path, num_t=num_t, x_size=x_size)
        self.dim = self.dataset.dim

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class MixedOmniArchDataset(Dataset):
    def __init__(self, split, num_t=10, datafile_paths=None, x_size=None) -> None:
        super().__init__()
        if datafile_paths is None:
            raise ValueError("datafile_paths must be specified")
        if split == 'train':
            self.datafile_paths = datafile_paths
        else:
            self.datafile_paths = datafile_paths
        self.sub_datasets_1d = []
        self.sub_datasets_2d = []
        self.sub_datasets_3d = []
        self.offsets = [0]
        for datafile_path in self.datafile_paths:
            dataset = OmniArchDataset(split, datafile_path, num_t=num_t, x_size=x_size)
            if len(dataset) > 0:
                if dataset.dim == 1:
                    self.sub_datasets_1d.append(dataset)
                elif dataset.dim == 2:
                    self.sub_datasets_2d.append(dataset)
                elif dataset.dim == 3:
                    self.sub_datasets_3d.append(dataset)
        self.sub_datasets_1d = torch.utils.data.ConcatDataset(self.sub_datasets_1d) if len(
            self.sub_datasets_1d) > 0 else []
        print(f"1d dataset len: {len(self.sub_datasets_1d)}")
        self.sub_datasets_2d = torch.utils.data.ConcatDataset(self.sub_datasets_2d) if len(
            self.sub_datasets_2d) > 0 else []
        print(f"2d dataset len: {len(self.sub_datasets_2d)}")
        self.sub_datasets_3d = torch.utils.data.ConcatDataset(self.sub_datasets_3d) if len(
            self.sub_datasets_3d) > 0 else []
        print(f"3d dataset len: {len(self.sub_datasets_3d)}")
        self.offsets.append(len(self.sub_datasets_1d))
        self.offsets.append(len(self.sub_datasets_1d) + len(self.sub_datasets_2d))
        self.offsets.append(len(self.sub_datasets_1d) + len(self.sub_datasets_2d) + len(self.sub_datasets_3d))

    def __getitem__(self, index):
        if index < self.offsets[1]:
            return self.sub_datasets_1d[index]
        elif index < self.offsets[2]:
            return self.sub_datasets_2d[index - self.offsets[1]]
        else:
            return self.sub_datasets_3d[index - self.offsets[2]]

    def __len__(self) -> int:
        return self.offsets[-1]


class OmniArchSampler(torch.utils.data.Sampler):
    """
        batch_sizes: batchsize for 1,2 and 3 dimension data, must be three integers
    """

    def __init__(self, datasets: MixedOmniArchDataset, batch_sizes=None, drop_last=False, seed=None):
        super().__init__(data_source=datasets)
        self.datasets = datasets
        self.sub_samplers = []
        self.batch_sizes = []
        if seed is None:
            seed = np.random.randint(1000)
        self.seed = seed
        if batch_sizes is None:
            batch_sizes = [256, 128, 6]
        elif len(batch_sizes) != 3:
            raise ValueError("the batch_sizes must be a list of three integers.")
        if len(datasets.sub_datasets_1d) > 0:
            self.sub_samplers.append(torch.utils.data.RandomSampler(
                datasets.sub_datasets_1d,
                generator=torch.Generator().manual_seed(seed))
            )
            self.batch_sizes.append(batch_sizes[0])
        if len(datasets.sub_datasets_2d) > 0:
            self.sub_samplers.append(torch.utils.data.RandomSampler(
                datasets.sub_datasets_2d,
                generator=torch.Generator().manual_seed(seed))
            )
            self.batch_sizes.append(batch_sizes[1])
        if len(datasets.sub_datasets_3d) > 0:
            self.sub_samplers.append(torch.utils.data.RandomSampler(
                datasets.sub_datasets_3d,
                generator=torch.Generator().manual_seed(seed))
            )
            self.batch_sizes.append(batch_sizes[2])
        self.drop_last = drop_last

    def __iter__(self):
        np.random.seed(self.seed)
        iterators = [iter(sub_sampler) for sub_sampler in self.sub_samplers]
        batch_sizes = [x for x in self.batch_sizes]
        offsets = [x for x in self.datasets.offsets]
        while len(iterators) > 0:
            index = np.random.randint(len(iterators))
            iterator = iterators[index]
            batch_size = batch_sizes[index]
            offset = offsets[index]
            batch = []
            for _ in range(batch_size):
                try:
                    batch.append(next(iterator) + offset)
                except StopIteration:
                    iterators.pop(index)
                    batch_sizes.pop(index)
                    offsets.pop(index)
                    break
            if len(batch) > 0:
                yield batch

    def __len__(self):
        return sum([len(sampler) for sampler in self.sub_samplers])
