"""
create inference dataset
@Author: Ned
"""
import pickle
import torch
import random
from datetime import datetime
import numpy as np

seed = 42
torch.manual_seed(seed)
random.seed(seed)


def create_dataset_from_oripkl(file_path, boundary):
    data_allday = list()
    data_daytime = list()

    with open(file_path, "rb") as pkl_file:
        datasets = pickle.load(pkl_file)

        for sample in datasets:
            v_x = sample['v_x'].values  # (29, 58)
            v_y = sample['v_y'].values  # (29, 58)

            is_nanx = np.isnan(v_x)
            is_nany = np.isnan(v_y)
            nan_ratex = np.mean(is_nanx)
            nan_ratey = np.mean(is_nany)

            s2n = sample['s2n'].values  # (29, 58) signal to noise
            corr = sample['corr'].values  # (29, 58) correlation

            x_pos = sample['x'].values
            y_pos = sample['y'].values

            X, Y = np.meshgrid(x_pos, y_pos)

            X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

            coordinates = torch.cat((X, Y), dim=-1)

            v_x = torch.tensor(v_x, dtype=torch.float32).unsqueeze(-1)
            v_y = torch.tensor(v_y, dtype=torch.float32).unsqueeze(-1)

            label = boundary.unsqueeze(-1)  # [H, W, 1] 0: nan, 1: val

            v_x = torch.where(torch.isnan(v_x), torch.zeros_like(v_x), v_x)
            v_y = torch.where(torch.isnan(v_y), torch.zeros_like(v_y), v_y)

            # WATER DEPTH
            water_depth = torch.tensor(float(sample.attrs['h_a']), dtype=torch.float32).view(-1)
            # DATE
            datetime_str = str(sample['date'].values)
            datetime_obj = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
            month = torch.tensor(datetime_obj.month).view(-1)
            hour = torch.tensor(datetime_obj.hour).view(-1)
            weekday = torch.tensor(datetime_obj.isoweekday()).view(-1)

            x = torch.cat((v_x, v_y, label, coordinates), dim=-1)

            ext = torch.cat((water_depth, month, weekday, hour), dim=-1)
            data_temp = {'x': x, 'ext': ext, 'id': datetime_str}

            if nan_ratex >= 0.2 and nan_ratey >= 0.2:
                # keep the samples with a missing value rate greater or equal than 80%
                data_allday.append(data_temp)
                if hour >= 5 and hour <= 18:
                    data_daytime.append(data_temp)
            else:
                continue

    pkl_file.close()

    return data_allday, data_daytime


def save_file(data, file_path, shuffle=False):
    if shuffle:
        random.shuffle(data)
    else:
        print('Without shuffle!')

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    f.close()


if __name__ == '__main__':
    file_path = '../data/dataset_all.pkl'

    with open('../data/boundary.pkl', 'rb') as f:
        boundary_mask = pickle.load(f)
    f.close()

    data_allday, data_daytime = create_dataset_from_oripkl(file_path, boundary_mask)

    save_path_allday = '../data/infer_dataset_allday.pkl'

    save_file(data_allday, save_path_allday)

    save_path_daytime = '../data/infer_dataset_daytime.pkl'

    save_file(data_daytime, save_path_daytime)
