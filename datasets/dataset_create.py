"""
create training [train/val] and testing dataset
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


def create_dataset_from_oripkl(file_path):
    datas = list()
    idx = 0

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

            # point label
            label = torch.ones_like(v_x)  # [H, W, 1]
            label = torch.where(torch.isnan(v_x), torch.zeros_like(label), label)  # [H, W, 1] 0: nan, 1: val

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
            data_temp = {'x': x, 'ext': ext}

            if nan_ratex < 0.2 and nan_ratey < 0.2:
                # filter out the samples with a missing value rate greater than 80%
                datas.append(data_temp)
                idx += 1

            else:
                continue

    pkl_file.close()
    return datas


def save_file(data, file_path_train, file_path_test, shuffle=False, ratios=0.9):
    train_list = list()
    test_list = list()

    total = len(data)

    num_train = int(total * ratios)

    if shuffle:
        random.shuffle(data)
    else:
        print('Without shuffle!')

    print(f'Num train: {len(train_list)}, Num test: {len(test_list)}')

    with open(file_path_train, "wb") as f:
        pickle.dump(data[:num_train], f)
    f.close()

    with open(file_path_test, "wb") as f:
        pickle.dump(data[num_train:], f)
    f.close()


if __name__ == '__main__':
    file_path = '../data/dataset_all.pkl'
    datas = create_dataset_from_oripkl(file_path)

    save_path_train = f'../data/dataset_train_we.pkl'
    save_path_test = f'../data/dataset_test_we.pkl'
    save_file(datas, save_path_train, save_path_test, True)
