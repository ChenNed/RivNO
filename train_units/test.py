from models.pino import PINO_img, PINO_graph
from datasets.data_utils import Dataset
import torch
import yaml
from metrics import rmse, mae, r2_score
import random


def get_model(config):
    mode = config['model']['mode']
    nl = config['model']['no_layers']
    if config['train']['graph'] == 'img':
        modes = [mode] * nl
        return PINO_img(modes1=modes, modes2=modes,
                        kernel=config['model']['kernel'], use_exf=bool(config['model']['use_exf']),
                        time=bool(config['model']['time']), in_dim=config['model']['in_dim'],
                        out_dim=config['model']['out_dim'], no_layers=config['model']['no_layers'],
                        act=config['model']['act'], hidden_features=config['model']['hidden_features'],
                        blocks=config['model']['blocks'])
    else:
        modes = [mode] * nl
        return PINO_graph(modes1=modes, modes2=modes,
                          kernel=config['model']['kernel'], use_exf=bool(config['model']['use_exf']),
                          time=bool(config['model']['time']), in_dim=config['model']['in_dim'],
                          out_dim=config['model']['out_dim'], gnn_layers=config['model']['gnn_layers'],
                          no_layers=config['model']['no_layers'], act=config['model']['act'],
                          hidden_features=config['model']['hidden_features'], blocks=config['model']['blocks'])


def load_model(config, best_epoch, model, device):
    graph = config['train']['graph']
    path = config['train']['save_dir']
    hidden_fea = config['model']['hidden_features']
    gnn_layers = config['model']['gnn_layers']
    no_layers = config['model']['no_layers']
    act = config['model']['act']
    ext = config['model']['use_exf']
    time = config['model']['time']
    mode = config['model']['mode']
    mask_way = config['model']['mask_way']
    kernel = config['model']['kernel']
    mask_ratio = config['train']['mask_ratio']
    K = config['model']['K']
    ckpt_path = f'checkpoints/{path}/neural_operator_h{hidden_fea}_gl{gnn_layers}_nl{no_layers}_{act}_e{ext}_t{time}_K{K}_{kernel}{mode}_{mask_way}_{graph}_{mask_ratio}_{best_epoch}.pt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    print('Weights loaded from %s' % ckpt_path)
    return model.to(device)


def test_operator_img(model,
                      val_loader,
                      device):
    model.eval()
    data_loss = torch.nn.MSELoss()
    with torch.no_grad():
        val_loss = 0.0
        val_x = 0.0
        val_y = 0.0

        val_rmse = 0.0
        val_x_rmse = 0.0
        val_y_rmse = 0.0

        val_mae = 0.0
        val_x_mae = 0.0
        val_y_mae = 0.0

        val_r2 = 0.0
        val_x_r2 = 0.0
        val_y_r2 = 0.0
        for batch in val_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            # data loss
            pred = model(batch['inp'], batch['ext'])
            mask = batch['mask']
            mask = mask.reshape((-1, 29, 58, 1))
            pred = pred.masked_fill(~mask, value=0)

            loss_vx = data_loss(pred[:, :, :, 0], batch['target'][:, :, :, 0])
            loss_vy = data_loss(pred[:, :, :, 1], batch['target'][:, :, :, 1])
            loss = loss_vx + loss_vy
            val_loss += loss.item()
            val_x += loss_vx.item()
            val_y += loss_vy.item()

            rmse_vx = rmse(pred[:, :, :, 0], batch['target'][:, :, :, 0])
            rmse_vy = rmse_vy(pred[:, :, :, 1], batch['target'][:, :, :, 1])
            loss_rmse = rmse_vx + rmse_vy
            val_rmse += loss_rmse.item()
            val_x_rmse += rmse_vx.item()
            val_y_rmse += rmse_vy.item()

            mae_vx = mae(pred[:, :, :, 0], batch['target'][:, :, :, 0])
            mae_vy = mae(pred[:, :, :, 1], batch['target'][:, :, :, 1])
            loss_mae = mae_vx + mae_vy
            val_mae += loss_mae.item()
            val_x_mae += mae_vx.item()
            val_y_mae += mae_vy.item()

            r2_vx = r2_score(pred[:, :, :, 0], batch['target'][:, :, :, 0])
            r2_vy = r2_score(pred[:, :, :, 1], batch['target'][:, :, :, 1])
            loss_r2 = r2_vx + r2_vy
            val_r2 += loss_r2.item()
            val_x_r2 += r2_vx.item()
            val_y_r2 += r2_vy.item()

    val_loss /= len(val_loader)
    val_x /= len(val_loader)
    val_y /= len(val_loader)

    val_rmse /= len(val_loader)
    val_x_rmse /= len(val_loader)
    val_y_rmse /= len(val_loader)

    val_mae /= len(val_loader)
    val_x_mae /= len(val_loader)
    val_y_mae /= len(val_loader)

    val_r2 /= len(val_loader)
    val_x_r2 /= len(val_loader)
    val_y_r2 /= len(val_loader)

    return val_loss, val_x, val_y, val_rmse, val_x_rmse, val_y_rmse, val_mae, val_x_mae, val_y_mae, val_r2, val_x_r2, val_y_r2


def test_operator_graph(model,
                        val_loader,
                        device):
    model.eval()
    data_loss = torch.nn.MSELoss()

    with torch.no_grad():
        val_loss = 0.0
        val_x = 0.0
        val_y = 0.0

        val_rmse = 0.0
        val_x_rmse = 0.0
        val_y_rmse = 0.0

        val_mae = 0.0
        val_x_mae = 0.0
        val_y_mae = 0.0

        val_r2 = 0.0
        val_x_r2 = 0.0
        val_y_r2 = 0.0

        for i, batch in enumerate(val_loader):
            batch.to(device)
            # data loss
            pred = model(batch)  # [B, H, W, 2]
            mask = batch.mask
            mask = mask.reshape((-1, 29, 58, 1))
            pred = pred.masked_fill(~mask, value=0)
            y = batch.y.reshape((-1, 29, 58, 2))

            loss_vx = data_loss(pred[:, :, :, 0], y[:, :, :, 0])
            loss_vy = data_loss(pred[:, :, :, 1], y[:, :, :, 1])
            loss = loss_vx + loss_vy
            val_loss += loss.item()
            val_x += loss_vx.item()
            val_y += loss_vy.item()

            rmse_vx = rmse(pred[:, :, :, 0], y[:, :, :, 0])
            rmse_vy = rmse(pred[:, :, :, 1], y[:, :, :, 1])
            loss_rmse = rmse_vx + rmse_vy
            val_rmse += loss_rmse.item()
            val_x_rmse += rmse_vx.item()
            val_y_rmse += rmse_vy.item()

            mae_vx = mae(pred[:, :, :, 0], y[:, :, :, 0])
            mae_vy = mae(pred[:, :, :, 1], y[:, :, :, 1])
            loss_mae = mae_vx + mae_vy
            val_mae += loss_mae.item()
            val_x_mae += mae_vx.item()
            val_y_mae += mae_vy.item()

            r2_vx = r2_score(pred[:, :, :, 0], y[:, :, :, 0])
            r2_vy = r2_score(pred[:, :, :, 1], y[:, :, :, 1])
            loss_r2 = r2_vx + r2_vy
            val_r2 += loss_r2.item()
            val_x_r2 += r2_vx.item()
            val_y_r2 += r2_vy.item()

    val_loss /= len(val_loader)
    val_x /= len(val_loader)
    val_y /= len(val_loader)

    val_rmse /= len(val_loader)
    val_x_rmse /= len(val_loader)
    val_y_rmse /= len(val_loader)

    val_mae /= len(val_loader)
    val_x_mae /= len(val_loader)
    val_y_mae /= len(val_loader)

    val_r2 /= len(val_loader)
    val_x_r2 /= len(val_loader)
    val_y_r2 /= len(val_loader)

    return val_loss, val_x, val_y, val_rmse, val_x_rmse, val_y_rmse, val_mae, val_x_mae, val_y_mae, val_r2, val_x_r2, val_y_r2


def loader(dataset, set, config):
    if "graph" in config[set]['graph']:
        from torch_geometric.loader import DataLoader
        dataset_loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=1)
    else:
        from torch.utils.data import DataLoader
        dataset_loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=1)

    return dataset_loader


if __name__ == '__main__':
    seed = 42
    print(f'Random seed :{seed}')
    torch.manual_seed(seed)
    random.seed(seed)

    config_file = '../configs/operator/no.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('Testing!--------------------------------------------------')
    # Testing
    test_set = Dataset(path=config['test']['path'], graph='veno_graph', scaler=config['test']['scaler'],
                       normal=config['test']['normal'], K=config['test']['K'], mask_way=config['test']['mask_way'],
                       mask_ratio=config['train']['mask_ratio'])
    test_loader = loader(test_set, 'test', config)

    best_epoch = 52  #

    model = get_model(config).to(device)
    model = load_model(config, best_epoch, model, device)

    if 'graph' in config['test']['graph']:
        val_loss, val_x, val_y, val_rmse, val_x_rmse, val_y_rmse, val_mae, val_x_mae, val_y_mae, val_r2, val_x_r2, val_y_r2 = test_operator_graph(
            model,
            test_loader,
            device)
    else:
        val_loss, val_x, val_y, val_rmse, val_x_rmse, val_y_rmse, val_mae, val_x_mae, val_y_mae, val_r2, val_x_r2, val_y_r2 = test_operator_img(
            model, test_loader, device)

    test_log = (
        'Test MSE: {:.5f} [vx: {:.5f}, vy: {:.5f}] | Test RMSE: {:.5f} [vx: {:.5f}, vy: {:.5f}] | Test MAE: {:.5f} [vx: {:.5f}, vy: {:.5f}] | Test R^2: {:.5f} [vx: {:.5f}, vy: {:.5f}]'.format(
            val_loss, val_x, val_y, val_rmse, val_x_rmse, val_y_rmse, val_mae, val_x_mae, val_y_mae, val_r2,
            val_x_r2,
            val_y_r2))
    print(test_log)
