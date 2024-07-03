from models.pino import PINO_img, PINO_graph
from torch.utils.data import DataLoader
from datasets.data_utils import Dataset
from tqdm import tqdm
import torch
from adam import Adam
import yaml
import os
import time
import torch.nn as nn


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


def save_checkpoint(path, name, model, optimizer=None, scheduler=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    if scheduler:
        scheduler_state = scheduler.state_dict()
    else:
        scheduler_state = None
    torch.save({
        'model': model_state_dict,
        'optim': optim_dict,
        'scheduler': scheduler_state
    }, ckpt_dir + name)
    print('Checkpoint is saved in %s' % name)


def train_operator_img(model,
                       train_loader,
                       optimizer,
                       device):
    model.train()
    data_loss = nn.MSELoss()

    train_loss = 0.0
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        optimizer.zero_grad()
        # data loss
        pred = model(batch['inp'], batch['ext'])
        mask = batch['mask']
        mask = mask.reshape((-1, 29, 58, 1))
        pred = pred.masked_fill(~mask, value=0)

        loss_vx = data_loss(pred[:, :, :, 0], batch['target'][:, :, :, 0])
        loss_vy = data_loss(pred[:, :, :, 1], batch['target'][:, :, :, 1])
        loss = loss_vx + loss_vy

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    return train_loss


def train_operator_graph(model,
                         train_loader,
                         optimizer,
                         device):
    model.train()
    data_loss = nn.MSELoss()
    train_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
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

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    return train_loss


def test_operator_img(model,
                      val_loader,
                      device):
    model.eval()
    data_loss = nn.MSELoss()
    with torch.no_grad():
        val_loss = 0.0
        val_x = 0.0
        val_y = 0.0
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

        val_loss /= len(val_loader)
        val_x /= len(val_loader)
        val_y /= len(val_loader)

    return val_loss, val_x, val_y


def test_operator_graph(model,
                        val_loader,
                        device):
    model.eval()
    data_loss = nn.MSELoss()
    with torch.no_grad():
        val_loss = 0.0
        val_x = 0.0
        val_y = 0.0
        for batch in val_loader:
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

        val_loss /= len(val_loader)
        val_x /= len(val_loader)
        val_y /= len(val_loader)
    return val_loss, val_x, val_y


def train(model, train_loader, val_loader, optimizer, scheduler, config, device):
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)
    best_val_loss = 10000.0
    best_epoch = 0
    patience = 20

    for epoch in pbar:
        lr = optimizer.param_groups[0]['lr']
        epoch_start_time = time.time()
        if 'graph' in config['train']['graph']:
            train_loss = train_operator_graph(model, train_loader, optimizer, device)
        else:
            train_loss = train_operator_img(model, train_loader, optimizer, device)

        if 'graph' in config['train']['graph']:
            val_loss, val_x, val_y = test_operator_graph(model, val_loader, device)
        else:
            val_loss, val_x, val_y = test_operator_img(model, val_loader, device)

        if val_loss < best_val_loss:
            save_name = config['train']['save_name'] + "_h{}_gl{}_nl{}_{}_e{}_t{}_K{}_{}{}_{}_{}_{}.pt".format(
                config['model']['hidden_features'], config['model']['gnn_layers'], config['model']['no_layers'],
                config['model']['act'], config['model']['use_exf'], config['model']['time'], config['model']['K'],
                config['model']['kernel'], config['model']['mode'], config['model']['mask_way'],
                config['train']['graph'], config['train']['mask_ratio'])
            save_checkpoint(config['train']['save_dir'],
                            save_name.replace('.pt', f'_{epoch}.pt'),
                            model, optimizer, scheduler)

            best_epoch = epoch
            best_val_loss = val_loss
            patience = 20
        else:
            patience = patience - 1

        if scheduler is not None:
            scheduler.step(epoch)

        pbar.set_description(
            (
                f'Epoch {epoch}, LR {lr:.5f}, train loss: {train_loss:.5f} '
                f'val loss: {val_loss:.5f} - v_x loss: {val_x:.5f} - v_y loss: {val_y:.5f}; '
            )
        )

        log = (
            'Epoch:{} - LR:{:.5f}|Train Loss:{:.5f}|Val Loss:{:.5f}|Val v_x:{:.5f}|Val v_y:{:.5f}|Time_Cost:{:.2f}|Best_Epoch:{}\n'.format(
                epoch, lr, train_loss, val_loss, val_x, val_y,
                time.time() - epoch_start_time,
                best_epoch))
        f = open(
            'log/no_h{}_gl{}_nl{}_{}_e{}_t{}_K{}_{}{}_{}_{}_{}_results.txt'.format(
                config['model']['hidden_features'],
                config['model']['gnn_layers'],
                config['model']['no_layers'],
                config['model']['act'],
                config['model']['use_exf'],
                config['model']['time'],
                config['model']['K'],
                config['model']['kernel'],
                config['model']['mode'],
                config['model']['mask_way'],
                config['train']['graph'],
                config['train']['mask_ratio']
            ),
            'a')
        f.write(log)

        if patience == 0:
            print("Early stopping! Epoch:", epoch)
            break

    return best_epoch


def loader(dataset, set, config):
    if "graph" in config[set]['graph']:
        from torch_geometric.loader import DataLoader
        dataset_loader = DataLoader(dataset, batch_size=config[set]['batch_size'], shuffle=config[set]['shuffle'],
                                    num_workers=config[set]['num_workers'])
    else:
        from torch.utils.data import DataLoader
        dataset_loader = DataLoader(dataset, batch_size=config[set]['batch_size'], shuffle=config[set]['shuffle'],
                                    num_workers=config[set]['num_workers'])
    print(f'Load dataset: {len(dataset_loader.dataset)}')
    return dataset_loader


def main(config):
    seed = 42
    print(f'Random seed :{seed}')
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create model, optimizer and scheduler
    model = get_model(config).to(device)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=config['train']['scheduler_gamma'])

    params = sum(p.numel() for p in model.parameters())

    total_bytes = params * 4  # a float usually occupies 4 bytes
    total_kb = total_bytes / 1024  # 1KB = 1024B
    total_mb = total_kb / 1024  # 1MB = 1024KB

    print('Total parameters: ', params)
    print('Total size in KB: ', total_kb)
    print('Total size in MB: ', total_mb)

    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['scheduler'])
        print('Weights loaded from %s' % ckpt_path)

    # Load datasets
    dataset = Dataset(path=config['train']['path'], graph=config['train']['graph'],
                      scaler=config['train']['scaler'], normal=config['train']['normal'], K=config['train']['K'],
                      mask_way=config['train']['mask_way'], mask_ratio=config['train']['mask_ratio'])  # 1288

    train_set, val_set = torch.utils.data.random_split(dataset, [1149, 144])

    train_loader = loader(train_set, 'train', config)
    val_loader = loader(val_set, 'train', config)

    # Training & Validation
    best_epoch = train(model, train_loader, val_loader, optimizer, scheduler, config, device)

    # Testing
    test_set = Dataset(path=config['test']['path'], graph=config['test']['graph'], scaler=config['test']['scaler'],
                       normal=config['test']['normal'], K=config['test']['K'], mask_way=config['test']['mask_way'],
                       mask_ratio=config['test']['mask_ratio'])  # 149
    test_loader = loader(test_set, 'test', config)

    model = load_model(config, best_epoch, model, device)

    if 'graph' in config['test']['graph']:
        test_loss, test_x, test_y = test_operator_graph(model, test_loader, device)
    else:
        test_loss, test_x, test_y = test_operator_img(model, test_loader, device)

    test_log = ('Test: {:.5f}|Test v_x: {:.5f}|Test v_y: {:.5f}'.format(test_loss, test_x, test_y))

    print(test_log)

    f = open(
        'log/no_h{}_gl{}_nl{}_{}_e{}_t{}_K{}_{}{}_{}_{}_{}_results.txt'.format(
            config['model']['hidden_features'],
            config['model']['gnn_layers'],
            config['model']['no_layers'],
            config['model']['act'],
            config['model']['use_exf'],
            config['model']['time'],
            config['model']['K'],
            config['model']['kernel'],
            config['model']['mode'],
            config['model']['mask_way'],
            config['train']['graph'],
            config['train']['mask_ratio']
        ),
        'a')
    f.write(test_log + '\n')


if __name__ == '__main__':
    start_time = time.time()
    config_file = '../configs/operator/no.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    main(config)

    f = open(
        'log/no_h{}_gl{}_nl{}_{}_e{}_t{}_K{}_{}{}_{}_{}_{}_results.txt'.format(
            config['model']['hidden_features'],
            config['model']['gnn_layers'],
            config['model']['no_layers'],
            config['model']['act'],
            config['model']['use_exf'],
            config['model']['time'],
            config['model']['K'],
            config['model']['kernel'],
            config['model']['mode'],
            config['model']['mask_way'],
            config['train']['graph'],
            config['train']['mask_ratio']
        ),
        'a')
    time_log = (
        f'Total running time: {(time.time() - start_time) // 60:.0f}mins {(time.time() - start_time) % 60:.0f}s')
    print(time_log)
    f.write(time_log + '\n')
    f.close()
