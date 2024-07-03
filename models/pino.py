import torch
import torch.nn as nn
from models.integral import simple_attn, SpectralConv2d
from models.layers import MPLayer, _get_act
import warnings

# Ignore UserWarning
warnings.filterwarnings('ignore', category=UserWarning)


class PINO_img(nn.Module):

    def __init__(self, modes1, modes2, kernel='fno', use_exf=True, time=True, in_dim=4, out_dim=2,
                 no_layers=None, act='gelu', hidden_features=64, blocks=16, pad_ratio=[0., 0.], H=29, W=58):
        super(PINO_img, self).__init__()
        """
        Args:
        -FNO:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
        -kernel: fno or attention
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to both sides. pad_ratio[0] pads left, pad_ratio[1] pads right. 
        """
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions'
        self.modes1 = modes1
        self.modes2 = modes2
        self.no_layers = no_layers
        self.pad_ratio = pad_ratio
        self.act = _get_act(act)
        self.hidden_features = hidden_features
        self.kernel = kernel
        self.H = H
        self.W = W
        self.use_exf = use_exf
        self.time = time

        # input channel is 4: (u(x, y), v(x, y), x, y)

        self.encoder = nn.Linear(in_dim + 1, self.hidden_features)

        if kernel == 'fno':
            self.convs = nn.ModuleList([SpectralConv2d(
                self.hidden_features, self.hidden_features, self.modes1[i], self.modes2[i]) for i in
                range(self.no_layers)])
            self.ws = nn.ModuleList([nn.Conv1d(self.hidden_features, self.hidden_features, 1)
                                     for _ in range(self.no_layers)])

        elif kernel == 'att':
            self.convs = nn.ModuleList([simple_attn(self.hidden_features, blocks) for _ in range(self.no_layers)])


        if use_exf:
            self.embed_day = nn.Embedding(8, 2)
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
            self.ext2lr_global = nn.Sequential(
                nn.Linear(6, 64),
                nn.Dropout(0.3),
                self.act(),
                nn.Linear(64, int(H * W)),
                self.act()
            )
        if time:
            time_span = 14
            self.time_conv = nn.ModuleList([])
            for i in range(time_span):
                self.time_conv.append(nn.Conv2d(self.hidden_features, self.hidden_features, 3, 1, 1))

        self.fc1 = nn.Linear(self.hidden_features, 128)
        self.fc2 = nn.Linear(128, self.hidden_features)
        self.fc3 = nn.Linear(self.hidden_features, out_dim)

    def fea_agg(self, inp, coords):
        # Initialize a tensor to store the distances
        distances = torch.zeros([coords.shape[0], coords.shape[1], 8]).to(coords.device)
        new_inp = inp.detach().clone().permute(1, 2, 0)  # [29, 58, 64]
        # print(new_inp.shape)  # [64, 29, 58]
        new_ = torch.zeros([coords.shape[0], coords.shape[1], 8 * self.hidden_features]).to(new_inp.device)

        # Calculate the distances
        for i in range(1, coords.shape[0] - 1):
            for j in range(1, coords.shape[1] - 1):
                neighbors = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1),
                             (i + 1, j), (i + 1, j + 1)]
                for k, (ni, nj) in enumerate(neighbors):
                    distances[i, j, k] = torch.dist(coords[i, j], coords[ni, nj])
                    # new_inp = torch.cat((new_inp.unsqueeze(-2), inp[ni, nj].unsqueeze(-2)), dim=-2)
                    new_[i, j, k * self.hidden_features:(k + 1) * self.hidden_features] = new_inp[ni, nj, :]

                    # distances [H, W, 8]
                    # new_inp [H, W ,8*c] -> pooling
        # print(new_.device, new_inp.device)
        new_ = torch.cat((new_inp, new_), dim=-1)
        return distances, new_

    def embed_ext(self, ext):
        ext_out1 = self.embed_day(ext[:, 2].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(
            ext[:, 3].long().view(-1, 1)).view(-1, 3)
        ext_out0 = ext[:, 0].view(-1, 1)

        return torch.cat([ext_out0, ext_out1, ext_out2], dim=1)

    def integral(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[-2], x.shape[-1]

        if self.kernel == 'fno':
            for i, (speconv, w) in enumerate(zip(self.convs, self.ws)):
                x1 = speconv(x)
                x2 = w(x.view(batchsize, self.hidden_features, -1)).view(batchsize, self.hidden_features, size_x,
                                                                         size_y)
                x = x1 + x2
                if i != self.no_layers - 1:
                    x = self.act()(x)
        else:
            for i in range(self.no_layers):
                x = self.convs[i](x)
        # print(x.shape)
        return x

    def decoder(self, x):
        # x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act()(x)
        x = self.fc2(x)
        x = self.act()(x)
        x = self.fc3(x)
        # x = x + self.inp
        return x

    def decoder_time(self, eif, x):
        output = []
        for i in range(x.size(0)):
            t = int(eif[i, 3].cpu().detach().numpy())
            t -= 5  # 5-18 0-13
            output.append(self.act()(self.time_conv[t](x[i].unsqueeze(0))))
        out = torch.cat(output, dim=0).permute(0, 2, 3, 1)  # [B, H, W, 64]
        return out

    def forward(self, inp, ext):
        b, h, w, _ = inp.shape
        ext = ext.squeeze(1)
        if self.use_exf:
            ext_emb = self.embed_ext(ext)
            ext_emb = self.ext2lr_global(ext_emb).reshape(b, h, w, 1)
            inp = torch.cat((inp, ext_emb), -1)  #
        feat = self.encoder(inp)
        feat = feat.permute(0, 3, 1, 2)  # B, C, X, Y

        h = self.integral(feat)
        # print(h.shape)

        if self.time:
            h = self.decoder_time(ext, h)
            out = self.decoder(h)

        else:
            h = h.permute(0, 2, 3, 1)
            out = self.decoder(h)

        return out


class PINO_graph(nn.Module):

    def __init__(self, modes1, modes2, kernel='fno', use_exf=True, time=True, in_dim=4, out_dim=2, gnn_layers=None,
                 no_layers=None, act='gelu', hidden_features=64, blocks=16, pad_ratio=[0., 0.], H=29, W=58):
        super(PINO_graph, self).__init__()
        """
        Args:
        -FNO:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
        -kernel: fno or attention
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to both sides. pad_ratio[0] pads left, pad_ratio[1] pads right. 
        """
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions'
        self.modes1 = modes1
        self.modes2 = modes2

        self.pad_ratio = pad_ratio
        self.kernel = kernel
        self.hidden_features = hidden_features
        self.act = _get_act(act)
        self.gnn_layers = gnn_layers
        self.no_layers = no_layers
        self.H = H
        self.W = W
        self.use_exf = use_exf
        self.time = time

        # input channel is 4: (u(x, y), v(x, y), x, y)

        self.transfer2latent = nn.Sequential(
            nn.Linear(in_dim, self.hidden_features),
            self.act(),
            nn.Linear(self.hidden_features, self.hidden_features),
            self.act()
        )

        if self.use_exf:
            self.gnns = torch.nn.ModuleList(modules=(MPLayer(
                in_features=self.hidden_features,
                hidden_features=self.hidden_features,
                out_features=self.hidden_features,
                n_variables=2,
                act=act
            ) for _ in range(self.gnn_layers - 1)))

            # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
            self.gnns.append(MPLayer(in_features=self.hidden_features,
                                     hidden_features=self.hidden_features,
                                     out_features=self.hidden_features,
                                     n_variables=2,
                                     act=act
                                     )
                             )
        else:
            self.gnns = torch.nn.ModuleList(modules=(MPLayer(
                in_features=self.hidden_features,
                hidden_features=self.hidden_features,
                out_features=self.hidden_features,
                n_variables=1,
                act=act
            ) for _ in range(self.gnn_layers - 1)))

            # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
            self.gnns.append(MPLayer(in_features=self.hidden_features,
                                     hidden_features=self.hidden_features,
                                     out_features=self.hidden_features,
                                     n_variables=1,
                                     act=act
                                     )
                             )

        if kernel == 'fno':
            self.convs = nn.ModuleList([SpectralConv2d(
                self.hidden_features, self.hidden_features, self.modes1[i], self.modes2[i]) for i in
                range(self.no_layers)])
            self.ws = nn.ModuleList([nn.Conv1d(self.hidden_features, self.hidden_features, 1)
                                     for _ in range(self.no_layers)])

        elif kernel == 'att':
            self.convs = nn.ModuleList([simple_attn(self.hidden_features, blocks) for _ in range(self.no_layers)])

        self.fc1 = nn.Linear(self.hidden_features, 128)
        self.fc2 = nn.Linear(128, self.hidden_features)
        self.fc3 = nn.Linear(self.hidden_features, out_dim)

        if use_exf:
            self.embed_day = nn.Embedding(8, 2)
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
            self.ext2lr_global = nn.Sequential(
                nn.Linear(6, 64),
                nn.Dropout(0.3),
                self.act(),
                nn.Linear(64, int(H * W)),
                self.act()
            )

        if time:
            time_span = 14
            self.time_conv = nn.ModuleList([])
            for i in range(time_span):
                self.time_conv.append(nn.Conv2d(self.hidden_features, self.hidden_features, 3, 1, 1))

    def embed_ext(self, ext):
        ext_out1 = self.embed_day(ext[:, 2].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(
            ext[:, 3].long().view(-1, 1)).view(-1, 3)
        ext_out0 = ext[:, 0].view(-1, 1)

        return torch.cat([ext_out0, ext_out1, ext_out2], dim=1)

    def lifting_encoder(self, x, u, pos, variables, edge_index, batch):
        # x [N, 5]
        h = self.transfer2latent(x)
        for i in range(self.gnn_layers):
            h = self.gnns[i](h, pos, variables, edge_index, batch)

        return h

    def integral(self, x):
        x = x.view((-1, self.H, self.W, self.hidden_features)).permute(0, 3, 1,
                                                                       2)  # [B*H*W, C] -> [B, H, W, C] -> [B, C, H, W]

        batchsize = x.shape[0]
        size_x, size_y = x.shape[-2], x.shape[-1]

        if self.kernel == 'fno':
            for i, (speconv, w) in enumerate(zip(self.convs, self.ws)):
                x1 = speconv(x)
                x2 = w(x.view(batchsize, self.hidden_features, -1)).view(batchsize, self.hidden_features, size_x,
                                                                         size_y)
                x = x1 + x2
                if i != self.no_layers - 1:
                    x = self.act()(x)
        else:
            for i in range(self.no_layers):
                x = self.convs[i](x)

        return x

    def decoder(self, x):
        x = self.fc1(x)
        x = self.act()(x)
        x = self.fc2(x)
        x = self.act()(x)
        x = self.fc3(x)
        return x

    def decoder_time(self, eif, x):
        output = []
        for i in range(x.size(0)):
            t = int(eif[i, 3].cpu().detach().numpy())
            t -= 5  # 5-18 0-13
            output.append(self.act()(self.time_conv[t](x[i].unsqueeze(0))))
        out = torch.cat(output, dim=0).permute(0, 2, 3, 1)  # [B, H, W, 64]
        return out

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch
        pos = x[:, -2:]  # [N, 2]
        u = x[:, 0:2]  # [N, 2]
        variable = x[:, 2].view(-1, 1)  # [N, 1]

        ext = graph.x_e
        if self.use_exf:
            ext_emb = self.embed_ext(ext)
            ext_emb = self.ext2lr_global(ext_emb).reshape(-1, 1)  # [B*N,1]
            variable = torch.cat((variable, ext_emb), dim=-1)  # [N, 2]

        h = self.lifting_encoder(x, u, pos, variable, edge_index,
                                 batch)  # lifting operation -> channel expander via GNN

        h = self.integral(h)  # latent space approximation

        if self.time:
            h = self.decoder_time(ext, h)
            out = self.decoder(h)

        else:
            h = h.permute(0, 2, 3, 1)
            out = self.decoder(h)

        return out
