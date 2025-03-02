import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn.functional import pad
from einops import rearrange


activation_fn = {
    'relu': lambda: nn.ReLU(inplace=True),
    'lrelu': lambda: nn.LeakyReLU(inplace=True),
    'prelu': lambda: nn.PReLU()
}


class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, times=1, is_bn=False, activation='relu', kernel_size=3, is_spe_norm=False):
        """
        Initializes a series of layers consisting of convolution, batch normalization, 
        and an activation function, repeated a specified number of times.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dimension (int, optional): Dimension of the input (2 for 2D, 3 for 3D). Defaults to 2.
            times (int, optional): Number of repetitions of the convolutional block. Defaults to 1.
            is_bn (bool, optional): Whether to include batch normalization. Defaults to False.
            activation (str, optional): Name of the activation function ('relu', 'lrelu', or 'prelu'). Defaults to 'relu'.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            is_spe_norm (bool, optional): Whether to apply spectral normalization. Defaults to False.
        """

        super().__init__()

        if dimension == 3:
            conv_fn = lambda in_c: torch.nn.Conv3d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda in_c: torch.nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        layers = []
        for i in range(times):
            if i == 0:
                layers.append(spectral_norm(conv_fn(in_channels)) if is_spe_norm else conv_fn(in_channels))
            else:
                layers.append(spectral_norm(conv_fn(out_channels)) if is_spe_norm else conv_fn(out_channels))

            if is_bn:
                layers.append(bn_fn())

            if activation is not None:
                layers.append(activation_fn[activation]())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (tensor): Input tensor.
        """

        return self.net(x)


class ConvtranBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, is_bn=False, activation='relu', kernel_size=3, is_spe_norm=False):
        """
        Initializes a series of layers consisting of transpose convolution, batch normalization, 
        and an activation function.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dimension (int, optional): Dimension of the input (2 for 2D, 3 for 3D). Defaults to 2.
            is_bn (bool, optional): Whether to include batch normalization. Defaults to False.
            activation (str, optional): Name of the activation function ('relu', 'lrelu', or 'prelu'). Defaults to 'relu'.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            is_spe_norm (bool, optional): Whether to apply spectral normalization. Defaults to False.
        """
    
        self.is_bn = is_bn
        super().__init__()
        if dimension == 3:
            conv_fn = lambda: torch.nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 2, 2),
                padding=kernel_size // 2,
                output_padding=(0, 1, 1)
            )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda: torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1
            )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        self.net1 = spectral_norm(conv_fn()) if is_spe_norm else conv_fn()
        if self.is_bn:
            self.net2 = bn_fn()
        self.net3 = activation_fn[activation]()

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (tensor): Input tensor.
        """

        ret = self.net1(x)
        if self.is_bn:
            ret = self.net2(ret)

        ret = self.net3(ret)

        return ret


class UNet(nn.Module):
    def __init__(self, dimension, i_nc=1, o_nc=1, f_root=32, conv_times=3, is_bn=False, activation='relu',
                 is_residual=False, up_down_times=3, is_spe_norm=False, padding=(0, 0)):

        """
        Initializes a standard U-Net architecture.

        Args:
            i_nc (int): Number of input channels.
            o_nc (int): Number of output channels.
            dimension (int): Dimension of the input (2 for 2D, 3 for 3D).
            f_root (int, optional): Number of feature maps in the first convolution layer. Defaults to 32.
            conv_times (int, optional): Number of repetitions of convolution layers before/after down/up-sampling. Defaults to 3.
            is_bn (bool, optional): Whether to include batch normalization. Defaults to False.
            activation (str, optional): Name of the activation function ('relu', 'lrelu', or 'prelu'). Defaults to 'relu'.
            is_residual (bool, optional): Whether to include residual connections. Defaults to False.
            up_down_times (int, optional): Number of down/up-sampling stages. Defaults to 3.
            is_spe_norm (bool, optional): Whether to apply spectral normalization. Defaults to False.
            padding (tuple, optional): Padding for the input image. Defaults to (0, 0).
        """

        self.is_residual = is_residual
        self.up_down_time = up_down_times
        self.dimension = dimension
        self.padding = padding

        super().__init__()

        if dimension == 2:
            self.down_sample = nn.MaxPool2d((2, 2))
        elif dimension == 3:
            self.down_sample = nn.MaxPool3d((1, 2, 2))
        else:
            raise ValueError()

        self.conv_in = ConvBnActivation(
            in_channels=i_nc,
            out_channels=f_root,
            is_bn=is_bn,
            activation=activation,
            dimension=dimension,
            is_spe_norm=is_spe_norm
        )

        self.conv_out = ConvBnActivation(
            in_channels=f_root,
            out_channels=o_nc,
            kernel_size=1,
            dimension=dimension,
            times=1,
            is_bn=False,
            activation=None,
            is_spe_norm=is_spe_norm
        )

        self.bottom = ConvBnActivation(
            in_channels=f_root * (2 ** (up_down_times - 1)),
            out_channels=f_root * (2 ** up_down_times),
            times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension,
            is_spe_norm=is_spe_norm
        )

        self.down_list = nn.ModuleList([
                                           ConvBnActivation(
                                               in_channels=f_root * 1,
                                               out_channels=f_root * 1,
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension,
                                               is_spe_norm=is_spe_norm
                                           )
                                       ] + [
                                           ConvBnActivation(
                                               in_channels=f_root * (2 ** i),
                                               out_channels=f_root * (2 ** (i + 1)),
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension,
                                               is_spe_norm=is_spe_norm
                                            )
                                           for i in range(up_down_times - 1)
                                       ])

        self.up_conv_list = nn.ModuleList([
            ConvBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension,
                is_spe_norm=is_spe_norm
            )
            for i in range(up_down_times)
        ])

        self.up_conv_tran_list = nn.ModuleList([
            ConvtranBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                is_bn=is_bn, activation=activation, dimension=dimension,
                is_spe_norm=is_spe_norm
            )
            for i in range(up_down_times)
        ])

    def forward(self, x):
        """
        Defines the forward pass of the U-Net.

        Args:
            x (tensor): Input tensor.
        """
    
        input_ = x

        x = pad(x, [0, self.padding[0], 0, self.padding[1]])

        x = self.conv_in(x)

        skip_layers = []
        for i in range(self.up_down_time):
            x = self.down_list[i](x)

            skip_layers.append(x)
            x = self.down_sample(x)

        x = self.bottom(x)

        for i in range(self.up_down_time):
            x = self.up_conv_tran_list[i](x)
            x = torch.cat([x, skip_layers[self.up_down_time - i - 1]], 1)
            x = self.up_conv_list[i](x)

        x = self.conv_out(x)

        if self.padding[0] > 0:
            x = x[..., :-self.padding[0]]
        if self.padding[1] > 0:
            x = x[..., :-self.padding[1], :]

        # x = x[..., :-self.padding[1], :-self.padding[0]]

        ret = input_ - x if self.is_residual else x

        return ret


def pair(t):
    """
    Converts an input to a tuple if it is not already a tuple.

    Args:
        t (Any): Input to be converted.

    Returns:
        tuple: Converted tuple.
    """

    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """
        Initializes a layer normalization followed by a customized function.

        Args:
            dim (int): Input dimension for layer normalization.
            fn (callable): Customized function applied to the output of layer normalization.
        """

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Defines the forward pass.

        Args:
            x (tensor): Input tensor.
        """
         
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        Initializes a feed-forward neural network layer.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float, optional): Dropout ratio. Defaults to 0.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Defines the forward pass.

        Args:
            x (tensor): Input tensor.
        """

        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        Initializes an attention layer.

        Args:
            dim (int): Input dimension.
            heads (int, optional): Number of attention heads. Defaults to 8.
            dim_head (int, optional): Dimension of each attention head. Defaults to 64.
            dropout (float, optional): Dropout ratio. Defaults to 0.
        """

        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Defines the forward pass.

        Args:
            x (tensor): Input tensor.
        """
    
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        Initializes a standard transformer architecture with multiple-head attention and feed-forward layers.

        Args:
            dim (int): Dimension of the input feature vectors.
            depth (int): Number of layers (repetitions of attention and feed-forward modules).
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
            mlp_dim (int): Dimension of the hidden layer in the feed-forward network.
            dropout (float, optional): Dropout rate to apply in attention and feed-forward networks. Defaults to 0.
        """

        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Defines the forward pass.

        Args:
            x (tensor): Input tensor.
        """
                
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
