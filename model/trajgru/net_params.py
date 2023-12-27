from collections import OrderedDict
from trajgru import TrajGRU
from model import activation

class Params():

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height # image height
        self.width = width # image width
        """
        act: activation function
        - leaky_ : leaky relu   
        - relu_ : relu
        - gelu_ : gelu
        """
        activation_function = activation('leaky', negative_slope=0.2, inplace=True)
        self.trajgru_encoder_params = [
            [
            # 1 - in_channel, out_channel, kernel_size, stride, padding
            OrderedDict({'conv1_leaky_1': [self.in_channels, self.out_channels//4, 3, 1, 1]}), 
            # 3 - in_channel=out_channel
            OrderedDict({'conv2_leaky_1': [self.out_channels//2, self.out_channels//2, 3, 2, 1]}),
            # 5 - in_channel=out_channels
            OrderedDict({'conv3_leaky_1': [self.out_channels, self.out_channels, 3, 2, 1]}),
            ],
            
            # 2 - num_features: out_channels (in_channels * 2)
            [
            TrajGRU(input_channel=self.out_channels//4, 
                num_filter=self.out_channels//2, 
                b_h_w=(4, self.height, self.width), 
                zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation_function),

            # 4 - num_features: out_channels (in_channels * 2)
            TrajGRU(input_channel=self.out_channels//2, 
                num_filter=self.out_channels, 
                b_h_w=(4, self.height//2, self.width//2), 
                zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation_function),

            # 6 - num_features: out_channels (in_channels * 2)
            TrajGRU(input_channel=self.out_channels, 
                num_filter=self.out_channels, 
                b_h_w=(4, self.height//4, self.width//4), 
                zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation_function)
            ]
        ]

        self.trajgru_decoder_params = [
            [
            # 8 - final channel, final channel, 4, 2, 1
            OrderedDict({'deconv1_leaky_1': [self.out_channels, self.out_channels, 4, 2, 1]}),
            # 10 - final channel, final channel, 4, 2, 1
            OrderedDict({'deconv2_leaky_1': [self.out_channels, self.out_channels, 4, 2, 1]}),

            # 12, 13 - final channel // b, d, 3, 1, 1 -> conv 1x1
            OrderedDict({                                         
                'conv3_leaky_1' : [self.out_channels//2, self.out_channels//4, 3, 1, 1], 
                'conv4_leaky_1': [self.out_channels//4, 1, 1, 1, 0],
            }),
            ],

            [
            # 7 - final channel, final_channel
            TrajGRU(input_channel=self.out_channels, 
                num_filter=self.out_channels, 
                b_h_w=(4, self.height//4, self.width//4), 
                zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation_function),
                    
            # 9 - final channel, final_channel
            TrajGRU(input_channel=self.out_channels, 
                num_filter=self.out_channels, 
                b_h_w=(4, self.height//2, self.width//2), 
                zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation_function),

            # 11 - final channel, final_channel // 2
            TrajGRU(input_channel=self.out_channels, 
                num_filter=self.out_channels//2, 
                b_h_w=(4, self.height, self.width), 
                zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation_function),
            ]
        ]
