from collections import OrderedDict
from convlstm import CLSTM_cell

class Params():

    def __init__(self, in_channels, out_channels, height, width):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height # image height
        self.width = width # image width

        self.convlstm_encoder_params = [
            [
            # 1 - in_channel, out_channel, kernel_size, stride, padding
            OrderedDict({'conv1_leaky_1': [self.in_channels, self.out_channels//4, 3, 1, 1]}), 
            # 3 - in_channel=out_channel
            OrderedDict({'conv2_leaky_1': [self.out_channels//2, self.out_channels//2, 3, 2, 1]}),
            # 5 - in_channel=out_channels
            OrderedDict({'conv3_leaky_1': [self.out_channels, self.out_channels, 3, 2, 1]}),
            ],

            [
            # 2 - num_features: out_channels (in_channels * 2)
            CLSTM_cell(shape=(self.height, self.width), 
                input_channels=self.out_channels//4, 
                filter_size=3, 
                num_features=self.out_channels//2),

            # 4 - num_features: out_channels (in_channels * 2)
            CLSTM_cell(shape=(self.height//2, self.width//2), 
                input_channels=self.out_channels//2, 
                filter_size=3, 
                num_features=self.out_channels), 

            # 6 - num_features: out_channels (in_channels * 2)
            CLSTM_cell(shape=(self.height//4, self.width//4), 
                input_channels=self.out_channels, 
                filter_size=3, 
                num_features=self.out_channels) 
            ]
        ]

        self.convlstm_decoder_params = [
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
            CLSTM_cell(shape=(self.height//4, self.width//4), 
                input_channels=self.out_channels, 
                filter_size=3, 
                num_features=self.out_channels),

            # 9 - final channel, final_channel
            CLSTM_cell(shape=(self.height//2, self.width//2), 
                input_channels=self.out_channels, 
                filter_size=3, 
                num_features=self.out_channels),

            # 11 - final channel, final_channel // 2
            CLSTM_cell(shape=(self.height, self.width), 
                input_channels=self.out_channels, 
                filter_size=3, 
                num_features=self.out_channels//2),
            ]
        ]
