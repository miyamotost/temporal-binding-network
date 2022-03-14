from torch import nn
from fusion_classification_network import Fusion_Classification_Network
from transforms import *
from collections import OrderedDict

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class HandBoxNetwork(nn.Module):
    def __init__(self):
        super(HandBoxNetwork, self).__init__()
        self.fc1 = nn.Linear(100, 16)

    def forward(self, input):
        input = input.view((-1, 100)) # [12, sample_len=5, 2, 10] -> [12, 100]
        output = self.fc1(input)      # [12, 100] -> [12, 16]
        return output

class HandTrajNetwork(nn.Module):
    def __init__(self, hand_traj_dim, hidden_dim):
        super(HandTrajNetwork, self).__init__()
        self.hand_traj_dim = hand_traj_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = hand_traj_dim,
                            hidden_size = hidden_dim,
                            batch_first = True)

    def forward(self, input):
        input = input.float()
        output, (h, c) = self.lstm(input, None)
        #output = output[:, -1, :] # [batch_size x sequence_length x hidden_size] -> [batch_size x 1(last time) x hidden_size]
        #output = output.squeeze(1) # [batch_size x 1 x hidden_size] -> [batch_size x hidden_size]
        return output, (h, c)

class HandTrajNetwork2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HandTrajNetwork2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        #input = input.view((-1, 100)) # [12, sample_len=5, 2, 10] -> [12, 100]
        #output = self.fc1(input)      # [12, 100] -> [12, 16]
        hidden = self.fc1(input)
        output = self.fc2(hidden)
        return output



class TBN(nn.Module):

    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, crop_num=1, midfusion='concat',
                 model_name=''):
        super(TBN, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.midfusion = midfusion
        self.model_name = model_name
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        self.new_length = OrderedDict()
        if new_length is None:
            for m in self.modality:
                self.new_length[m] = 1 if (m in ["RGB", "Spec", "HandBoxMask"]) else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)
        self._prepare_tbn()

        for m in self.modality:
            if m=='Flow':
                print("Converting the ImageNet model to a flow init model")
                self.base_model['Flow'] = self._construct_flow_model(self.base_model['Flow'])
            if m=='RGBDiff':
                print("Converting the ImageNet model to RGB+Diff init model")
                self.base_model['RGBDiff'] = self._construct_diff_model(self.base_model['RGBDiff'])
            if m=='Spec':
                print("Converting the ImageNet model to a spectrogram init model")
                self.base_model['Spec'] = self._construct_spec_model(self.base_model['Spec'])
            if m=='HandBox':
                print("Set a handbox init model")
                self.base_model['HandBox'] = HandBoxNetwork()
            if m=='HandTraj':
                print("Set a handtraj init model")
                if self.model_name == 'LSTM_all':
                    self.bn_handtraj = nn.BatchNorm1d(12)
                    self.base_model['HandTraj'] = HandTrajNetwork(12, 1024) # trajectory dim, hidden dim
                elif self.model_name == 'FFN_all':
                    self.base_model['HandTraj'] = HandTrajNetwork2(24000, 2048, 1024)
                    #self.base_model['HandTraj'] = HandTrajNetwork2(8000, 2048, 1024)
                    #self.base_model['HandTraj'] = HandTrajNetwork2(16000, 2048, 1024)
                else:
                    print('invalid model_name!')
                    exit()
        print('\n')

        for m in self.modality:
            self.add_module(m.lower(), self.base_model[m])

    def _remove_last_layer(self):
        # This works only with BNInception.
        for m in self.modality:
            delattr(self.base_model[m], self.base_model[m].last_layer_name)
            for tup in self.base_model[m]._op_list:
                if tup[0] == self.base_model[m].last_layer_name:
                    self.base_model[m]._op_list.remove(tup)

    def _prepare_tbn(self):

        self._remove_last_layer()

        self.fusion_classification_net = Fusion_Classification_Network(
            self.feature_dim, self.modality, self.midfusion, self.num_class,
            self.consensus_type, self.before_softmax, self.dropout, self.num_segments)

    def _prepare_base_model(self, base_model):

        if base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = OrderedDict()
            self.input_size = OrderedDict()
            self.input_mean = OrderedDict()
            self.input_std = OrderedDict()

            for m in self.modality:
                self.base_model[m] = getattr(tf_model_zoo, base_model)()
                self.base_model[m].last_layer_name = 'fc'
                self.input_size[m] = 224
                self.input_std[m] = [1]

                if m == 'Flow':
                    self.input_mean[m] = [128]
                elif m == 'RGBDiff':
                    self.input_mean[m] = self.input_mean[m] * (1 + self.new_length[m])
                elif m == 'RGB':
                    self.input_mean[m] = [104, 117, 128]
            self.feature_dim = 1024
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def freeze_fn(self, freeze_mode):

        if freeze_mode == 'modalities':
            for m in self.modality:
                print('Freezing ' + m + ' stream\'s parameters')
                base_model = getattr(self, m.lower())
                for param in base_model.parameters():
                    param.requires_grad_(False)

        elif freeze_mode == 'partialbn_parameters':
            for mod in self.modality:
                count = 0
                print("Freezing BatchNorm2D parameters except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown parameters update in frozen mode
                            m.weight.requires_grad_(False)
                            m.bias.requires_grad_(False)

        elif freeze_mode == 'partialbn_statistics':
            for mod in self.modality:
                count = 0
                print("Freezing BatchNorm2D statistics except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown running statistics update in frozen mode
                            m.eval()
        elif freeze_mode == 'bn_statistics':
            for mod in self.modality:
                print("Freezing BatchNorm2D statistics.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # shutdown running statistics update in frozen mode
                        m.eval()
        else:
            raise ValueError('Unknown mode for freezing the model: {}'.format(freeze_mode))

    def forward(self, input):
        concatenated = []
        for m in self.modality:
            if (m == 'HandBox'):
                channel = 1
                sample_len = channel * self.new_length[m] # TODO: channel と new_lengthの見直し
                base_model = getattr(self, m.lower())
                input[m] = input[m].view((-1, sample_len) + input[m].size()[-2:]) # view((-1, sample_len=5, 2, 10)) -> [12, sample_len=5, 2, 10]
                base_out = base_model(input[m]) # [12, sample_len=5, 2, 10] => [12, ?]
            elif (m == 'HandTraj'):
                base_model = getattr(self, m.lower())
                if self.model_name == "LSTM_all":
                    sample_len = 1*1 # channel × new_length
                    seq_length = input['HandTraj_length'].to("cpu").to(torch.int64)
                    t = input[m]
                    t = t.squeeze(1)

                    # TODO: 正規化
                    #print('bn')
                    #print(t[:, :, -1])
                    #t = torch.permute(t, (0, 2, 1))
                    #t = self.bn_handtraj(t)
                    #t = torch.permute(t, (0, 2, 1))
                    #print(t[:, :, -1])
                    t = pack_padded_sequence(t, seq_length, batch_first=True, enforce_sorted=False)  # [batch, max_len, size] -> [batch x seq_len, 12]
                    #print(t.data.shape)

                    input[m] = t
                    output, _ = base_model(input[m])
                    output, seq_length = pad_packed_sequence(output, batch_first=True)

                    base_out = []
                    for i, out in enumerate(output):                                  # [batch, max_len, hidden] -> [batch, hidden]
                        last = out[seq_length[i]-1, :]                                # [max_len, hidden] -> [hidden]
                        last = last.unsqueeze(0)                                      # [hidden] -> [1, hidden]
                        last = torch.cat((last, last, last), 0)                       # [1, hidden] -> [cons, hidden]
                        base_out.append(last)
                    base_out = torch.stack(base_out)                                  # list of [cons, hidden] -> [batch, cons, hidden]
                    base_out = base_out.view((-1, sample_len) + base_out.size()[-1:]) # [batch, cons, hidden] -> [batch × cons, 1, hidden]
                    base_out = base_out.squeeze(1)                                    # [batch × cons, 1, hidden] -> [batch × cons, hidden]
                elif self.model_name == "FFN_all":
                    print('FFN_all')
                    exit()
                else:
                    print('choose model_name!')
                    exit()
                    #(1)input[m] = input[m].view((-1, sample_len) + input[m].size()[-2:]) # [batch, cons, frames, elements] -> [batch x cons, 1, frames, elements]
                    #input[m] = input[m].view((-1, sample_len) + input[m].size()[-1:]) #(2)
                    #input[m] = torch.reshape(input[m], (12, 16000))
            else:
                if (m == 'RGB'):
                    channel = 3
                elif (m == 'Flow'):
                    channel = 2
                elif (m == 'Spec'):
                    channel = 1
                elif (m == 'HandBoxMask'):
                    channel = 3
                sample_len = channel * self.new_length[m]

                if m == 'RGBDiff':
                    sample_len = 3 * self.new_length[m]
                    input[m] = self._get_diff(input[m])

                base_model = getattr(self, m.lower())
                base_out = base_model(input[m].view((-1, sample_len) + input[m].size()[-2:]))
                base_out = base_out.view(base_out.size(0), -1)
            concatenated.append(base_out)

        output = self.fusion_classification_net(concatenated)

        return output

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3
        input_view = input.view((-1, self.num_segments, self.new_length['RGBDiff'] + 1, input_c,)
            + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length['RGBDiff'] + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['Flow'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length['Flow'], ) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length['Flow'], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach() # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model


    def _construct_spec_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['Spec'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).contiguous()

        new_conv = nn.Conv2d(self.new_length['Spec'], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach() # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)

        # replace the avg pooling at the end, so that it matches the spectrogram dimensionality (256x256)
        pool_layer = getattr(self.base_model['Spec'], 'global_pool')
        new_avg_pooling = nn.AvgPool2d(8, stride=pool_layer.stride, padding=pool_layer.padding)
        setattr(self.base_model['Spec'], 'global_pool', new_avg_pooling)

        return base_model


    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model['RGBDiff'].modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length['RGBDiff'],) + kernel_size[2:]
            new_kernels = params[0].detach().mean(dim=1).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length['RGBDiff'],) + kernel_size[2:]
            new_kernels = torch.cat((params[0].detach(), params[0].detach().mean(dim=1).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length['RGBDiff'],) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        scale_size = {k: v * 256 // 224 for k, v in self.input_size.items()}
        return scale_size

    def get_augmentation(self):
        augmentation = {}
        if 'RGB' in self.modality:
            augmentation['RGB'] = torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size['RGB'], [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        if 'Flow' in self.modality:
            augmentation['Flow'] = torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size['Flow'], [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        if 'RGBDiff' in self.modality:
            augmentation['RGBDiff'] = torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size['RGBDiff'], [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        if 'HandBoxMask' in self.modality:
            augmentation['HandBoxMask'] = torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size['HandBoxMask'], [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])

        return augmentation
