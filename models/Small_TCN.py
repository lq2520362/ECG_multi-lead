#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
import nemo


class Small_TCN(nn.Module):
    def __init__(self):
        super(Small_TCN, self).__init__()
        n_inputs = 12
        # Hyperparameters for TCN
        Kt = (32,)
        pt = 0.05
        Ft = 12
        classes = 9

        self.pad0 = nn.ConstantPad1d(padding=(Kt[0] - 1, 0), value=0)
        self.conv0 = nn.Conv1d(in_channels=n_inputs, out_channels=n_inputs + 20, kernel_size=Kt, bias=False)
        self.act0 = nn.ReLU()
        self.batchnorm0 = nn.BatchNorm1d(num_features=n_inputs + 20)

        # First block
        dilation = 1
        self.upsample = nn.Conv1d(in_channels=n_inputs + 20, out_channels=Ft, kernel_size=(1,), bias=False)
        self.upsamplerelu = nn.ReLU()
        self.upsamplebn = nn.BatchNorm1d(num_features=Ft)
        self.pad1 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv1 = nn.Conv1d(in_channels=n_inputs + 20, out_channels=Ft, kernel_size=Kt, dilation=(1,), bias=False)
        self.batchnorm1 = nn.BatchNorm1d(num_features=Ft)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=pt)
        self.pad2 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv2 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=(1,), bias=False)
        self.batchnorm2 = nn.BatchNorm1d(num_features=Ft)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=pt)
        self.add1 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd1 = nn.ReLU()

        # Second block
        dilation = 2
        self.pad3 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv3 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm3 = nn.BatchNorm1d(num_features=Ft)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=pt)
        self.pad4 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv4 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm4 = nn.BatchNorm1d(num_features=Ft)
        self.act4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=pt)
        self.add2 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd2 = nn.ReLU()

        # Third block
        dilation = 4
        self.pad5 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv5 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm5 = nn.BatchNorm1d(num_features=Ft)
        self.act5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=pt)
        self.pad6 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv6 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm6 = nn.BatchNorm1d(num_features=Ft)
        self.act6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=pt)
        self.add3 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd3 = nn.ReLU()

        # Fourth block
        dilation = 8
        self.pad7 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv7 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm7 = nn.BatchNorm1d(num_features=Ft)
        self.act7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=pt)
        self.pad8 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv8 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm8 = nn.BatchNorm1d(num_features=Ft)
        self.act8 = nn.ReLU()
        self.dropout8 = nn.Dropout(p=pt)
        self.add4 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd4 = nn.ReLU()

        # Fifth block
        dilation = 16
        self.pad9 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv9 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm9 = nn.BatchNorm1d(num_features=Ft)
        self.act9 = nn.ReLU()
        self.dropout9 = nn.Dropout(p=pt)
        self.pad10 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv10 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm10 = nn.BatchNorm1d(num_features=Ft)
        self.act10 = nn.ReLU()
        self.dropout10 = nn.Dropout(p=pt)
        self.add5 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd5 = nn.ReLU()

        # Sixth block
        dilation = 32
        self.pad11 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv11 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm11 = nn.BatchNorm1d(num_features=Ft)
        self.act11 = nn.ReLU()
        self.dropout11 = nn.Dropout(p=pt)
        self.pad12 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv12 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm12 = nn.BatchNorm1d(num_features=Ft)
        self.act12 = nn.ReLU()
        self.dropout12 = nn.Dropout(p=pt)
        self.add6 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd6 = nn.ReLU()

        # Seventh block
        dilation = 64
        self.pad13 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv13 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm13 = nn.BatchNorm1d(num_features=Ft)
        self.act13 = nn.ReLU()
        self.dropout13 = nn.Dropout(p=pt)
        self.pad14 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv14 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm14 = nn.BatchNorm1d(num_features=Ft)
        self.act14 = nn.ReLU()
        self.dropout14 = nn.Dropout(p=pt)
        self.add7 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd7 = nn.ReLU()

        # Eighth block
        dilation = 128
        self.pad15 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv15 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm15 = nn.BatchNorm1d(num_features=Ft)
        self.act15 = nn.ReLU()
        self.dropout15 = nn.Dropout(p=pt)
        self.pad16 = nn.ConstantPad1d(padding=((Kt[0] - 1) * dilation, 0), value=0)
        self.conv16 = nn.Conv1d(in_channels=Ft, out_channels=Ft, kernel_size=Kt, dilation=dilation, bias=False)
        self.batchnorm16 = nn.BatchNorm1d(num_features=Ft)
        self.act16 = nn.ReLU()
        self.dropout16 = nn.Dropout(p=pt)
        self.add8 = nemo.quant.pact.PACT_IntegerAdd()
        self.reluadd8 = nn.ReLU()

        # Last layer
        self.linear = nn.Linear(in_features=Ft * 3000, out_features=classes, bias=False)

    def forward(self, x):
        # Now we propagate through the network correctly
        x = self.pad0(x)
        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = self.act0(x)

        # TCN
        # First block
        res = self.pad1(x)
        res = self.conv1(res)
        res = self.batchnorm1(res)
        res = self.act1(res)
        res = self.dropout1(res)
        res = self.pad2(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.act2(res)
        res = self.dropout2(res)

        x = self.upsample(x)
        x = self.upsamplebn(x)
        x = self.upsamplerelu(x)

        x = self.add1(x, res)
        x = self.reluadd1(x)

        # Second block
        res = self.pad3(x)
        # res = self.pad3(res)
        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.act3(res)
        res = self.dropout3(res)
        res = self.pad4(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.act4(res)
        res = self.dropout4(res)
        x = self.add2(x, res)
        x = self.reluadd2(x)

        # Third block
        res = self.pad5(x)
        # res = self.pad5(res)
        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.act5(res)
        res = self.dropout5(res)
        res = self.pad6(res)
        res = self.conv6(res)
        res = self.batchnorm6(res)
        res = self.act6(res)
        res = self.dropout6(res)
        x = self.add3(x, res)
        x = self.reluadd3(x)

        # Fourth block
        res = self.pad7(x)
        # res = self.pad7(res)
        res = self.conv7(res)
        res = self.batchnorm7(res)
        res = self.act7(res)
        res = self.dropout7(res)
        res = self.pad8(res)
        res = self.conv8(res)
        res = self.batchnorm8(res)
        res = self.act8(res)
        res = self.dropout8(res)
        x = self.add4(x, res)
        x = self.reluadd4(x)

        # Fifth block
        res = self.pad9(x)
        # res = self.pad9(res)
        res = self.conv9(res)
        res = self.batchnorm9(res)
        res = self.act9(res)
        res = self.dropout9(res)
        res = self.pad10(res)
        res = self.conv10(res)
        res = self.batchnorm10(res)
        res = self.act10(res)
        res = self.dropout10(res)
        x = self.add5(x, res)
        x = self.reluadd5(x)

        # Sixth block
        res = self.pad11(x)
        # res = self.pad11(res)
        res = self.conv11(res)
        res = self.batchnorm11(res)
        res = self.act11(res)
        res = self.dropout11(res)
        res = self.pad12(res)
        res = self.conv12(res)
        res = self.batchnorm12(res)
        res = self.act12(res)
        res = self.dropout12(res)
        x = self.add6(x, res)
        x = self.reluadd6(x)

        # Seventh block
        res = self.pad13(x)
        # res = self.pad13(res)
        res = self.conv13(res)
        res = self.batchnorm13(res)
        res = self.act13(res)
        res = self.dropout13(res)
        res = self.pad14(res)
        res = self.conv14(res)
        res = self.batchnorm14(res)
        res = self.act14(res)
        res = self.dropout14(res)
        x = self.add7(x, res)
        x = self.reluadd7(x)

        # Eighth block
        res = self.pad15(x)
        # res = self.pad15(res)
        res = self.conv15(res)
        res = self.batchnorm15(res)
        res = self.act15(res)
        res = self.dropout15(res)
        res = self.pad16(res)
        res = self.conv16(res)
        res = self.batchnorm16(res)
        res = self.act16(res)
        res = self.dropout16(res)
        x = self.add8(x, res)
        x = self.reluadd8(x)

        # Linear layer to classify
        x = x.flatten(1)
        o = self.linear(x)
        return o
