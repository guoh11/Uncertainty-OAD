import torch
import torch.nn as nn


def fc_relu(input_features, output_features, inplace=True):
    return nn.Sequential(
        nn.Linear(input_features, output_features),
        nn.ReLU(inplace=inplace),
    )


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class FeatureSelector(nn.Module):
    def __init__(self, args):
        super(FeatureSelector, self).__init__()

        if args.inputs in ['rgb', 'flow', 'two-stream']:
            self.with_rgb = 'flow' not in args.inputs
            self.with_flow = 'rgb' not in args.inputs
        else:
            raise(RuntimeError('Unknown inputs of {}'.format(args.inputs)))

        if self.with_rgb and self.with_flow:
            self.fusion_size = 2048 + 1024
        elif self.with_rgb:
            self.fusion_size = 2048
        elif self.with_flow:
            self.fusion_size = 1024

        self.input_linear = nn.Sequential(
            nn.Linear(self.fusion_size, self.fusion_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_input, flow_input):
        if self.with_rgb and self.with_flow:
            fusion_input = torch.cat((rgb_input, flow_input), 1)
        elif self.with_rgb:
            fusion_input = rgb_input
        elif self.with_flow:
            fusion_input = flow_input
        return self.input_linear(fusion_input)


class UTRN(nn.Module):
    def __init__(self, args):
        super(UTRN, self).__init__()
        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.pro_size = args.pro_size
        self.num_classes = args.num_classes
        self.dropout = args.dropout


        self.feature_extractor = FeatureSelector(args)
        self.future_size = self.feature_extractor.fusion_size
        self.fusion_size = self.feature_extractor.fusion_size * 2

        self.hidden_transform = fc_relu(self.hidden_size, self.hidden_size)
        self.cell_transform = fc_relu(self.hidden_size, self.hidden_size)
        self.fusion_linear = fc_relu(self.num_classes, self.hidden_size)
        self.future_linear = fc_relu(self.hidden_size, self.future_size)

        self.enc_drop = nn.Dropout(self.dropout)
        self.enc_cell = nn.LSTMCell(self.fusion_size, self.hidden_size)
        self.dec_drop = nn.Dropout(self.dropout)
        self.dec_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.m_layer = nn.Linear(self.hidden_size, self.pro_size)
        self.u_layer = nn.Linear(self.hidden_size, self.pro_size)
        self.enc_classifier = nn.Linear(self.pro_size, self.num_classes)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

    def attention(self, rgb_input, enc_steps, bs=12, uncertainty=torch.rand(3,64)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        att = torch.ones([bs, enc_steps]).to(device)
        for i in range(bs):
            att[i, :] += uncertainty[i, :]/uncertainty.sum(dim=1)[i]
        att = torch.unsqueeze(att, dim=2)
        rgb_input = torch.mul(rgb_input, att)
        att.cpu()

        return rgb_input

    def pro_layer(self, input):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = self.m_layer(input).to(device)
        u = self.u_layer(input).to(device)
        output = m + u*torch.rand(self.pro_size).to(device)
        return output

    def encoder(self, rgb_input, flow_input, future_input, enc_hidden, enc_cell):
        fusion_input = self.feature_extractor(rgb_input, flow_input)
        fusion_input = torch.cat((fusion_input, future_input), 1)
        enc_hidden, enc_cell = self.enc_cell(self.enc_drop(fusion_input), (enc_hidden, enc_cell))
        enc_pro = self.pro_layer(enc_hidden)
        enc_score = self.enc_classifier(self.enc_drop(enc_pro))
        return enc_hidden, enc_cell, enc_score

    def decoder(self, fusion_input, dec_hidden, dec_cell):
        dec_hidden, dec_cell = self.dec_cell(self.dec_drop(fusion_input), (dec_hidden, dec_cell))
        dec_score = self.classifier(self.dec_drop(dec_hidden))
        return dec_hidden, dec_cell, dec_score

    def step(self, rgb_input, flow_input, future_input, enc_hidden, enc_cell):
        # Encoder -> time t
        enc_hidden, enc_cell, enc_score = self.encoder(rgb_input, flow_input, future_input, enc_hidden, enc_cell)

        # Decoder -> time t + 1
        dec_score_stack = []
        dec_hidden = self.hidden_transform(enc_hidden)
        dec_cell = self.cell_transform(enc_cell)
        fusion_input = rgb_input.new_zeros((rgb_input.shape[0], self.hidden_size))
        future_input = rgb_input.new_zeros((rgb_input.shape[0], self.future_size))
        for dec_step in range(self.dec_steps):
            dec_hidden, dec_cell, dec_score = self.decoder(fusion_input, dec_hidden, dec_cell)
            dec_score_stack.append(dec_score)
            fusion_input = self.fusion_linear(dec_score)
            future_input = future_input + self.future_linear(dec_hidden)
        future_input = future_input / self.dec_steps

        return future_input, enc_hidden, enc_cell, enc_score, dec_score_stack

    def forward(self, rgb_inputs, flow_inputs, uncertainty):
        batch_size = rgb_inputs.shape[0]
        if uncertainty!=None:
            uncertainty = torch.squeeze(uncertainty, dim=-1)
            rgb_inputs = self.attention(rgb_inputs, enc_steps=self.enc_steps, bs=batch_size, uncertainty=uncertainty)
        enc_hidden = rgb_inputs.new_zeros((batch_size, self.hidden_size))
        enc_cell = rgb_inputs.new_zeros((batch_size, self.hidden_size))
        future_input = rgb_inputs.new_zeros((batch_size, self.future_size))
        enc_score_stack = []
        dec_score_stack = []

        # Encoder -> time
        for enc_step in range(self.enc_steps):
            enc_hidden, enc_cell, enc_score = self.encoder(
                rgb_inputs[:, enc_step],
                flow_inputs[:, enc_step],
                future_input, enc_hidden, enc_cell,
            )
            enc_score_stack.append(enc_score)

            # Decoder -> time t + 1
            dec_hidden = self.hidden_transform(enc_hidden)
            dec_cell = self.cell_transform(enc_cell)
            fusion_input = rgb_inputs.new_zeros((batch_size, self.hidden_size))
            future_input = rgb_inputs.new_zeros((batch_size, self.future_size))
            for dec_step in range(self.dec_steps):
                dec_hidden, dec_cell, dec_score = self.decoder(fusion_input, dec_hidden, dec_cell)
                dec_score_stack.append(dec_score)
                fusion_input = self.fusion_linear(dec_score)
                future_input = future_input + self.future_linear(dec_hidden)
            future_input = future_input / self.dec_steps

        enc_scores = torch.stack(enc_score_stack, dim=1).view(-1, self.num_classes)
        dec_scores = torch.stack(dec_score_stack, dim=1).view(-1, self.num_classes)
        return enc_scores, dec_scores
