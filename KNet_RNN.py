import torch
from torch import nn


class KNet_RNN(torch.nn.Module):
    # def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    def __init__(self):
        super(KNet_RNN, self).__init__()
        # TODO: change these bruh and also dt
        self.dt = 0
        self.v = torch.zeros(1, 6)
        self.w_ib_b = torch.randn(1, 3)
        self.f_ib_b = torch.randn(1, 3)
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.num_layers = num_layers
        # self.output_dim = output_dim

    def Initialize(self, SysModel, x0):
        self.InitSystemDynamics(SysModel)
        self.InitSequence(x0)
        self.InitRNN()

    def InitSequence(self, x0):
        # initialize with first row
        self.x_post = torch.tensor(x0)
        self.x_post_prev = self.x_post
        self.x_prior_prev = self.x_post
        self.y_prev = self.h(self.x_post)
        # print(f"x_post {self.x_post}")
        # print(f"x_post_prev {self.x_post_prev}")
        # print(f"x_prior_prev {self.x_prior_prev}")
        # print(f"y_prev {self.y_prev}")

    def InitSystemDynamics(self, SysModel):
        self.f = SysModel.f
        self.h = SysModel.h
        self.Q = SysModel.Q
        self.R = SysModel.R
        # print(f"f {self.f}")
        # print(f"h {self.h}")
        # print(f"Q {self.Q}")
        # print(f"R {self.R}")

    def InitRNN(self):
        input_size = self.x_post.size(0)
        output_size = input_size
        hidden_size = input_size ** 2
        num_layers = 3
        bias = True
        batch_first = None
        dropout = 0
        bidirectional = False

        self.hidden = torch.randn((2 if bidirectional else 1) * num_layers, hidden_size)

        self.RNN_State = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # bias=bias,
            # batch_first=batch_first,
            # dropout=dropout,
            # bidirectional=bidirectional
        )

        self.fc_state_in = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        )

        self.fc_state_out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

        self.RNN_Q = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.fc_Q = nn.Linear(
            hidden_size,
            output_size
        )

    def KNet_Step(self, y, hidden):
        # Calculate priors
        # print(f"x_post {self.x_post}")
        self.x_prior = torch.tensor(self.f(self.x_post, self.v, self.w_ib_b, self.f_ib_b, self.dt/1000))
        # print(f"x_prior {self.x_prior}")
        self.y = self.h(self.x_prior)

        self.dt += 1

        self.KGain_Step(y, hidden)

        dy = (y - self.y).reshape(1, 15)

        # print(f"KG {self.KG.size()}")
        # print(f"dy {dy.size()}")

        inov = torch.mul(self.KG, dy)

        # print(f"inov {inov}")
        # print(f"x_post {self.x_post.size()}")
        # print(f"x_prior {self.x_prior}")

        self.x_post_prev = self.x_post
        self.x_post = (self.x_prior + inov)[0]

        self.x_prior_prev = self.x_prior
        self.y_prev = y
        # print(f"x_post {self.x_post}")

        return self.x_post.reshape(1,15), self.hidden

    def KGain_Step(self, y, hidden):
        # obs_diff = y - self.y_prev
        # obs_innov_diff = y - self.y
        #
        # fw_evolv_diff = self.x_post - self.x_post_prev
        # fw_update_diff = self.x_post - self.x_prior_prev
        #
        # obs_diff = torch.nn.functional.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        # obs_innov_diff = torch.nn.functional.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        # fw_evolv_diff = torch.nn.functional.normalize(fw_evolv_diff, p=2, dim=1, eps=1e-12, out=None)
        # fw_update_diff = torch.nn.functional.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        # KG = self.KGain(obs_diff, obs_innov_diff, fw_evolv_diff, fw_update_diff)
        # KG_input = torch.nn.functional.normalize(self.x_post - self.x_prior_prev, p=2, dim=1, eps=1e-12, out=None)
        # print(f"x_prior_prev {self.x_prior_prev}")
        KG_input = torch.nn.functional.normalize(self.x_post - self.x_prior_prev, p=2, dim=0)
        if any(torch.isnan(KG_input)):
            # print(f"dt {self.dt}")
            dy = (y - self.y).reshape(1, 15)
            print(f"KG_input {KG_input}")
            # print(f"KG {self.KG}")
            # print(f"dy {dy}")
            # print(f"y {y}")
            # print(f"x_prior {self.x_prior}")
            # print(f"self.y {self.y}")
            # print(f"innov {torch.mul(self.KG, dy)}")
            # print(f"x_post {self.x_post}")
            # print(f"x_prior_prev {self.x_prior_prev}")
            # print(f"x_post - x_prior_prev {self.x_post - self.x_prior_prev}")
            raise Exception("NAN in KG_input")
        KG = self.KGain(KG_input, hidden)

        self.KG = KG

    def KGain(self, KG_input, hidden):
        if hidden is None:
            hidden = self.hidden
        input_out = self.fc_state_in(KG_input)
        input_out = input_out.unsqueeze(0)
        # print(input_out)
        rnn_out, self.hidden = self.RNN_State(input_out, hidden)
        KG = self.fc_state_out(rnn_out)
        return KG

    def forward(self, y, hidden):
        return self.KNet_Step(y, hidden)
