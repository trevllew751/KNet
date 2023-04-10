import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init

nGRU = 2

class KNet_RNN(torch.nn.Module):
    # def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    def __init__(self):
        super(KNet_RNN, self).__init__()
        # TODO: change these bruh and also dt
        self.dt = 0
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.num_layers = num_layers
        # self.output_dim = output_dim

    def Initialize(self, SysModel, x0):
        self.InitSystemDynamics(SysModel)
        self.InitSequence(x0)
        self.InitRNN(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S )

    def InitSystemDynamics(self, SysModel):
        self.f = SysModel.f
        self.h = SysModel.h
        self.Q = SysModel.Q
        self.R = SysModel.R
        self.m = SysModel.m
        self.n = SysModel.n

    def InitSequence(self, x0):
        # initialize with first row'
        # print(x0[:15].size())
        # print(x0[:15])
        self.x_post = torch.tensor(x0[:15])
        self.x_post_prev = self.x_post
        self.x_prior_prev = self.x_post
        self.y_prev = self.h(self.x_post) #need to configure so that h outputs the correct observation possibly with second input or iterating through each output
        #print(f"x_post: {self.x_post}")
        #print(f"x_post_prev: {self.x_post_prev}")
        #print(f"x_prior_prev: {self.x_prior_prev}")
        #print(f"y_prev: {self.y_prev}")

    def InitRNN(self, priorQ, priorSig, priorS):

        self.num_layers = 1
        i_d_mult = 5 # input dimension multiplier
        o_d_mult = 40 # output dimension multiplier

        #self.hidden = torch.zeros(num_layers, hidden_size)

        self.prior_Q = priorQ
        self.prior_Sigma = priorSig
        self.prior_S = priorS

        ### Define Kalman Gain Network ###
        # GRU to track Q
        self.Q_GRU_in_d = self.m * i_d_mult
        self.Q_GRU_h_d = self.m ** 2
        self.Q_GRU = nn.GRU(self.Q_GRU_in_d, self.Q_GRU_h_d, self.num_layers)

        # GRU to track Sigma
        self.Sig_GRU_in_d = self.Q_GRU_h_d + self.m * i_d_mult
        self.Sig_GRU_h_d = self.m ** 2
        self.Sig_GRU = nn.GRU(self.Sig_GRU_in_d, self.Sig_GRU_h_d, self.num_layers)

        # GRU to track S
        self.S_GRU_in_d = self.n ** 2 + 2 * self.n * i_d_mult
        self.S_GRU_h_d = self.n ** 2
        self.S_GRU = nn.GRU(self.S_GRU_in_d, self.S_GRU_h_d, self.num_layers)

        # Fully connected 1
        self.d_in_FC1 = self.Sig_GRU_h_d
        self.d_out_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(nn.Linear(self.d_in_FC1, self.d_out_FC1), nn.ReLU())

        # Fully connected 2
        self.d_in_FC2 = self.S_GRU_h_d + self.Sig_GRU_h_d
        self.d_out_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_in_FC2 * o_d_mult
        self.FC2 = nn.Sequential(nn.Linear(self.d_in_FC2, self.d_hidden_FC2), nn.ReLU(), nn.Linear(self.d_hidden_FC2, self.d_out_FC2))

        # Fully connected 3
        self.d_in_FC3 = self.S_GRU_h_d + self.d_out_FC2
        self.d_out_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(nn.Linear(self.d_in_FC3, self.d_out_FC3), nn.ReLU())

        # Fully connected 4
        self.d_in_FC4 = self.Sig_GRU_h_d + self.d_out_FC3
        self.d_out_FC4 = self.Sig_GRU_h_d
        self.FC4 = nn.Sequential(nn.Linear(self.d_in_FC4, self.d_out_FC4), nn.ReLU())
        
        # Fully connected 5
        self.d_in_FC5 = self.m
        self.d_out_FC5 = self.m * i_d_mult
        self.FC5 = nn.Sequential(nn.Linear(self.d_in_FC5, self.d_out_FC5), nn.ReLU())

        # Fully connected 6
        self.d_in_FC6 = self.m
        self.d_out_FC6 = self.m * i_d_mult
        self.FC6 = nn.Sequential(nn.Linear(self.d_in_FC6, self.d_out_FC6), nn.ReLU())
        
        # Fully connected 7
        self.d_in_FC7 = 2 * self.n
        self.d_out_FC7 = 2 * self.n * i_d_mult
        self.FC7 = nn.Sequential(nn.Linear(self.d_in_FC7, self.d_out_FC7), nn.ReLU())

        # Apply He initialization to FC layers
        # FC1
        init.kaiming_uniform_(self.FC1[0].weight, nonlinearity='relu')
        self.FC1[0].bias.data.fill_(0)
        # FC2
        init.kaiming_uniform_(self.FC2[0].weight, nonlinearity='relu')
        self.FC2[0].bias.data.fill_(0)
        init.kaiming_uniform_(self.FC2[2].weight, nonlinearity='relu')
        self.FC2[2].bias.data.fill_(0)
        # FC3
        init.kaiming_uniform_(self.FC3[0].weight, nonlinearity='relu')
        self.FC3[0].bias.data.fill_(0)
        # FC4
        init.kaiming_uniform_(self.FC4[0].weight, nonlinearity='relu')
        self.FC4[0].bias.data.fill_(0)
        # FC5
        init.kaiming_uniform_(self.FC5[0].weight, nonlinearity='relu')
        self.FC5[0].bias.data.fill_(0)
        # FC6
        init.kaiming_uniform_(self.FC6[0].weight, nonlinearity='relu')
        self.FC6[0].bias.data.fill_(0)
        # FC7
        init.kaiming_uniform_(self.FC7[0].weight, nonlinearity='relu')
        self.FC7[0].bias.data.fill_(0)

        """
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
        """

    def KNet_Step(self, y, hidden):
        # Calculate priors
        # print(f"x_post {self.x_post}")
        # self.x_prior = torch.tensor(self.f(self.x_post, self.v, self.w_ib_b, self.f_ib_b, self.dt/1000))
        v = y[15: 21]
        w_ib_b = y[21: 24]
        f_ib_b = y[24:]
        # print(v)
        # print(w_ib_b)
        # print(f_ib_b)

        #step prior
        self.x_prior = torch.tensor(self.f(self.x_post, v, w_ib_b, f_ib_b, self.dt/1000))
        # print(f"x_prior {self.x_prior}")
        self.y = self.h(self.x_prior)

        self.dt += 1

        #Calculate Kalman Gain
        self.KGain_Step(y, hidden)

        #Innovation
        dy = (y[:15] - self.y).reshape(1, 15)

        # print(f"KG {self.KG.size()}")
        # print(f"dy {dy.size()}")

        inov = torch.mul(self.KG, dy)

        # print(f"inov {inov}")
        # print(f"x_post {self.x_post.size()}")
        # print(f"x_prior {self.x_prior}")

        self.x_post_prev = self.x_post
        self.x_post = (self.x_prior + inov)[0]

        self.x_prior_prev = self.x_prior
        self.y_prev = y[:15]
        # print(f"x_post {self.x_post}")

        return self.x_post.reshape(1,15), hidden

    def KGain_Step(self, y, hidden):
        
        #print("The size of y[:15] is ", y[:15].size())
        #print("The size of self.y is ", self.y.size())
        #print("The size of self.y_prev is ", self.y_prev.size())
        #print("The size of self.x_post is ", self.x_post.size())
        #print("The size of self.x_post_prev is ", self.x_post_prev.size())
        #print("The size of self.x_prior_prev is ", self.x_prior_prev.size())
        
        obs_diff = y[:15] - self.y_prev
        obs_innov_diff = y[:15] - self.y
        
        fw_evol_diff = self.x_post - self.x_post_prev
        fw_update_diff = self.x_post - self.x_prior_prev
       
        """
        # both in size [batch_size, n]
        obs_diff = torch.squeeze(y[:15],2) - torch.squeeze(self.y_prev,2) 
        obs_innov_diff = torch.squeeze(y[:15],2) - torch.squeeze(self.y,2)
        # both in size [batch_size, m]
        fw_evol_diff = torch.squeeze(self.x_post,2) - torch.squeeze(self.x_post_prev,2)
        fw_update_diff = torch.squeeze(self.x_post,2) - torch.squeeze(self.x_prior_prev,2)

        obs_diff = func.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = func.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)
        """
        KG = self.KGain(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        # KG_input = torch.nn.functional.normalize(self.x_post - self.x_prior_prev, p=2, dim=1, eps=1e-12, out=None)
        # print(f"x_prior_prev {self.x_prior_prev}")
        #KG_input = torch.nn.functional.normalize(self.x_post - self.x_prior_prev, p=2, dim=0).type(torch.FloatTensor)
        #if any(torch.isnan(KG)):
            # print(f"dt {self.dt}")
            #dy = (y[:15] - self.y).reshape(1, 15)
            #print(f"KG {KG}")
            # print(f"KG {self.KG}")
            # print(f"dy {dy}")
            # print(f"y {y}")
            # print(f"x_prior {self.x_prior}")
            # print(f"self.y {self.y}")
            # print(f"innov {torch.mul(self.KG, dy)}")
            # print(f"x_post {self.x_post}")
            # print(f"x_prior_prev {self.x_prior_prev}")
            # print(f"x_post - x_prior_prev {self.x_post - self.x_prior_prev}")
            #raise Exception("NAN in KG")
        #KG = self.KGain(KG_input)

        self.KG = torch.reshape(KG, (self.m, self.n))
        #print("self.KG size: ", self.KG.size())
        

    def KGain(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        def expand_dim(x):
            expanded = torch.empty(1, 1, x.shape[-1])
            expanded[0, :, :] = x
            return expanded
        

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        #print("obs_diff size: ", obs_diff.size())
        #print("obs_innov_diff size: ", obs_innov_diff.size())
        #print("fw_evol_diff size: ", fw_evol_diff.size())
       # print("fw_update_diff size: ", fw_update_diff.size())

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)
        #print("out_FC5 size: ", out_FC5.size())

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.Q_GRU(in_Q, self.h_Q)
        #print("out_Q size: ", out_Q.size())
        #print("self.h_Q size: ", self.h_Q.size())

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)
        #print("out_FC6 size: ", out_FC6.size())

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        #print("in_Sigma size: ", in_Sigma.size())
        out_Sigma, self.h_Sigma = self.Sig_GRU(in_Sigma, self.h_Sigma)
        #print("out_Sigma size: ", out_Sigma.size())
        #print("self.h_Sigma size: ", self.h_Sigma.size())

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)
        #print("out_FC1 size: ", out_FC1.size())

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)
        #print("out_FC7 size: ", out_FC7.size())


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.S_GRU(in_S, self.h_S)
        #print("out_S size: ", out_S.size())


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)
        #print("out_FC2 size: ", out_FC2.size())

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2 # m*n KG output
    
        """
        input_out = self.fc_state_in(KG_input)
        input_out = input_out.unsqueeze(0)
        # print(input_out)
        rnn_out, self.hidden = self.RNN_State(input_out, hidden)
        KG = self.fc_state_out(rnn_out)
        return KG
        """

    def forward(self, y, hidden):
        return self.KNet_Step(y, hidden)
    
    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, 1, self.S_GRU_h_d).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(1,1, 1) # batch size expansion
        hidden = weight.new(1, 1, self.Sig_GRU_h_d).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1,1, -1).repeat(1,1, 1) # batch size expansion
        hidden = weight.new(1, 1, self.Q_GRU_h_d).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1,1, -1).repeat(1,1, 1) # batch size expansion
