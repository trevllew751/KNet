import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init

nGRU = 2

class KNet_RNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
    #def __init__(self):
        super(KNet_RNN, self).__init__()
        # TODO: change these bruh and also dt
        self.dt = 0
        self.input_dim = input_dim
        self.hidden_neurons = input_dim ** 2
        self.output_dim = output_dim
        self.i_d_mult = 5*self.input_dim # input dimension multiplier
        self.o_d_mult = 40*(self.input_dim**2 + self.output_dim**2) # output dimension multiplier
        self.GRU_layers = 1

        #first FC and GRU layer for forward evolution difference
        self.FC5 = nn.Sequential(nn.Linear(self.input_dim, self.i_d_mult), nn.ReLU())
        self.Q_GRU = nn.GRU(self.i_d_mult, self.hidden_neurons, self.GRU_layers)

        #Second FC and GRU layer for forward update difference
        self.FC6 = nn.Sequential(nn.Linear(self.input_dim, self.i_d_mult), nn.ReLU())
        self.Sig_GRU = nn.GRU(self.i_d_mult + self.hidden_neurons, self.hidden_neurons, self.GRU_layers)

        #Third FC layer for Sigma output
        self.FC1 = nn.Sequential(nn.Linear(self.input_dim**2, self.output_dim**2), nn.ReLU())

        #Fourth FC layer for observation and innovation difference and Third GRU layer for S
        self.FC7 = nn.Sequential(nn.Linear(self.output_dim*2, 2*self.i_d_mult), nn.ReLU())
        self.S_GRU = nn.GRU(2*self.i_d_mult + self.output_dim**2, self.output_dim**2, self.GRU_layers)

        #Output layer
        self.FC2 = nn.Sequential(nn.Linear(self.input_dim**2 + self.output_dim**2, self.o_d_mult), nn.ReLU(), nn.Linear(self.o_d_mult, self.input_dim*self.output_dim))

        #Backwards FC layers
        self.FC3 = nn.Sequential(nn.Linear(self.output_dim**2 + self.input_dim*self.output_dim, self.input_dim**2), nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(self.hidden_neurons + self.input_dim**2, self.input_dim**2), nn.ReLU())


    def Initialize(self, SysModel, x0):
        self.InitSystemDynamics(SysModel)
        self.InitSequence(x0)
        self.InitRNN()

    def InitSystemDynamics(self, SysModel):
        self.f = SysModel.f
        self.h = SysModel.h
        self.Q = SysModel.Q
        self.R = SysModel.R
        self.m = SysModel.m
        self.n = SysModel.n
        self.prior_Q = SysModel.prior_Q
        self.prior_Sigma = SysModel.prior_Sigma
        self.prior_S = SysModel.prior_S

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

    def InitRNN(self):

        self.batch_size = 1
        #i_d_mult = 5 # input dimension multiplier
        #o_d_mult = 40 # output dimension multiplier

        #self.hidden = torch.zeros(num_layers, hidden_size)
        """
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
        """
        ### Initialize network parameters ###
        # Apply Xavier initialization to GRU layers
        # GRU Q
        for name, param in self.Q_GRU.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        # GRU Sigma
        for name, param in self.Sig_GRU.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        # GRU S
        for name, param in self.S_GRU.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

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

    def KG_Step_est(self, y, hidden):
        # both in size [batch_size, n]
        #obs_diff = torch.squeeze(y[:15]) - torch.squeeze(self.y_prev) 
        #obs_innov_diff = torch.squeeze(y[:15]) - torch.squeeze(self.y)
        # both in size [batch_size, m]
        #fw_evol_diff = torch.squeeze(self.x_post) - torch.squeeze(self.x_post_prev)
        #fw_update_diff = torch.squeeze(self.x_post) - torch.squeeze(self.x_prior_prev)

        obs_diff = y[:15] - self.y_prev
        obs_innov_diff = y[:15] - self.m1y
        
        fw_evol_diff = self.x_post - self.x_post_prev
        fw_update_diff = self.x_post - self.x_prior_prev

        #print("obs_diff size: ", obs_diff.size())
        #print("obs_innov_diff size: ", obs_innov_diff.size())
        #print("fw_evol_diff size: ", fw_evol_diff.size())
        #print("fw_update_diff size: ", fw_update_diff.size())

        #obs_diff = func.normalize(obs_diff, p=2, dim=0, eps=1e-12, out=None)
        #obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=0, eps=1e-12, out=None)
        #fw_evol_diff = func.normalize(fw_evol_diff, p=2, dim=0, eps=1e-12, out=None)
        #fw_update_diff = func.normalize(fw_update_diff, p=2, dim=0, eps=1e-12, out=None)

        #print("obs_diff normalized size: ", obs_diff.size())
        #print("obs_innov_diff normalized size: ", obs_innov_diff.size())
        #print("fw_evol_diff normalized size: ", fw_evol_diff.size())
        #print("fw_update_diff normalized size: ", fw_update_diff.size())

        KG = self.KGain_Step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)

        #print("KG size: ", KG.size())
        self.KG = torch.reshape(KG, (self.m, self.n))
        #print("self.KG size: ", self.KG.size())


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

        #step prior - 1st moment of x and y
        self.x_prior = torch.tensor(self.f(self.x_post, v, w_ib_b, f_ib_b, self.dt/1000))
        # print(f"x_prior {self.x_prior}")
        self.m1y = self.h(self.x_prior)

        self.dt += 1

        #Calculate Kalman Gain
        #print("The size of hidden is: ", hidden.size())
        #self.KGain_Step(y, hidden)
        self.KG_Step_est(y, hidden)

        #Innovation
        dy = (y[:15] - self.m1y)
        # print(f"KG {self.KG.size()}")
        #print(f"dy {dy.size()}")

        #Calculate 1st posterior moment
        inov = torch.mul(self.KG, dy)

        # print(f"inov {inov}")
        # print(f"x_post {self.x_post.size()}")
        # print(f"x_prior {self.x_prior}")

        self.x_post_prev = self.x_post
        self.x_post = (self.x_prior + inov)[0]

        self.x_prior_prev = self.x_prior
        self.y_prev = y[:15]
        #print(f"x_post size: {self.x_post.size()}")

        return self.x_post, hidden

    def KGain_Step(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        
        #print("The size of y[:15] is ", y[:15].size())
        #print("The size of self.y is ", self.y.size())
        #print("The size of self.y_prev is ", self.y_prev.size())
        #print("The size of self.x_post is ", self.x_post.size())
        #print("The size of self.x_post_prev is ", self.x_post_prev.size())
        #print("The size of self.x_prior_prev is ", self.x_prior_prev.size())
       
        def expand_dim(x):
            expanded = torch.empty(self.GRU_layers, self.batch_size, x.shape[-1])
            expanded[0, :, :] = x
            return expanded
        
        #torch.Size([layers, 1, m=15])
        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        #print("obs_diff expanded size: ", obs_diff.size())
        #print("obs_innov_diff expanded size: ", obs_innov_diff.size())
        #print("fw_evol_diff expanded size: ", fw_evol_diff.size())
        #print("fw_update_diff expanded size: ", fw_update_diff.size())

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5 = forward evolution difference (INPUT)
        in_FC5 = fw_evol_diff.float()
        #print("in_FC5 size: ", in_FC5.size())
        out_FC5 = self.FC5(in_FC5)  #torch.Size([layers, 1, m*in_mult = 75])
        #print("out_FC5 size: ", out_FC5.size())

        # Q-GRU
        in_Q = out_FC5
        #print("in_Q size: ", in_Q.size())
        out_Q, self.h_Q = self.Q_GRU(in_Q, self.h_Q)  #torch.Size([layers, 1, m^2 = 225])
        #print("out_Q size: ", out_Q.size())
        #print("self.h_Q size: ", self.h_Q.size())

        # FC 6 = forward update difference (INPUT)
        in_FC6 = fw_update_diff
        #print("in_FC6 size: ", in_FC6.size())
        out_FC6 = self.FC6(in_FC6)  #torch.Size([layers, 1, m*in_mult = 75])
        #print("out_FC6 size: ", out_FC6.size())

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        #print("in_Sigma size: ", in_Sigma.size())
        out_Sigma, self.h_Sigma = self.Sig_GRU(in_Sigma, self.h_Sigma) #torch.Size([layers, 1, m^2 = 225])
        #print("out_Sigma size: ", out_Sigma.size())
        #print("self.h_Sigma size: ", self.h_Sigma.size())

        # FC 1
        in_FC1 = out_Sigma
        #print("in_FC1 size: ", in_FC1.size())
        out_FC1 = self.FC1(in_FC1)  #torch.Size([layers, 1, n^2 = 225])
        #print("out_FC1 size: ", out_FC1.size())

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        #print("in_FC7 size: ", in_FC7.size())
        out_FC7 = self.FC7(in_FC7)  #torch.Size([layers, 1, n*2*in_mult = 150])
        #print("out_FC7 size: ", out_FC7.size())


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        #print("in_S size: ", in_S.size())
        out_S, self.h_S = self.S_GRU(in_S, self.h_S)  #torch.Size([layers, 1, n^2 = 225])
        #print("out_S size: ", out_S.size())


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        #print("in_FC2 size: ", in_FC2.size())
        out_FC2 = self.FC2(in_FC2) #torch.Size([layers, 1, m*n = 225])
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

        #combine output nodes
        #combine_nodes = nn.AdaptiveAvgPool3d((1, self.m, self.n))
        #KG = torch.sigmoid(out_FC2)
        #print("K_GainStep KG size: ", KG.size())
        #print("Output layer size: ", out_FC2.size())

        KG = torch.reshape(out_FC2, (1, self.m, self.n))
        #y2 = torch.reshape(out_FC4[1], (1, self.m, self.n))
        #print("Output y1 size: ", y1.size())
        #print("Output y2 size: ", y2.size())

        return KG# ( m * n) KG output


        
        # KG_input = torch.nn.functional.normalize(self.x_post - self.x_prior_prev, p=2, dim=1, eps=1e-12, out=None)
        # print(f"x_prior_prev {self.x_prior_prev}")
        #KG_input = torch.nn.functional.normalize(self.x_post - self.x_prior_prev, p=2, dim=0).type(torch.FloatTensor)
        if (torch.isnan(KG).any()):
            # print(f"dt {self.dt}")
            #dy = (y[:15] - self.y).reshape(1, 15)
            print(f"KG {KG}")
            #print(f"KG {self.KG}")
            # print(f"dy {dy}")
            print(f"y {y}")
            # print(f"x_prior {self.x_prior}")
            print(f"self.y {self.y}")
            # print(f"innov {torch.mul(self.KG, dy)}")
            # print(f"x_post {self.x_post}")
            # print(f"x_prior_prev {self.x_prior_prev}")
            # print(f"x_post - x_prior_prev {self.x_post - self.x_prior_prev}")
            raise Exception("NAN in KG")
        #KG = self.KGain(KG_input)

        
        

    def KGain(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
             
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
        
        hidden = weight.new(self.GRU_layers, self.batch_size, self.hidden_neurons).zero_()
        #print("The size of h_S is: ", hidden.size())
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.GRU_layers,self.batch_size, 1) # batch size expansion
        #print("The size of self.h_S is: ", self.h_S.size())
        
        hidden = weight.new(self.GRU_layers, self.batch_size, self.hidden_neurons).zero_()
        #print("The size of h_Sigma is: ", hidden.size())
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1,1, -1).repeat(self.GRU_layers,self.batch_size, 1) # batch size expansion
        #print("The size of self.h_Sigma is: ", self.h_Sigma.size())
        
        hidden = weight.new(self.GRU_layers, self.batch_size, self.hidden_neurons).zero_()
        #print("The size of h_Q is: ", hidden.size())
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1,1, -1).repeat(self.GRU_layers,self.batch_size, 1) # batch size expansion
        #print("The size of self.h_Q is: ", self.h_Q.size())
