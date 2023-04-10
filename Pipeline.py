import torch.nn
import matplotlib.pyplot as plt


class Pipeline():
    def __init__(self, SysModel, RNN, data):
        self.SysModel = SysModel
        self.RNN = RNN
        self.data = data

    def Initialize(self):
        self.RNN.Initialize(self.SysModel, self.data[0])

    def SetTrainingParams(self, n_steps, learning_rate, weight_decay):
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.RNN.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        #for name, param in self.RNN.named_parameters():
        # if param.requires_grad:
        #   print(name)
        

    def train(self):
        #feature_size = self.RNN.feature_size
        hidden = None
        self.RNN.train()
        self.RNN.init_hidden_KNet()

        N_B = len(self.data)

        MSE_training_loss_batch = torch.empty([N_B])
        self.training_loss = torch.empty([self.n_steps])
        self.training_loss_dB = torch.empty([self.n_steps])
        

        for e in range(self.n_steps):
            predictions = torch.zeros(1, self.SysModel.m)
            self.RNN.InitSequence(self.data[0])
            print(f"epoch: {e} out of {self.n_steps}")
            Batch_Optimizing_LOSS_sum = 0
            for i in range(0, N_B):
                print(f"row {i} out of {N_B}")
                row = self.data[i]
                #print(f"row size: {row.size()}")
                #print("Prediction size: ",predictions.size())
                #print("Predictions: ",predictions)
                prediction, hidden = self.RNN(row, hidden)
                prediction.reshape(1, self.SysModel.m)
                #hidden = hidden.data
                
                # print(predictions.size())
                predictions = torch.cat([predictions, prediction])
                loss = self.loss_fn(prediction, row[:15])
                print(f"Loss: {loss.item()}")
                MSE_training_loss_batch[i] = loss.item()
                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + loss
            # print(predictions.size())
            # print(predictions)
            self.training_loss[e] = torch.mean(MSE_training_loss_batch)
            self.training_loss_dB[e] = 10 * torch.log10(self.training_loss[e])
            
            
            #Optimizing
            self.optimizer.zero_grad()
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / N_B
            Batch_Optimizing_LOSS_mean.backward()
            self.optimizer.step()

            print(f"Test {e} Loss: {self.training_loss_dB}")
        print(f"KGain: {self.RNN.KG}")
        epochs = list(range(0, self.n_steps))
        plt.plot(epochs, self.training_loss, "r.")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (dB)")
        plt.grid(visible=True)
        plt.savefig("loss_plot.png")
