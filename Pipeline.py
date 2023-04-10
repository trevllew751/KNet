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
        """
        for name, param in self.RNN.named_parameters():
         if param.requires_grad:
            print(name)
        """

    def train(self):
        feature_size = self.RNN.feature_size
        hidden = None
        self.RNN.train()
        losses = []
        for e in range(self.n_steps):
            predictions = torch.zeros(1, feature_size)
            self.RNN.InitSequence(self.data[0])
            for i in range(0, self.data.size(0) - 1):
                # print(f"epoch: {e}, row: {i}")
                row = self.data[i]
                # print(f"row: {row}")
                prediction, hidden = self.RNN(row, hidden)
                prediction.reshape(1, feature_size)
                hidden = hidden.data
                # print(prediction.size())
                # print(predictions.size())
                predictions = torch.cat([predictions, prediction])
            # print(predictions.size())
            # print(predictions)
            loss = self.loss_fn(predictions[1:], self.data[:-1][:, :feature_size])
            loss = 10 * torch.log10(loss)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Test {e} Loss: {loss}")
        print(f"KGain: {self.RNN.KG}")
        epochs = list(range(0, self.n_steps))
        plt.plot(epochs, losses, "r.")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (dB)")
        plt.grid(visible=True)
        plt.savefig("loss_plot.png")
