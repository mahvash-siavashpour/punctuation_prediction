
import torch
from torch import nn
import numpy as np
from datetime import datetime

from seqeval.metrics import performance_measure

from datasets import load_metric
metric = load_metric("seqeval")

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, use_cnn, dropout=0.1):
        super(LSTM_Model, self).__init__()

        self.use_cnn = use_cnn

        if use_cnn == "yes":
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, stride=1, padding="same"),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding="same", dilation=2),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size=2, stride=2)
            )
            self.dropout = nn.Dropout(dropout) 
            self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.dropout = nn.Dropout(dropout) 
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        #cnn takes input of shape (batch_size, channels, seq_len)
        # print(x.shape)
        if self.use_cnn == "yes":
            out = self.cnn(x)
            out = self.dropout(out) #
            # lstm takes input of shape (batch_size, seq_len, input_size)
            out = out.permute(0, 2, 1)
            # print(out.shape)
            out, _ = self.lstm(out)
        else:
            x = x.permute(0, 2, 1)
            x = self.dropout(x)
            out, _ = self.lstm(x)
        # print(out.shape)

        predictions  = self.fc(out)
        # tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        # print(predictions.shape)
        return predictions 




def train_one_epoch(epoch_index, train_loader, optimizer, model, loss_function, id2tag, num_classes):
    running_loss = 0.
    last_loss = 0.
    ave_f1 = 0
    cnt =0

    accumulated_outputs = []
    accumulated_ys= []
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        x = data['x']
        x = x.permute(0, 2, 1)
        y = data['y']

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(x)

        # Compute the loss and its gradients
        # print(outputs.view(-1, 5))
        # print(y.view(-1))
        # print(x.shape)
        # print(outputs.view(-1, 5).shape)
        # print(y.view(-1).shape)
        loss = loss_function(outputs.view(-1, num_classes), y.view(-1))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        new_y = y.detach().numpy()
        new_y = new_y.reshape(-1).tolist()
        notated_y = [id2tag[i] for i in new_y]
        new_outputs = outputs.detach().numpy()
        new_outputs = np.argmax(new_outputs,axis=2)
        new_outputs = new_outputs.reshape(-1).tolist()
        notated_outputs = [id2tag[i] for i in new_outputs]

        accumulated_outputs.append(notated_outputs)
        accumulated_ys.append(notated_y)

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            
            results = metric.compute(predictions=accumulated_outputs, references=accumulated_ys)
            # f1 = f1_score(y_true=new_y.reshape(-1), y_pred=new_outputs.reshape(-1), average='macro') 
            f1 = results['overall_f1']
            ave_f1 += f1
            # f1 = 0
            last_loss = running_loss / 1000 # loss per batch
            # print(f'  batch {i + 1} loss: {last_loss} f1-score: {f1}')
            tb_x = epoch_index * len(train_loader) + i + 1
            print(f"Loss/train {last_loss, tb_x} ")
            running_loss = 0.
            cnt += 1

            accumulated_outputs = []
            accumulated_ys= []
    
    if cnt > 0:
        ave_f1 /= cnt

    return last_loss, ave_f1



def train(epochs, model, train_loader, test_loader, optimizer, loss_function, id2tag, timestamp, best_vloss, num_classes):
    

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, avg_f1 = train_one_epoch(epoch, train_loader, optimizer, model, loss_function, id2tag, num_classes)
        print(f"Average F1: {avg_f1}")

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        f1 = 0

        accumulated_voutputs = []
        accumulated_vlabels= []

        for i, vdata in enumerate(test_loader):
            vinputs = vdata['x']
            vinputs = vinputs.permute(0, 2, 1)
            vlabels = vdata['y']
            voutputs = model(vinputs)
            vloss = loss_function(voutputs.view(-1, num_classes), vlabels.view(-1))
            running_vloss += vloss

            new_vlabels = vlabels.detach().numpy()
            new_vlabels = new_vlabels.reshape(-1).tolist()
            notated_vlabels = [id2tag[i] for i in new_vlabels]
            new_voutputs = voutputs.detach().numpy()
            new_voutputs = np.argmax(new_voutputs,axis=2)
            new_voutputs = new_voutputs.reshape(-1).tolist()
            notated_voutputs = [id2tag[i] for i in new_voutputs]

            accumulated_voutputs.append(notated_voutputs)
            accumulated_vlabels.append(notated_vlabels)

        
        results = metric.compute(predictions=accumulated_voutputs, references=accumulated_vlabels)
        f1 = results['overall_f1']
        # f1 = f1_score(y_true=new_vlabels.reshape(-1), y_pred=new_voutputs.reshape(-1), average='macro') 

        avg_vloss = running_vloss / (i + 1)

        pm = performance_measure(accumulated_vlabels, accumulated_voutputs)
        print(f"\n{pm}\n")
        TP = pm['TP']+pm['TN']
        TN = pm['TN']
        FP=pm['FP']
        FN=pm['FN']
        f1 = TP/(TP+(.5*(FP+FN)))
        f1_O = TN/(TN+(.5*(FP+FN)))
    
        print(f'LOSS=> train {avg_loss} valid {avg_vloss} \n ***Test metrics overall f1:{results["overall_f1"]}')
        print(f"overall f1 with TN: {f1}")
        print(f"O f1: {f1_O}")

   

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

    
    return model

