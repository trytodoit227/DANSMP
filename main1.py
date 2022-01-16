from Model import *
from utils import *
import pickle
import torch
from Layers import *
from torch import optim
import argparse

##note that:if you want to run this code on CSI300E,please change all 73 (the number of firm nodes on CSI100E) to 185 (the number of firm nodes on CSI300E)

parser = argparse.ArgumentParser()

parser.add_argument('--grid-search', type=int, default='0',
                    help='0 False. 1 True')
parser.add_argument('--optim', type=int, default='1',
                    help='0 SGD. 1 Adam')
parser.add_argument('--eval', type=int, default='1',
                    help='if set the last day as eval')
parser.add_argument('--max-epoch', type=int, default='400',
                    help='Training max epoch')
parser.add_argument('--wait-epoch', type=int, default='30',
                    help='Training min epoch')
parser.add_argument('--eta', type=float, default='1e-4',
                    help='Early stopping')
parser.add_argument('--lr', type=float, default='8e-4',
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='0',
                    help='GPU to use')
parser.add_argument('--heads-att', type=int, default='2',
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='39',
                    help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='78',
                    help='rnn hidden nodes')
parser.add_argument('--weight-constraint', type=float, default='0.00098',
                    help='L2 weight constraint')
parser.add_argument('--rnn-length', type=int, default='20',
                    help='rnn length')
parser.add_argument('--dropout', type=float, default='0.3',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='0.45',
                    help='rnn clip')
parser.add_argument('--save', type=bool, default=True,
                    help='save model')

##note that:if you want to run this code on CSI300E,you should change all the path of data

def load_dataset(device1):
    with open('./data/x_num_standard.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open('./data/y_1.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('./data/x_newtext.pkl', 'rb') as handle:
        stock_sentiments = pickle.load(handle)
    with open('./data/edge_new.pkl', 'rb') as handle:
        edge_list=pickle.load(handle)
    with open('./data/interactive.pkl', 'rb') as handle:##the information of executives working in the company
        interactive_metric=pickle.load(handle)

    markets = markets.astype(np.float64)
    x = torch.tensor(markets, device=device1)
    x.to(torch.double)
    x_sentiment = torch.tensor(stock_sentiments, device=device1)
    x_sentiment.to(torch.double)
    y = torch.tensor(y_load, device=device1).squeeze()
    y = (y>0).to(torch.long)
    inter_metric=torch.tensor(interactive_metric,device=device1)
    inter_metric=inter_metric.squeeze(2)
    inter_metric=inter_metric.transpose(0, 1)
    return x, y, x_sentiment,edge_list,inter_metric


def train(model, x_train, x_sentiment_train, y_train, edge_list,inter_metric,device1):
    model.train()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length:]
    random.shuffle(train_seq)
    total_loss = 0
    total_loss_count = 0
    batch_train = 15
    for i in train_seq:
        output= model(x_train[i - rnn_length + 1: i + 1], x_sentiment_train[i - rnn_length + 1: i + 1], edge_list,inter_metric,device1)
        loss = criterion(output, y_train[i][:73])
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return total_loss / total_loss_count

def evaluate(model, x_eval, x_sentiment_eval, y_eval,edge_list,device1):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []
    for i in seq:
        output= model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], edge_list,inter_metric,device1)
        output = output.detach().cpu()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i][:73].cpu().numpy())
    acc, auc = metrics(trues, preds)
    return acc,  auc


if __name__=="__main__":
    args = parser.parse_args()
    device1 = "cuda:" + args.device
    device1=device1
    print(device1)
    criterion = torch.nn.NLLLoss()
    set_seed(1021)
    # load dataset
    print("loading dataset")
    x, y, x_sentiment,edge_list,inter_metric = load_dataset(device1)
    # hyper-parameters
    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)
    MAX_EPOCH =  args.max_epoch
    hidn_rnn = args.hidn_rnn
    N_heads = args.heads_att
    hidn_att= args.hidn_att
    lr = args.lr
    rnn_length = args.rnn_length
    t_mix = 1
    edge_list=edge_list

    #train-valid-test split
    x_train = x[: -100]
    x_eval = x[-100 - rnn_length : -50]
    x_test = x[-50 - rnn_length:]

    y_train = y[: -100]
    y_eval = y[-100 - rnn_length : -50]
    y_test = y[-50 - rnn_length:]

    x_sentiment_train = x_sentiment[: -100]
    x_sentiment_eval = x_sentiment[-100 - rnn_length : -50]
    x_sentiment_test = x_sentiment[-50 - rnn_length:]

    ## initialize
    best_model_file = 0
    epoch = 0
    wait_epoch = 0
    eval_epoch_best = 0

    model = GraphCNN(num_stock=NUM_STOCK, d_market = D_MARKET,d_news= D_NEWS,out_c=2,
                      d_hidden = D_MARKET*2, hidn_rnn = hidn_rnn, hid_c= hidn_att, n_heads=N_heads,dropout = args.dropout,t_mix = t_mix)

    model.cuda(device=device1)

    model.to(torch.double)
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.weight_constraint)



    #train
    while epoch < MAX_EPOCH:
        train_loss= train(model, x_train,x_sentiment_train, y_train,edge_list,inter_metric,device1)
        eval_acc, eval_auc = evaluate(model, x_eval, x_sentiment_eval, y_eval,edge_list,device1)
        test_acc, test_auc = evaluate(model, x_test, x_sentiment_test, y_test,edge_list,device1)
        eval_str = "epoch{},train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(epoch,train_loss, eval_auc, eval_acc, test_auc, test_acc)
        print(eval_str)

        if eval_auc > eval_epoch_best:
            eval_epoch_best = eval_auc
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(epoch, train_loss, eval_auc,eval_acc, test_auc, test_acc)
            wait_epoch = 0
            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./SavedModels/eval:auc{:.4f}_acc{:.4f}_test:auc{:.4f}_acc{:.4f}".format(eval_auc, eval_acc, test_auc, test_acc)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch >=50:
            print("saved_model_result:",eval_best_str)
            break
        epoch += 1