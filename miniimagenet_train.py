import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from tqdm import tqdm
from meta import Meta
import pandas as pd


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    print("The seed is ",args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(args)

    train_accs = pd.DataFrame(columns = [str(a) + ' Step' for a in range(args.update_step+1)])
    valid_accs = pd.DataFrame(columns = [str(a) + ' Step' for a in range(args.update_step_test+1)])
    train_index = 0
    valid_index = 0

    
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),

        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),

        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),

        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),

        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        
        ('flatten', []),
        ('linear', [args.n_way, 32 * 9 * 9])
    ]

    device = torch.device('cuda:1')
    maml = Meta(args, config).to(device)
    
#     # Add one layer attempt
#     print("Before")
#     print(maml.net.config)
#     print(maml.net.vars)
#     conv = maml.net.add_layer('conv2d', [32, 32, 3, 3, 1, 1], (0 + 1) * 2)
#     maml.meta_optim.add_param_group({"params":conv})
#     maml.net.add_layer('relu', [True], 1 + (0 + 1) * 2 + 1)
#     maml.to(device)
#     maml.net.vars.to(device)
#     print("Now")
#     print(maml.net.config)
#     print(maml.net.vars)
    
    
    
    
    

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    add_layer_count = 0
    for epoch in tqdm(range(args.epoch//10000)):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(tqdm(db)):
            torch.cuda.empty_cache()
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)
            train_accs.loc[train_index] = accs.tolist()
            train_index += 1
            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)

            if step % 500 == 499:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                # Store validation accuracy
                valid_accs.loc[valid_index] = accs.tolist()
                valid_index += 1
                print('Test acc:', accs)
                # Save the result
                train_accs.to_csv('./expdata/train_grow_seed'+str(args.seed)+".csv")
                valid_accs.to_csv('./expdata/valid_grow_seed'+str(args.seed)+".csv")
              
            # Network Growth
            # 2, 8, 14, 20
            # Record at 999,1999,2500+999, 2500+1999
            if add_layer_count < 4 and step % 1000 == 999: # trigger
                conv = maml.net.add_layer('conv2d', [32, 32, 3, 3, 1, 1], add_layer_count * 6 + 2, disruption=False)
                maml.meta_optim.add_param_group({"params":conv})
                maml.net.add_layer('relu', [True], add_layer_count * 6 + 3)
                maml.to(device)
                maml.net.vars.to(device)
                print("Now")
                print(maml.net.config)
                print(maml.net.vars)

                add_layer_count += 1
                
#             maml.leaner.add_layer('bn', [32, 32, 3, 3, 1, 0], 14)

    train_accs.to_csv('./expdata/train_grow_seed'+str(args.seed)+".csv")
    valid_accs.to_csv('./expdata/valid_grow_seed'+str(args.seed)+".csv")

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    argparser.add_argument('--seed', type=int, help='seed', default=0)

    args = argparser.parse_args()

    main()
