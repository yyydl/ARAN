import warnings
warnings.filterwarnings("ignore")
from apex import amp
import torch.utils.data as data
from torchvision import transforms
import os, torch
import argparse
import DataSet
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import ARAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=70, help='Total training epochs.')
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()


def run_training():
    args = parse_args()
    if args.wandb:
        import wandb
        wandb.init(project='raf-db')
    model = ARAN()
    print(model)
    print("batch_size:", args.batch_size)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1))
    ])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)
    val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)

    train_num = train_dataset.__len__()
    print('Train set size:', train_num)
    val_num = val_dataset.__len__()
    print('Validation set size:', val_num)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=args.val_batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)
    
    params = model.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    print(optimizer)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model = model.cuda()
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    CE_criterion = torch.nn.CrossEntropyLoss()
    LCE_criterion = LabelSmoothingCrossEntropy()

    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs = model(imgs)
            targets = targets.cuda()
            loss = LCE_criterion(outputs, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss/iter_cnt
        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f' %
              (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))
        scheduler.step()
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                outputs = model(imgs.cuda())
                targets = targets.cuda()
                loss = LCE_criterion(outputs, targets)
                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()

            val_loss = val_loss/iter_cnt
            val_acc = bingo_cnt.float()/float(val_num)
            val_acc = np.around(val_acc.numpy(), 4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, val_acc, val_loss))

            if args.wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                )

            if val_acc > 0.88 or val_acc > best_acc:
            # if val_acc > best_acc:
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('./RAF-DB', "epoch" + str(i) + "_acc" + str(val_acc) + ".pth"))
                print('Model saved.')
            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc:" + str(best_acc))

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        loss = -log_pred.sum(dim=-1)
        loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_pred, target,
                                                               reduction=self.reduction,
                                                               ignore_index=self.ignore_index)


if __name__ == "__main__":
    run_training()
