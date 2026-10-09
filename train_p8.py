"""Audit Plan P8: Accuracy Completion.
Trains FP32 baselines and fine-tunes pruned models with a 45k/5k split and fixed seeds.
Power-loss resilient.

Usage: 
  python3 train_p8.py --arch resnet18 --epochs 30 --seed 42 --out checkpoints/resnet18_fp32_s42.pt
  python3 train_p8.py --arch resnet18 --epochs 25 --seed 42 --finetune-from checkpoints/resnet18_pruned50.pt --out checkpoints/resnet18_p50_ft_s42.pt
"""
import argparse, json, time, random
from pathlib import Path

import torch, torchvision
from torch import nn
from torchvision import transforms
from train_baseline import build_model, CIFAR_MEAN, CIFAR_STD

import power_state

def interrupted_path(out: str) -> Path:
    p = Path(out)
    return p.with_name(p.stem + '_interrupted.pt')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=['resnet18', 'mobilenet_v3_small', 'efficientnet_b0'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--data', default='./data')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--finetune-from', type=str, default=None, help="Pruned model to fine-tune")
    p.add_argument('--resume', type=str, default=None)
    a = p.parse_args()

    resume_ckpt = torch.load(a.resume, map_location='cpu', weights_only=False) if a.resume else None
    if resume_ckpt:
        a = resume_ckpt['args'] # Restore args from the initial run

    torch.manual_seed(a.seed)
    random.seed(a.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    full_train_ds = torchvision.datasets.CIFAR10(a.data, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(a.data, train=False, download=True, transform=test_tf)
    
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    # Validation dataset should not have random crop/flip, but we must use a hack to apply test_tf
    full_train_ds_val = torchvision.datasets.CIFAR10(a.data, train=True, download=False, transform=val_tf)

    # 45k / 5k split
    g = torch.Generator().manual_seed(a.seed)
    indices = torch.randperm(len(full_train_ds), generator=g).tolist()
    train_indices = indices[:45000]
    val_indices = indices[45000:]
    
    train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_train_ds_val, val_indices)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    if a.finetune_from:
        model = torch.load(a.finetune_from, map_location='cpu', weights_only=False)
    else:
        model = build_model(a.arch)
    
    model = model.to(device)

    # For fine-tuning, smaller LR is usually better, but we let CosineAnnealing handle it
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()

    start_epoch, log, wall_clock_before = 0, [], 0.0
    best_val_acc = 0.0
    best_state_dict = None

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model_state'])
        opt.load_state_dict(resume_ckpt['optimizer_state'])
        sched.load_state_dict(resume_ckpt['scheduler_state'])
        start_epoch, log = resume_ckpt['epoch'], resume_ckpt['log']
        wall_clock_before = resume_ckpt['wall_clock_s_so_far']
        best_val_acc = resume_ckpt.get('best_val_acc', 0.0)
        best_state_dict = resume_ckpt.get('best_state_dict', None)
        print(f'Resuming {a.arch} (seed {a.seed}) from epoch {start_epoch+1}/{a.epochs}', flush=True)

    t0 = time.time() - wall_clock_before
    for epoch in range(start_epoch, a.epochs):
        model.train()
        train_loss, train_ok, train_n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
            train_ok += (out.argmax(1) == y).sum().item()
            train_n += x.size(0)
        sched.step()

        model.eval()
        val_loss, val_ok, val_n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += crit(out, y).item() * x.size(0)
                val_ok += (out.argmax(1) == y).sum().item()
                val_n += x.size(0)

        val_acc = val_ok / val_n
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            import copy
            best_state_dict = copy.deepcopy(model.state_dict())

        e_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss / train_n,
            'train_acc': train_ok / train_n,
            'val_loss': val_loss / val_n,
            'val_acc': val_acc
        }
        log.append(e_log)
        print(f"Epoch {epoch+1:03d} | Train Acc: {e_log['train_acc']:.4f} | Val Acc: {e_log['val_acc']:.4f}", flush=True)

        # Power loss resilience check
        if power_state.stop_requested():
            print(f'\n[train_p8] STOP_MARKER detected at end of epoch {epoch+1}. Saving resume checkpoint...', flush=True)
            ipath = interrupted_path(a.out)
            torch.save({
                'args': a,
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': opt.state_dict(),
                'scheduler_state': sched.state_dict(),
                'log': log,
                'best_val_acc': best_val_acc,
                'best_state_dict': best_state_dict,
                'wall_clock_s_so_far': time.time() - t0 - wall_clock_before
            }, ipath)
            power_state.write_resume_hint(f'python3 train_p8.py --resume {ipath}')
            import sys
            sys.exit(99) # Exit code 99 means interrupted

    # Finished training, test the best model
    model.load_state_dict(best_state_dict)
    model.eval()
    test_ok, test_n = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_ok += (out.argmax(1) == y).sum().item()
            test_n += x.size(0)
    
    test_acc = test_ok / test_n
    print(f"\nFinished {a.arch} | Best Val: {best_val_acc:.4f} | Test: {test_acc:.4f}")

    # Save final model
    out_path = Path(a.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if a.finetune_from:
        torch.save(model, out_path) # save full module since it's pruned
    else:
        torch.save(model.state_dict(), out_path)
    
    # Save log
    with open(out_path.with_suffix('.json'), 'w') as f:
        json.dump({'test_acc': test_acc, 'best_val_acc': best_val_acc, 'log': log}, f, indent=2)

    power_state.clear_resume_hint()
    interrupted_path(a.out).unlink(missing_ok=True)

if __name__ == '__main__':
    main()
