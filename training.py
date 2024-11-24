import argparse
import torch
import torch.nn as nn
from dataset import TrainDataset, EvalDataset
from models import SRCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities import calculate_psnr

def parse_args():
    parser = argparse.ArgumentParser(description="Train SRCNN for Super-Resolution")
    parser.add_argument('--train-images', type=str, default='train_images.hdf5', help='Path to training dataset (HDF5 file)')
    parser.add_argument('--eval-images', type=str, default='eval_images.hdf5', help='Path to evaluation dataset (HDF5 file)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--eval-batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, choices=['cpu', 'mps'], default='mps' if torch.mps.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--output-model', type=str, default='best_model.pth', help='File to save the best model')
    parser.add_argument('--output-loss', type=str, default='eval_loss.txt', help='File to save evaluation PSNR values')
    return parser.parse_args()

def main():
    args = parse_args()

    train_dataset = TrainDataset(args.train_images)
    train_data = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    eval_dataset = EvalDataset(args.eval_images)
    eval_data = DataLoader(dataset=eval_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = SRCNN(channels=3).to(args.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': model.extraction.parameters()},
        {'params': model.mapping.parameters()},
        {'params': model.reconstruction.parameters(), "lr": 1e-5},
    ], lr=args.lr)

    epochs = args.epochs
    eval_psnr = []
    best_psnr = -float('inf')

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        with tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch {epoch+1}/{epochs} - Training") as train_bar:
            for i, (X, y) in train_bar:
                X, y = X.to(args.device), y.to(args.device)

                preds = model(X)
                loss = criterion(preds, y)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_data)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.10f}")

        eval_loss = 0.0
        epoch_psnr = 0.0
        model.eval()

        with torch.no_grad():
            with tqdm(enumerate(eval_data), total=len(eval_data), desc=f"Epoch {epoch+1}/{epochs} - Evaluation") as eval_bar:
                for i, (X, y) in eval_bar:
                    X, y = X.to(args.device), y.to(args.device)

                    preds = model(X)
                    loss = criterion(preds, y)
                    eval_loss += loss.item()

                    epoch_psnr += calculate_psnr(preds.cpu(), y.cpu())

                    eval_bar.set_postfix(loss=loss.item())

        epoch_psnr /= len(eval_data)
        eval_psnr.append(epoch_psnr)

        eval_loss /= len(eval_data)
        print(f"Epoch {epoch+1}/{epochs} - Evaluation Loss: {eval_loss:.10f}, PSNR: {epoch_psnr:.4f} dB")

        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            torch.save(model.state_dict(), args.output_model)
            print(f"New best PSNR: {best_psnr:.4f} dB - Model saved to {args.output_model}!")

    print(f"Best PSNR achieved: {best_psnr:.4f} dB")

    with open(args.output_loss, "w") as file:
        for psnr in eval_psnr:
            file.write(f"{psnr}\n")
    print(f"Evaluation PSNR values saved to {args.output_loss}")

if __name__ == '__main__':
    main()
