
from tqdm import tqdm
from torch.utils.data import DataLoader

from pnpl.datasets import LibriBrainSpeech

def main():

    train_dataset = LibriBrainSpeech("./datasets", partition="train", tmin=0.0, tmax=1.0)
    val_dataset = LibriBrainSpeech("./datasets", partition="validation", tmin=0.0, tmax=1.0)

    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=6, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=6, pin_memory=True, shuffle=False)

    for batch in tqdm(train_dataloader):

        print(batch[1])
        print(batch[0].shape)



if __name__ == "__main__":
    main()