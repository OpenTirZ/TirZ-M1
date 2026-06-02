import torch
from torch.utils.data import Dataset , DataLoader

class GTPDatasetV1(Dataset) :
    def __init__(self, txt, tokenizer , max_lenght , stride) :
        self.input_ids = []
        self.target_ids = []

        # Pass allowed_special to the tokenizer.encode method
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'}, disallowed_special=())

        for i in range(0,len(token_ids) - max_lenght , stride) :
            input_chunk = token_ids[i:i+max_lenght]
            output_chunk = token_ids[i+1:i+max_lenght+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))

    def __len__(self) :
        return len(self.input_ids)

    def __getitem__(self, idx) :
        return self.input_ids[idx] , self.target_ids[idx]

def create_dataloader(text , batch_size =4,max_length = 256 , stride = 128 ,
                      shuffle = True , drop_last = True , num_workers = 0) :

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GTPDatasetV1(text, tokenizer=tokenizer , max_lenght=max_length , stride=stride)

    dataloader = DataLoader(
        dataset=dataset,

        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader
