import torch
import config

class EntityDataset:
    def __init__(self,texts,tags):
        # texts: [["hi",",","I","am","learning"],["hello"," ","nice","to","meet","you"]]
        # pos/tags: [[1 2 3 4 5],[2 3 4 5 6]]
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)
    def __getitem__(self,item):
        text = self.texts[item]
        tag = self.tags[item]
        padding_len = config.MAX_LEN - len(text)
        text = text + [0]*padding_len
        tag = tag + [0]*padding_len

        return {
            "sentence":torch.tensor(text, dtype=torch.long),
            "tags":torch.tensor(tag, dtype=torch.long),
        }
    