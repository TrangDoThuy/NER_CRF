from tqdm import tqdm

def train_fn(data_loader,model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader,total=len(data_loader)):

        no = 0

        for k,v in data.items():
            data[k] = v.to(device)
            no =  v.shape[0]
        optimizer.zero_grad()

        for i in range(no):
            loss = model.neg_log_likelihood(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss/len(data_loader)

def eval_fn(data_loader,model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader,total=len(data_loader)):
        for k,v in data.items():
            data[k] = v.to(device)
        loss = model.neg_log_likelihood(**data)
        final_loss += loss.item()
    return final_loss/len(data_loader)
