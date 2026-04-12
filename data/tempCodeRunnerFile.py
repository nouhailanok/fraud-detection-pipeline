    path = "data/node_1/tensors"
    tr_loader, te_loader = get_split_dataloaders(path, train_ratio=0.8)
    print(f"Batches Train : {len(tr_loader)} | Batches Test : {len(te_loader)}")