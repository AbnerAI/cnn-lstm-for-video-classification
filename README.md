# CNN + LSTM for Video Classification
### V 0.2

> 1. view-> permute
> 2. Solve cross-border issues(def __len__(self): return len(self.data_lis)//self.seq_len)
> 3. There was a problem with the crop code before, now the data has been re-crop.
> 4. Find the reason of no drop in accuracy, Mainly because of data imbalance.
> 5. Fixed bug: losses.update(loss.item(), data.size(1)) & accuracies.update(acc, data.size(1)) # data.size(1) is batch-size, not data.size(0)
> 6. Add validation set code.

Note that: V0.2 used the thyroid data set of the medical examination department, and I **manually duplicated** the **C** classification to ensure that the data is balanced.

### V 0.1 

> 1. Implement seq_len in dataloader module of pytorch.
> 2. Implement classification prediction for consecutive  video frames.

## Environments

GPU: 2080 Ti 

Cuda: 11.0 

The more specific environment in **environments.yaml** 

## License

This project is licensed under the MIT License 

