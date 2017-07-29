# Classification errors on different hyperparameters choices
| Learning rate | Decrease const | Hidden layers | L2 | L1 | Activation | Train error | Validation error | Test error | Train NLL | Validation NLL | Test NLL |
|---------------|----------------|---------------|----|----|------------|-------------|------------------|------------|-----------|----------------|----------|
|      0.1      |        0       |    [64,32]    |  0 |  0 |   Sigmoid  |    0.7744   |      0.7568      |   0.7652   |   0.7510  |     0.8041     |  0.7788  |
|      0.1      |        0       |    [32,16]    |  0 |  0 |   Sigmoid  |    0.7521   |      0.7376      |    0.747   |   0.8346  |     0.8826     |  0.8553  |
|      0.1      |        0       |     [16,8]    |  0 |  0 |   Sigmoid  |    0.6574   |      0.6451      |   0.6463   |   1.1580  |     1.2135     |  1.1801  |
|      **0.1**      |        **0**       |    **[100,50]**   |  **0**|  **0** |   **Sigmoid**  |    **0.7912**   |      **0.7779**      |    **0.789**   |   **0.692**   |     **0.7409**     |  **0.7162**  |
|      0.1      |        0       |  [100,75,50]  |  0 |  0 |   Sigmoid  |    0.7090   |      0.6978      |   0.7087   |   0.989   |      1.017     |  0.9953  |
|      0.1      |        0       |    [100,50]   |  0 |  0 |    Tanh    |    0.4024   |      0.3986      |   0.4052   |   3.0013  |      3.009     |   2.984  |
|      0.1      |        0       |    [64,32]    |  0 |  0 |    Tanh    |    0.4576   |      0.4583      |   0.4548   |   2.112   |      2.120     |   2.127  |
|      0.01     |        0       |    [100,50]   |  0 |  0 |   Sigmoid  |    0.6477   |      0.6396      |   0.6412   |   1.257   |      1.288     |   1.263  |
|       1       |        0       |    [100,50]   |  0 |  0 |   Sigmoid  |    0.294    |      0.3019      |   0.2934   |   2.4305  |     2.4035     |  2.4494  |
|      0.5      |        0       |    [100,50]   |  0 |  0 |   Sigmoid  |    0.635    |      0.6233      |   0.6309   |   1.2537  |     1.3005     |  1.2842  |

I test through a few combinations of hyperparameters, and see that the combination with *learning rate* of 0.1 and *hidden layers size* is [100, 50] yields the best result.

![Trainset accuracy and NLL](images/train.png?raw=true)


![Validset accuracy and NLL](images/valid.png?raw=true)
