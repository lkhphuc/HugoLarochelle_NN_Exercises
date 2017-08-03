# Classification errors on different hyperparameters choices
| Learning rate | DC | Size         | Pretrain LR | Pretrain n_epochs | Pretrain CD-K | Train1  | Valid1 | Test1  | Train2          | Valid2         | Test2          |
|---------------|----|--------------|-------------|-------------------|---------------|---------|--------|--------|-----------------|----------------|----------------|
**| 0.01          | 0  | [200,100]    | 0.01        | 10                | 1             | 0.02929 | 0.1237 | 0.1207 | 0.120777241229  | 0.445547235093 | 0.440961553369 |**
| 0.1           | 0  | [200,100]    | 0.01        | 10                | 1             | 0.01001 | 0.1334 | 0.1289 | 0.0304282351538 | 0.74049225121  | 0.681836233563 |
| 0.1           | 0  | [200,100]    | 0.01        | 1                 | 1             | 0.07532 | 0.1403 | 0.1392 | 0.219293034641  | 0.52120577611  | 0.49653205696  |
| 0.1           | 0  | [200,100]    | 0.1         | 10                | 1             | 0.09433 | 0.1788 | 0.1724 | 0.304933101967  | 0.856697783771 | 0.82479175184  |
| 0.1           | 0  | [200,100]    | 0.01        | 1                 | 10            | 0.02883 | 0.131  | 0.1298 | 0.0792054784129 | 0.613738153689 | 0.604029163899 |
| 0.001         | 0  | [200,100]    | 0.01        | 10                | 1             | 0.09890 | 0.1362 | 0.1331 | 0.337138923697  | 0.470323721249 | 0.454587078984 |
| 0.1           | 0  | [200,100,50] | 0.01        | 1                 | 1             | 0.10021 | 0.1456 | 0.1394 | 0.312640120379  | 0.504966159424 | 0.492347980635 |

The Train/Valid/Test**1** is the classification error (the percentage of wrong classification).
The Train/Valid/Test**2** is the Negative Log-Likelihood.


## Train set progression of classification error and Negative Log-likelihood
![Trainset accuracy](images/train_error.png?raw=true)
![Trainset NLL](images/train_NLL.png?raw=true)

## Validation set progression of classification error and Negative Log-likelihood
![Validset accuracy](images/valid_error.png?raw=true)
![Validset NLL](images/valid_NLL.png?raw=true)

## Test set classification error 95% confidence interval:
error +/- Z(0.95) * sqrt( (error * (1 - error)) / n)
= 0.1207 +/- 1.96 * sqrt( (0.1207 * (1 - 0.1207)) / 10000)
=0.1207 +/- 0.0032577
0.1207 +/- 0.3257%
