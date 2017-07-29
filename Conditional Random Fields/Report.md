# Classification errors on different hyperparameters choices
| Learning rate | Decrease const           | L2   | L1   | Train error | Stderr  | Valid error | Stderr  | Test error | Stderr |
|---------------|--------------------------|------|------|-------------|---------|-------------|---------|------------|--------|
|      0.1      |             0            |   0  |   0  |    0.0775   |  0.0013 |    0.0786   |  0.0036 |   0.07401  | 0.0036 |
| 1e-2          | 1e-1                     | 1e-3 | 1e-3 | 0.1634      | 0.0018  | 0.1599      | 0.0050  | 0.1503     | 0.0050 |
| 1e-1          | 1e-2                     | 1e-3 | 1e-3 | 0.280       | 0.00219 | 0.273       | 0.00612 | 0.2768     | 0.0062 |
| 1e-1          | 1e-8(after each example) | 1e-3 | 1e-3 | 0.128       | 0.00164 | 0.1316      | 0.00464 | 0.1190     | 0.0045 |
| 1e-1          | 1e-7(after each example) | 1e-3 | 1e-3 | 0.1289      | 0.001   | 0.1316      | 0.00464 | 0.1190     | 0.0045 |
| 1e-1          | 1e-1                     | 1e-3 | 1e-3 | 0.1706      | 0.00184 | 0.1667      | 0.00512 | 0.1746     | 0.0053 |
| 1e-1          | 1e-2                     | 1e-3 | 1e-3 | 0.2807      | 0.0021  | 0.2731      | 0.00612 | 0.2768     | 0.0062 |
| 5e-1          | 5e-2                     | 1e-3 | 1e-3 | 0.1722      | 0.0018  | 0.16487     | 0.00509 | 0.1774     | 0.0053 |
| 5e-2          | 1e-3                     | 1e-3 | 1e-3 | 0.4828      | 0.0024  | 0.4667      | 0.00685 | 0.4885     | 0.0069 |
| 5e-2          | 1e-4                     | 1e-3 | 1e-3 | 0.5232      | 0.00244 | 0.51141     | 0.00686 | 0.5151     | 0.0069 |
**| 5e-2          | 1e-5                     | 1e-3 | 1e-3 | 0.5232      | 0.00244 | 0.51141     | 0.00686 | 0.5151     | 0.0069 |**
| 5e-2          | 1e-6                     | 1e-3 | 1e-3 | 0.5232      | 0.00244 | 0.51141     | 0.00686 | 0.51512    | 0.0070 |

I test through a few combinations of hyperparameters, most of them do not converge.
Real progress only show when **decrease constant** is taken into account. After every epoch, the **learning rate** is multiplied with the **decrease constant**.
The training time  is long but the result is still not good.

### Update
The training converged, it converged after the first epoch, even though the rate is still not very high, but increased to 0.636



![Trainset accuracy](images/train_accu.png?raw=true)
![Trainset NLL](images/train_nll.png?raw=true)

![Validset accuracy](images/valid_accu.png?raw=true)
![Validset NLL](images/valid_nll.png?raw=true)
