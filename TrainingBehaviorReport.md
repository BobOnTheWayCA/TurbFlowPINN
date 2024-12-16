# **Turbulent Flow PINN Framework**

### **Analysis of Training Behavior**  

**_A personal project by Bob Bu_**  

---



### Training Behavior with 1M and 160M Datasets: Code Traces


#### 1. Training Trace for the 1M Dataset

```plaintext
Epoch 1/10, LR: 1e-05, PDE Weight: 0.0001
Epoch 1/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [03:16<00:00, 12.41batch/s, loss=0.563]
Epoch 1/10, Train Loss: 0.5629, Val Loss: 1.3389
Model saved at epoch 1
New best model saved with Val Loss: 1.3389
Epoch 2/10, LR: 1e-05, PDE Weight: 0.0010900000000000003
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [03:36<00:00, 11.27batch/s, loss=0.517]
Epoch 2/10, Train Loss: 0.5172, Val Loss: 1.0639
Model saved at epoch 2
New best model saved with Val Loss: 1.0639
Epoch 3/10, LR: 1e-05, PDE Weight: 0.0020800000000000003
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [04:01<00:00, 10.13batch/s, loss=0.516]
Epoch 3/10, Train Loss: 0.5156, Val Loss: 1.0359
Model saved at epoch 3
New best model saved with Val Loss: 1.0359
Epoch 4/10, LR: 1e-05, PDE Weight: 0.00307
Epoch 4/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [04:19<00:00,  9.39batch/s, loss=0.519]
Epoch 4/10, Train Loss: 0.5195, Val Loss: 1.0528
Model saved at epoch 4
Epoch 5/10, LR: 1e-05, PDE Weight: 0.004060000000000001
Epoch 5/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [04:01<00:00, 10.11batch/s, loss=0.517]
Epoch 5/10, Train Loss: 0.5171, Val Loss: 1.0732
Model saved at epoch 5
Epoch 6/10, LR: 5e-06, PDE Weight: 0.005050000000000001
Epoch 6/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [04:17<00:00,  9.48batch/s, loss=0.515]
Epoch 6/10, Train Loss: 0.5153, Val Loss: 1.0763
Model saved at epoch 6
Epoch 7/10, LR: 5e-06, PDE Weight: 0.00604
Epoch 7/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [03:33<00:00, 11.44batch/s, loss=0.518]
Epoch 7/10, Train Loss: 0.5181, Val Loss: 1.0849
Model saved at epoch 7
Epoch 8/10, LR: 2.5e-06, PDE Weight: 0.007030000000000001
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [03:34<00:00, 11.41batch/s, loss=0.522]
Epoch 8/10, Train Loss: 0.5223, Val Loss: 1.0865
Model saved at epoch 8
Epoch 9/10, LR: 2.5e-06, PDE Weight: 0.008020000000000001
Epoch 9/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [04:20<00:00,  9.39batch/s, loss=0.517]
Epoch 9/10, Train Loss: 0.5168, Val Loss: 1.0884
Model saved at epoch 9
Epoch 10/10, LR: 1.25e-06, PDE Weight: 0.00901
Epoch 10/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 2442/2442 [03:30<00:00, 11.59batch/s, loss=0.519]
Epoch 10/10, Train Loss: 0.5191, Val Loss: 1.0895
Model saved at epoch 10
```

#### 2. Training Trace for the 160M Dataset

```plaintext
Epoch 1/10, LR: 1e-05, PDE Weight: 0.0001
Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:25<00:00, 12.91batch/s, loss=6.57]
Epoch 1/10, Train Loss: 6.5680, Val Loss: 1.3391
Model saved at epoch 1
New best model saved with Val Loss: 1.3391
Epoch 2/10, LR: 1e-05, PDE Weight: 0.0010900000000000003
Epoch 2/10: 100%|███████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:26<00:00, 12.91batch/s, loss=1.47e+3]
Epoch 2/10, Train Loss: 1472.4204, Val Loss: 1.4431
Model saved at epoch 2
Epoch 3/10, LR: 1e-05, PDE Weight: 0.0020800000000000003
Epoch 3/10: 100%|███████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:07<00:00, 12.99batch/s, loss=1.12e+3]
Epoch 3/10, Train Loss: 1124.4971, Val Loss: 1.4385
Model saved at epoch 3
Epoch 4/10, LR: 5e-06, PDE Weight: 0.00307
Epoch 4/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:06<00:00, 12.99batch/s, loss=88.8]
Epoch 4/10, Train Loss: 88.7782, Val Loss: 1.6401
Model saved at epoch 4
Epoch 5/10, LR: 5e-06, PDE Weight: 0.004060000000000001
Epoch 5/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:08<00:00, 12.99batch/s, loss=73.8]
Epoch 5/10, Train Loss: 73.8404, Val Loss: 1.8376
Model saved at epoch 5
Epoch 6/10, LR: 2.5e-06, PDE Weight: 0.005050000000000001
Epoch 6/10: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:11<00:00, 12.97batch/s, loss=31]
Epoch 6/10, Train Loss: 30.9868, Val Loss: 1.7640
Model saved at epoch 6
Epoch 7/10, LR: 2.5e-06, PDE Weight: 0.00604
Epoch 7/10: 100%|████████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:07<00:00, 12.99batch/s, loss=3.4e+3]
Epoch 7/10, Train Loss: 3399.5942, Val Loss: 1.8292
Model saved at epoch 7
Epoch 8/10, LR: 1.25e-06, PDE Weight: 0.007030000000000001
Epoch 8/10: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:19<00:00, 12.94batch/s, loss=3.9]
Epoch 8/10, Train Loss: 3.8981, Val Loss: 1.6957
Model saved at epoch 8
Epoch 9/10, LR: 1.25e-06, PDE Weight: 0.008020000000000001
Epoch 9/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:13<00:00, 12.96batch/s, loss=21.3]
Epoch 9/10, Train Loss: 21.3386, Val Loss: 1.6285
Model saved at epoch 9
Epoch 10/10, LR: 6.25e-07, PDE Weight: 0.00901
Epoch 10/10: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 39063/39063 [50:36<00:00, 12.87batch/s, loss=8.17]
Epoch 10/10, Train Loss: 8.1739, Val Loss: 1.6201
Model saved at epoch 10
```

## Analysis of Training Behavior in PINNs for Tesla Valve Turbulence Modeling

### Introduction

Training a physics-informed neural network (PINN) with k-omega SST-IDDES turbulence data reveals significant differences between using 1M and 160M datasets. The smaller dataset, representing 1/16th of a second, achieves stable training and validation losses. In contrast, the larger dataset, spanning a full second, shows unstable training loss despite maintaining consistent validation performance.

---

### Observations and Analysis

The 1M dataset offers a simpler learning task. It reflects localized turbulence patterns over a short timeframe, which the PINN’s architecture—8 layers of 512 units with LeakyReLU activation—handles effectively. Training loss converges to approximately 0.52, while validation loss stabilizes around 1.08. These results show the model can capture key turbulence features under relatively uniform conditions.

The 160M dataset introduces significant complexity. It spans a broader range of turbulence states, including transitional and fully developed flows. The same model struggles to represent this diversity, leading to fluctuations in training loss. Despite this, the validation loss remains around 1.62, indicating the model successfully generalizes to simpler or more representative conditions in the validation set.

Numerical challenges also contribute. The physics loss function, which enforces conservation laws through higher-order derivatives, is sensitive to errors that amplify over long training runs. Increasing the weight of the physics loss during training exacerbates this issue, especially with the larger dataset’s varied flow regimes.

The dataset itself compounds the difficulty. The k-omega SST-IDDES model provides a balance between fidelity and computational efficiency, but it introduces approximations that may conflict with the PINN’s optimization process. The 1M subset, representing less variability, allows the model to adapt more easily, while the full 160M dataset’s richness in flow dynamics highlights the limitations of the fixed architecture.

---

### Implications for Tesla Valve Simulation and Visualization

The 1M-trained model (`LeakyReLU_1M.pth`) captures localized turbulence features effectively and serves as a baseline for simpler simulations. It is well-suited for visualizing Tesla valve flow in scenarios where precision in fine-scale dynamics is less critical.

The 160M-trained models provide a broader perspective. The best-performing model (`LeakyReLU_160M_Best_Epoch.pth`) captures dominant patterns with high fidelity, making it ideal for more comprehensive analysis. Meanwhile, the 10th-epoch model (`LeakyReLU_160M_10th_Epoch.pth`) highlights the challenges of optimizing over complex datasets and serves as a reference for training progress.

To observe different model performances, you can modify the Python script `TeslaValveFlowSimulator.py`.

---

The instability observed when training on 160M samples arises from the fixed model architecture, numerical sensitivity of the physics loss, and the dataset’s inherent complexity. While the smaller dataset yields stable training outcomes, the larger dataset provides a more detailed and varied learning challenge. The availability of models trained on both datasets offers tools for exploring Tesla valve flow in localized and global contexts, enabling practical visualization and deeper turbulence analysis.

---

**Department of Computing Science**  
*University of Alberta*  
**December 2024, Edmonton, AB, Canada**