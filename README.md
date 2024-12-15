# **Turbulent Flow PINN Framework**

### **Simulation Workflow and Data Processing**  

**_A personal project by Bob Bu_**  

---

## **1. Load Environment**

Load the OpenFOAM environment:  
```bash
source /opt/openfoam2406/etc/bashrc
```

> **Note**: This is only an example. The actual setup may vary depending on your OpenFOAM version and installation method.  
> For detailed instructions, refer to the [OpenFOAM Official Documentation](https://www.openfoam.com/documentation).  

---

### **Recommended Approach**  
Using **Docker** is highly recommended for consistent and portable OpenFOAM environments:  
1. Install Docker: Follow the [Docker Installation Guide](https://docs.docker.com/get-docker/).  
2. Pull the official OpenFOAM Docker image:  
   ```bash
   docker pull openfoam/openfoam2406
   ```
3. Run OpenFOAM in a container:  
   ```bash
   docker run -it --rm openfoam/openfoam2406 /bin/bash
   ```

---

## **2. Mesh Generation**
Generate the mesh for the simulation:
```bash
blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
```

---

## **3. Parallel Decomposition**
Decompose the domain for parallel computation:
```bash
decomposePar
```

---

## **4. Run `pisoFoam` in Parallel**
Execute the simulation using 8 cores:
```bash
mpirun --allow-run-as-root --np 8 pisoFoam -parallel
```

---

## **5. Merge Data**
After the simulation is complete, reconstruct the decomposed data:
```bash
reconstructPar
```
The processed simulation results can be found in:  
`Boeing737max/postProcessing/ensightWrite/data`.

To save disk space, delete the following:
- **Time step folders** in the root directory (e.g., `0.017`, `0.033`).
- **Processor results** generated for parallel runs (e.g., `processor0`, `processor1`, ...).

---

## **6. Visualization with ParaView**
### Steps to Load and Process the Simulation Data:
1. **Open ParaView** and load the `.case` file:
   ```
   ~/Boeing737max/postProcessing/ensightWrite/_.case
   ```

2. **Enable the Visualization**:  
   In the **Pipeline Browser**, click the **eye icon** next to `_.case` to display the simulation results.

3. **Select Cell Data**:
   - In the **Properties** panel, under **Cell Array**, select all four data fields (`U`, `k`, `omega`, and `p`).
   - Click **Apply**.

4. **Calculator Operation**:
   - In the **Pipeline Browser**, select `_.case`.
   - Use the search bar under **Help** to search for "Calculator".
   - A new `Calculator1` layer will appear in the Pipeline Browser. Enable it by clicking the **eye icon**.
   - In the **Properties** panel for `Calculator1`:
     - Click the **gear icon** to reveal advanced options.
     - Set **Attribute Type** to "Point Data".
     - Check **Coordinate Results** and **Result Normals**.
     - Set the **Result Array Name** to `Coordinates`.
     - Enter `coords` as the expression, then click **Apply**.

5. **Convert Point Data to Cell Data**:
   - Search for **"Point Data to Cell Data"** under **Help**.
   - Add the converter and enable the `PointDatatoCellData1` layer in the Pipeline Browser.
   - In the **Properties** panel:
     - Uncheck **Process All Arrays**.
     - Ensure "Coordinates" is selected, then click **Apply**.

6. **Export the Data**:
   - Make all layers visible in the **Pipeline Browser**.
   - Select **File -> Save Data…** and choose a suitable location with the `.csv` format.
   - In the **Configure Writer** dialog, configure the following settings:
     - Check **Write Time Steps**, **Write Time Steps Separately**, and **Choose Arrays to Write**.
     - Ensure all five datasets (`Coordinates`, `U`, `k`, `omega`, `p`) are selected.
     - Set **Precision** to `5`.
     - Set **Field Association** to "Cell Data".
     - Check **Add Meta Data**, **Add Time Step**, **Add Time**, and **Use String Delimiter**.

   After applying these settings, ParaView will generate the training dataset in `.csv` format.

### **Common Issue**: Missing or Invisible `Coordinates` Data  

If the **`Coordinates`** entry is missing or not visible:  
1. Go to the **Pipeline Browser** and select the `PointDatatoCellData1` layer.  
2. In the **Properties** panel:  
- Check and immediately uncheck **Pass Point Data**.  
- Click **Apply** to refresh the layer.  

This refreshes the data pipeline and will make the `Coordinates` entry visible for export.  

---

## **7. Expected Training Dataset Format**
The exported `.csv` file will follow this structure:
```
"TimeStep","Time","Cell Type","Coordinates:0","Coordinates:1","Coordinates:2","U:0","U:1","U:2","k","omega","p"
```
**Example Data Row**:
```
0,0.017,12,253.12,56.25,15.625,-74.897,-0.51436,0.063483,0.20531,1.3843,146.9
```

> **Note**: Processing speed depends on your hardware. A full 5-second simulation generates approximately **50GB** of data, containing around **800 million rows**.  
If a smaller dataset is required, adjust the following in the OpenFOAM configuration file `Boeing737max/system/controlDict`:
- **Time step size**
- **Number of time steps**
- **Total simulation time**
- **Data save frequency**

Rerun the workflow from the `decomposePar` step onward after making these changes.

---

## **8. Model Training**
Run the `cfd.py` script to train the model. This script performs the following steps:
1. Converts `.csv` data to **HDF5** binary format for optimized storage and faster loading.
2. Uses **Optuna** to search for the best hyperparameters on a smaller training dataset.
3. Applies the selected hyperparameters to train the **Physics-Informed Neural Network (PINN)** on a larger dataset.

### **Training Notes**:
- Data is read in a **forward direction** for training and in a **reverse direction** for validation.
- Parameters like batch size and data fraction can be customized in `cfd.py`.
- Full model training on an **NVIDIA RTX GPU** can take several days to months, depending on the batch size and dataset size. For faster convergence, use a smaller dataset (e.g., 1/80th of the full data) and moderate batch sizes (e.g., 4096).

---

## **9. Model Testing and Visualization**
To test the trained model, run the `TeslaValveFlowSimulator.py` script.  
This script:
- Passes **coordinate** and **time** information into the trained neural network.
- Outputs predictions for **airflow dynamics**.
- Visualizes the airflow process using a graphical interface built with **PyQt** and **PyVista**.

The visualization includes both **forward** and **reverse** flow modes, demonstrating the Tesla valve's physical behavior:  
> *"The interior of the conduit is provided with enlargements, recesses, projections, baffles, or buckets which, while offering virtually no resistance to the passage of the fluid in one direction, other than surface friction, constitute an almost impassable barrier to its flow in the opposite direction."*

---

## **Credits**

- **Data Collected From**: OpenFOAM Simulations  
- **Visualization Software**: ParaView, PyQt and PyVista  
- **Simulation Environment**: Docker Containers  
- **3D Model Sources**: GrabCAD and Sketchfab Contributors  
- **Boeing 737 MAX 8**: © *The Boeing Company*  
- **Simulation Data Collected On**: macOS  
- **Trained On**: NVIDIA RTX GPU  



---

**Department of Computing Science**  
*University of Alberta*  
**December 2024, Edmonton, AB, Canada**

---
