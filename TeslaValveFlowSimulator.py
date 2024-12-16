import numpy as np
import torch
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QSlider, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from scipy.spatial import cKDTree

# Define the model
class PINNModel(torch.nn.Module):
    def __init__(self, input_dim=4, output_dim=6, hidden_layers=8, hidden_units=512):
        super(PINNModel, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_units))
        layers.append(torch.nn.Tanh())
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(hidden_units, hidden_units))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(hidden_units, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Load the model
def load_model(model_path):
    model = PINNModel(input_dim=4, output_dim=6)
    # Add map_location=torch.device('cpu') for none cuda devices
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Trained PINN model loaded successfully.")
    return model

# Use the model to predict
def predict_flow(model, grid_points, batch_size=2000):
    total_points = len(grid_points)
    outputs = []
    try:
        for i in range(0, total_points, batch_size):
            batch = grid_points[i:i+batch_size]
            inputs = torch.tensor(batch, dtype=torch.float32)
            batch_output = model(inputs).detach().numpy()
            outputs.append(batch_output)
            # Print min and max pressure in the current batch
            # print(f"Batch {i // batch_size}: Pressure range: "
                  # f"{batch_output[:, 5].min():.2f} to {batch_output[:, 5].max():.2f}")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None
    return np.concatenate(outputs, axis=0)

class WindTunnelApp(QWidget):
    def __init__(self, model_path):
        super().__init__()
        self.model = load_model(model_path)
        self.plotter = BackgroundPlotter()

        # Parameter settings
        self.x_range = (-10, 10)
        self.y_range = (-3, 3)
        self.z_range = (-3, 3)
        self.resolution = 10
        self.t = 0
        self.time_step = 0.1
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_time_step)

        self.flow_direction = "forward"  # Initial flow direction is forward

        self.init_ui()
        self.initialize_static_scene()

    def init_ui(self):
        layout = QVBoxLayout()

        # Add title
        title_label = QLabel("Tesla Valve Flow Dynamics Simulator")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)

        # Button area
        button_layout = QHBoxLayout()
        self.run_button = QPushButton('Initialize')
        self.run_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_button)

        self.play_button = QPushButton('Start')
        self.play_button.clicked.connect(self.play_animation)
        button_layout.addWidget(self.play_button)

        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_animation)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        # Mode switch and reset buttons
        mode_layout = QHBoxLayout()
        self.direction_button = QPushButton('Switch to Forward Flow')
        self.direction_button.clicked.connect(self.switch_flow_direction)
        mode_layout.addWidget(self.direction_button)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_simulation)
        mode_layout.addWidget(self.reset_button)

        layout.addLayout(mode_layout)

        # Time step slider
        slider_layout = QHBoxLayout()
        self.slider_label = QLabel('Time Step: 1.0')
        slider_layout.addWidget(self.slider_label)

        self.time_step_slider = QSlider(Qt.Horizontal)
        self.time_step_slider.setMinimum(1)
        self.time_step_slider.setMaximum(100)
        self.time_step_slider.setValue(10)
        self.time_step_slider.valueChanged.connect(self.update_time_step)
        slider_layout.addWidget(self.time_step_slider)

        layout.addLayout(slider_layout)
        # Add sliders for controlling clim
        clim_layout = QVBoxLayout()

        # Reverse flow clim range
        self.positive_clim_label = QLabel('Reverse Flow CLim: [-5.00, 5.00]')
        clim_layout.addWidget(self.positive_clim_label)

        self.positive_clim_slider_min = QSlider(Qt.Horizontal)
        self.positive_clim_slider_min.setMinimum(-500)
        self.positive_clim_slider_min.setMaximum(500)
        self.positive_clim_slider_min.setValue(-500)
        self.positive_clim_slider_min.valueChanged.connect(self.update_clim)
        clim_layout.addWidget(self.positive_clim_slider_min)

        self.positive_clim_slider_max = QSlider(Qt.Horizontal)
        self.positive_clim_slider_max.setMinimum(-500)
        self.positive_clim_slider_max.setMaximum(500)
        self.positive_clim_slider_max.setValue(500)
        self.positive_clim_slider_max.valueChanged.connect(self.update_clim)
        clim_layout.addWidget(self.positive_clim_slider_max)

        # Positive flow clim range
        self.reverse_clim_label = QLabel('Positive Flow CLim: [-5.00, 5.00]')
        clim_layout.addWidget(self.reverse_clim_label)

        self.reverse_clim_slider_min = QSlider(Qt.Horizontal)
        self.reverse_clim_slider_min.setMinimum(-500)
        self.reverse_clim_slider_min.setMaximum(500)
        self.reverse_clim_slider_min.setValue(-500)
        self.reverse_clim_slider_min.valueChanged.connect(self.update_clim)
        clim_layout.addWidget(self.reverse_clim_slider_min)

        self.reverse_clim_slider_max = QSlider(Qt.Horizontal)
        self.reverse_clim_slider_max.setMinimum(-500)
        self.reverse_clim_slider_max.setMaximum(500)
        self.reverse_clim_slider_max.setValue(500)
        self.reverse_clim_slider_max.valueChanged.connect(self.update_clim)
        clim_layout.addWidget(self.reverse_clim_slider_max)

        layout.addLayout(clim_layout)

        self.setLayout(layout)
        self.setWindowTitle("Visualization Controller")

    def update_clim(self):
        """
        Update the clim ranges based on the slider values and update the visualization.
        """
        # Get slider values
        positive_min = self.positive_clim_slider_min.value() / 100
        positive_max = self.positive_clim_slider_max.value() / 100
        reverse_min = self.reverse_clim_slider_min.value() / 100
        reverse_max = self.reverse_clim_slider_max.value() / 100

        # Update labels
        self.positive_clim_label.setText(f'Positive Flow CLim: [{positive_min:.2f}, {positive_max:.2f}]')
        self.reverse_clim_label.setText(f'Reverse Flow CLim: [{reverse_min:.2f}, {reverse_max:.2f}]')

        # Update clim dynamically
        if self.flow_direction == "forward":
            self.plotter.update_scalar_bar_range([positive_min, positive_max])
        else:
            self.plotter.update_scalar_bar_range([reverse_min, reverse_max])

    def initialize_static_scene(self):
        """
        Initialize the static scene, including loading the Tesla Valve model and setting the lighting.
        """
        self.plotter.set_background("white")  # Set background color

        try:
            self.tesla_valve = pv.read("CFD_3DModel/TeslaValve_CFD_3DModel.stl").extract_surface()
            self.plotter.add_mesh(self.tesla_valve, color="gray", opacity=0.8)

            bounds = self.tesla_valve.bounds
            self.x_range = (bounds[0], bounds[1])
            self.y_range = (bounds[2], bounds[3])
            self.z_range = (bounds[4], bounds[5])

        except Exception as e:
            print(f"Error loading Tesla Valve model: {e}")

        light = pv.Light(position=(bounds[1] + 50, bounds[3] + 50, bounds[5] + 50), intensity=1.2)
        self.plotter.add_light(light)

    def switch_flow_direction(self):
        """
        Toggle the flow direction and adjust the Tesla Valve model's orientation.
        """
        if self.flow_direction == "forward":
            self.flow_direction = "reverse"
            self.direction_button.setText("Switch to Reverse Flow")
            self.tesla_valve.points[:, 0] = -self.tesla_valve.points[:, 0]  # Flip the X coordinates
            # self.tesla_valve.points[:, 1] = -self.tesla_valve.points[:, 1]  # Flip the Y coordinates
            # self.tesla_valve.points[:, 2] = -self.tesla_valve.points[:, 2]  # Flip the Z coordinates
            print("Switched to forward flow. Model flipped along X-axis.")
        else:
            self.flow_direction = "forward"
            self.direction_button.setText("Switch to Forward Flow")
            self.tesla_valve.points[:, 0] = -self.tesla_valve.points[:, 0]  # Restore X coordinates
            # self.tesla_valve.points[:, 1] = -self.tesla_valve.points[:, 1]  # Flip the Y coordinates
            # self.tesla_valve.points[:, 2] = -self.tesla_valve.points[:, 2]  # Flip the Z coordinates
            print("Switched to reverse flow. Model restored.")
        self.reset_simulation()

    def run_simulation(self):
        """
        Update the dynamic part (pressure field), mapping the pressure field to the Tesla Valve model.
        """
        # Generate grid points
        grid_points, _ = generate_wind_tunnel(self.x_range, self.y_range, self.z_range, self.t, self.resolution)

        # Predict pressure values using the model
        predictions = predict_flow(self.model, grid_points)
        if predictions is None:
            print("Error: Model failed to generate predictions.")
            return

        # Create PolyData and add pressure data
        grid_data = pv.PolyData(grid_points[:, :3])
        grid_data["pressure"] = predictions[:, 5]

        # Log the overall pressure range
        # print(f"Overall Pressure range: {grid_data['pressure'].min():.2f} to {grid_data['pressure'].max():.2f}")

        # Nearest-neighbor interpolation
        surface_points = self.tesla_valve.points
        tree = cKDTree(grid_data.points)
        _, indices = tree.query(surface_points)
        self.tesla_valve["pressure"] = grid_data["pressure"][indices]

        # Clear previous render
        self.plotter.clear()

        # Determine clim based on flow direction
        if self.flow_direction == "forward":
            clim_range = [self.positive_clim_slider_min.value() / 100, self.positive_clim_slider_max.value() / 100]
        else:
            clim_range = [self.reverse_clim_slider_min.value() / 100, self.reverse_clim_slider_max.value() / 100]

        # Visualize the pressure field on the Tesla Valve
        self.plotter.add_mesh(
            self.tesla_valve,
            scalars="pressure",
            cmap="jet",
            opacity=1.0,
            clim=clim_range,  # Use the dynamically updated clim range
            scalar_bar_args={
                "title": "Pressure",
                "title_font_size": 14,
                "label_font_size": 10,
                "vertical": False,
                "height": 0.025,
                "width": 0.45,
                "position_x": 0.45,
                "position_y": 0.05
            }
        )

    def next_time_step(self):
        self.t += self.time_step
        self.run_simulation()

    def play_animation(self):
        if not self.is_playing:
            self.is_playing = True
            self.timer.start(10)

    def stop_animation(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()

    def reset_simulation(self):
        self.t = 0
        self.run_simulation()

    def update_time_step(self, value):
        self.time_step = value / 10.0
        self.slider_label.setText(f'Time Step: {self.time_step:.2f}')

def generate_wind_tunnel(x_range, y_range, z_range, t, resolution):
    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    z = np.linspace(*z_range, resolution)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    t_array = np.full_like(xx, t)
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel(), t_array.ravel()], axis=-1)
    return points, xx.shape

def print_usage_guide():
    """
    Prints a comprehensive usage guide for the Tesla Valve Flow Dynamics Simulator.
    """
    print("\n================= Tesla Valve Flow Dynamics Simulator User Guide =================\n")
    print("The Tesla Valve Flow Dynamics Simulator, an interactive tool for visualizing "
          "fluid dynamics through a Tesla Valve.\n")

    print("1. Initialize:")
    print("   Sets up the simulation environment, loading the Tesla Valve model and preparing for visualization.")

    print("\n2. Start:")
    print("   Begins the animation of fluid dynamics across the Tesla Valve.")
    print("   The simulation will advance time steps to show pressure variations in real-time.")

    print("\n3. Stop:")
    print("   Pauses the ongoing animation, allowing you to observe the current state of the simulation.")

    print("\n4. Reset:")
    print("   Resets the simulation to its initial state (time step = 0).")
    print("   All previous calculations and visualizations will be cleared.")

    print("\n5. Switch Flow:")
    print("   Toggles the flow direction between 'Reverse' and 'Forward'.")
    print("   In 'Reverse' mode, the fluid flows in the default direction.")
    print("   In 'Forward' mode, the Tesla Valve is mirrored, and the flow is visualized in the opposite direction.")

    print("\n6. Time Step Slider:")
    print("   Adjusts the speed of the simulation by controlling the time step size.")
    print("   Move the slider to values between 0.1 and 10.0 to slow down or speed up the simulation.")

    print("\n7. CLim Sliders:")
    print("   Adjusts the color mapping range (CLim) for the pressure field visualization.")
    print("   Two sets of sliders control the CLim for forward and reverse flow states:")
    print("      Positive Flow CLim: Controls color scaling when in forward flow mode.")
    print("      Reverse Flow CLim: Controls color scaling when in reverse flow mode.")
    print("   Use these sliders to enhance visibility of the pressure gradient.")

    print("\nTips:")
    print("   The visualization window allows full 3D interaction:")
    print("      Zoom: Use the scroll wheel or pinch gesture.")
    print("      Pan: Click and drag using the right mouse button or two-finger swipe.")
    print("      Rotate: Click and drag using the left mouse button or single-finger swipe.")

    print("   The color bar (legend) dynamically adjusts based on the pressure range and flow direction.")
    print("      You can fine-tune the range using the CLim sliders to highlight critical pressure regions.")

    print("   Ensure the desired files are present in the working directory before starting.\n")

    print("The simulator uses a trained Physics-Informed Neural Network (PINN) to predict the fluid dynamics.")
    print("The predictions are efficient but approximate, ideal for real-time visual exploration.")

    print("\nExample Workflow:")
    print("1. Click 'Initialize' to load the Tesla Valve model and prepare the simulation.")
    print("2. Adjust the CLim sliders to set the desired color range.")
    print("3. Use 'Start' to begin the animation and observe real-time pressure changes.")
    print("4. Toggle between 'Forward' and 'Reverse' flow to compare pressure behavior.")
    print("5. Use the Time Step Slider to slow down or speed up the animation as needed.")
    print("6. Pause ('Stop') or reset the simulation to analyze specific time states.")

    print("\n==================================================================================\n")

def main():
    app = QApplication([])
    print("/*------------------------------------------------------------------------------*\\")
    print("|                                                                                |")
    print("|                                                                                |")
    print("|     _______         _     ______ _               _____ _____ _   _ _   _       |")
    print("|    |__   __|       | |   |  ____| |             |  __ \\_   _| \\ | | \\ | |      |")
    print("|       | |_   _ _ __| |__ | |__  | | _____      _| |__) || | |  \\| |  \\| |      |")
    print("|       | | | | | '__| '_ \\|  __| | |/ _ \\ \\ /\\ / /  ___/ | | | . ` | . ` |      |")
    print("|       | | |_| | |  | |_) | |    | | (_) \\ V  V /| |    _| |_| |\\  | |\\  |      |")
    print("|       |_|\\__,_|_|  |_.__/|_|    |_|\\___/ \\_/\\_/ |_|   |_____|_| \\_|_| \\_|      |")
    print("|                                                                                |")
    print("|              Turbulent Flow PINN Training and Validation Framework             |")
    print("|                          A Personal Project by Bob Bu                          |")
    print("|             Department of Computing Science, University of Alberta             |")
    print("|                                                                                |")
    print("\\*------------------------------------------------------------------------------*/")

    print_usage_guide()
    model_path = 'TrainedPINNS/LeakyReLU_160M_10th_Epoch.pth' # Change to use different models
    wind_tunnel_app = WindTunnelApp(model_path)
    wind_tunnel_app.show()
    app.exec_()

if __name__ == "__main__":
    main()
