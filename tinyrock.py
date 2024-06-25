import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import piexif
import json
import os
import sys

def print_environment_info():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print("Python path:")
    for path in sys.path:
        print(f"  {path}")
    print("\nInstalled packages:")
    os.system(f"{sys.executable} -m pip list")

print_environment_info()

print("\nAttempting to import required libraries...")

try:
    import cv2
    print("OpenCV (cv2) imported successfully.")
except ImportError as e:
    print(f"Error importing cv2: {e}")

try:
    import numpy as np
    print("NumPy imported successfully.")
except ImportError as e:
    print(f"Error importing numpy: {e}")

try:
    from scipy.optimize import minimize
    print("SciPy optimize imported successfully.")
except ImportError as e:
    print(f"Error importing scipy.optimize: {e}")

try:
    import matplotlib.pyplot as plt
    print("Matplotlib pyplot imported successfully.")
except ImportError as e:
    print(f"Error importing matplotlib.pyplot: {e}")

try:
    from PIL import Image
    print("PIL Image imported successfully.")
except ImportError as e:
    print(f"Error importing PIL.Image: {e}")

try:
    import piexif
    print("piexif imported successfully.")
except ImportError as e:
    print(f"Error importing piexif: {e}")

print("\nImport attempts completed.")

class PaverPatioOptimizer:
    def __init__(self):
        self.image = None
        self.metadata = None
        self.user_defined_area = None
        self.patio_dimensions = None
        self.optimized_layout = None
        self.paver_sizes = {
            "3x6": (0.0762, 0.1524),  # 3x6 inches in meters
            "6x6": (0.1524, 0.1524),  # 6x6 inches in meters
            "9x6": (0.2286, 0.1524),  # 9x6 inches in meters
            "12x12": (0.3048, 0.3048)  # 12x12 inches in meters
        }

    def load_image_and_metadata(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: The file {image_path} does not exist.")
            return False

        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Error: Unable to load the image {image_path}. It may be corrupted or in an unsupported format.")
            return False

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        self.metadata = self.extract_metadata(image_path)
        
        if not self.metadata:
            print("No metadata found in the image. You'll need to enter it manually.")
        
        return True

    def extract_metadata(self, image_path):
        metadata = {}
        try:
            img = Image.open(image_path)
            exif_dict = piexif.load(img.info.get('exif', b''))
            
            if piexif.GPSIFD in exif_dict:
                gps_info = exif_dict[piexif.GPSIFD]
                if 6 in gps_info:  # Altitude
                    altitude = gps_info[6][0] / gps_info[6][1]
                    metadata['altitude'] = altitude
            
            if piexif.ExifIFD in exif_dict:
                exif_info = exif_dict[piexif.ExifIFD]
                if 37386 in exif_info:  # Focal Length
                    focal_length = exif_info[37386][0] / exif_info[37386][1]
                    metadata['focal_length'] = focal_length
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
        
        return metadata

    def estimate_fov(self, focal_length, sensor_width=35):
        # Estimate FOV based on focal length and sensor width
        # Default sensor width is 35mm (full frame)
        return 2 * np.arctan(sensor_width / (2 * focal_length)) * 180 / np.pi

    def display_image(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.image)
        ax.set_title("Click to define patio area (close window when done)")
        ax.axis('off')
        
        points = plt.ginput(-1, show_clicks=True)
        plt.close(fig)
        
        self.user_defined_area = points
        
        # Display the image again with the selected points
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.image)
        ax.plot(*zip(*self.user_defined_area), 'ro-')
        ax.set_title("Selected Patio Area")
        ax.axis('off')
        plt.show()

    def calculate_real_world_dimensions(self):
        if not self.metadata or 'altitude' not in self.metadata or 'camera_fov' not in self.metadata:
            print("Error: Missing required metadata (altitude and/or camera_fov)")
            return False

        altitude = self.metadata['altitude']
        fov = self.metadata['camera_fov']
        
        area_points = np.array(self.user_defined_area)
        area_pixels = cv2.contourArea(area_points.astype(np.float32))
        
        image_width = self.image.shape[1]
        image_width_meters = 2 * altitude * np.tan(np.radians(fov / 2))
        
        pixel_area = (image_width_meters / image_width) ** 2
        total_area = area_pixels * pixel_area
        
        aspect_ratio = max(np.ptp(area_points[:, 0]) / np.ptp(area_points[:, 1]),
                           np.ptp(area_points[:, 1]) / np.ptp(area_points[:, 0]))
        width = np.sqrt(total_area * aspect_ratio)
        length = total_area / width
        
        self.patio_dimensions = (width, length)
        return True

    def optimize_paver_layout(self):
        if not self.patio_dimensions:
            print("Error: Patio dimensions not calculated yet")
            return False

        width, length = self.patio_dimensions
        total_area = width * length
        
        def objective(x):
            paver_areas = [size[0] * size[1] * count for size, count in zip(self.paver_sizes.values(), x)]
            total_paver_area = sum(paver_areas)
            area_difference = abs(total_area - total_paver_area)
            
            # Penalty for using only one type of paver
            variety_penalty = 1000 * (4 - sum(1 for count in x if count > 0))
            
            # Penalty for having less than 10% of any paver type
            min_quantity_penalty = sum(1000 for count in x if count < 0.1 * sum(x))
            
            return area_difference + variety_penalty + min_quantity_penalty
        
        # Initial guess: equal distribution of pavers
        x0 = [total_area / (4 * size[0] * size[1]) for size in self.paver_sizes.values()]
        
        # Constraints: non-negative integer number of pavers
        constraints = [{'type': 'ineq', 'fun': lambda x: x}]
        
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)
        
        self.optimized_layout = {size: max(int(count), 1) for size, count in zip(self.paver_sizes.keys(), result.x)}
        return True

    def calculate_cuts(self):
        width, length = self.patio_dimensions
        cuts = {}
        for paver_name, (paver_width, paver_length) in self.paver_sizes.items():
            whole_pavers_width = int(width // paver_width)
            whole_pavers_length = int(length // paver_length)
            cuts[paver_name] = (width % paver_width > 0.01) * whole_pavers_length + (length % paver_length > 0.01) * whole_pavers_width
        return cuts

    def visualize_layout(self):
        if not self.patio_dimensions or not self.optimized_layout:
            print("Error: Optimization not performed yet")
            return

        width, length = self.patio_dimensions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Plot the original image with selected area
        ax1.imshow(self.image)
        polygon = plt.Polygon(self.user_defined_area, fill=None, edgecolor='r')
        ax1.add_patch(polygon)
        ax1.set_title("Selected Patio Area")
        ax1.axis('off')

        # Plot the optimized layout
        ax2.set_xlim(0, width)
        ax2.set_ylim(0, length)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.optimized_layout)))
        color_dict = dict(zip(self.paver_sizes.keys(), colors))
        
        legend_elements = []

        x, y = 0, 0
        row = 0
        while y < length:
            x = 0
            while x < width:
                for paver_name, count in self.optimized_layout.items():
                    if count > 0:
                        paver_width, paver_height = self.paver_sizes[paver_name]
                        if x + paver_width <= width and y + paver_height <= length:
                            rect = plt.Rectangle((x, y), paver_width, paver_height, 
                                                 fill=True, facecolor=color_dict[paver_name], 
                                                 edgecolor='black', linewidth=0.5)
                            ax2.add_patch(rect)
                            x += paver_width
                            self.optimized_layout[paver_name] -= 1
                            break
                else:
                    x += min(self.paver_sizes.values(), key=lambda size: size[0])[0]
            y += max(self.paver_sizes.values(), key=lambda size: size[1])[1]
            row += 1
            # Offset every other row for a more realistic brick pattern
            if row % 2 == 1:
                x = -min(self.paver_sizes.values(), key=lambda size: size[0])[0] / 2

        for paver_name, color in color_dict.items():
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=f'{paver_name}: {self.optimized_layout[paver_name]}'))

        cuts = self.calculate_cuts()
        for paver_name, cut_count in cuts.items():
            legend_elements.append(plt.Line2D([0], [0], color='white', label=f'{paver_name} cuts: {cut_count}', markerfacecolor='black', markersize=15))

        ax2.legend(handles=legend_elements, title="Paver Quantities and Cuts", loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_title("Optimized Paver Layout")
        ax2.set_xlabel("Width (m)")
        ax2.set_ylabel("Length (m)")
        ax2.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.show()

    def display_results(self):
        if not self.patio_dimensions or not self.optimized_layout:
            print("Error: Optimization not performed yet")
            return

        print(f"Patio Dimensions: {self.patio_dimensions[0]:.2f}m x {self.patio_dimensions[1]:.2f}m")
        print("\nOptimized Paver Layout:")
        for paver_name, count in self.optimized_layout.items():
            print(f"  {paver_name}: {count} pieces")
        
        cuts = self.calculate_cuts()
        print("\nEstimated Cuts:")
        for paver_name, cut_count in cuts.items():
            print(f"  {paver_name}: {cut_count} cuts")

def main():
    optimizer = PaverPatioOptimizer()
    
    image_path = r"C:\Users\cc123\repos\TinyRock\img\test1.jpg"
    if not optimizer.load_image_and_metadata(image_path):
        print("Failed to load image or metadata. Please check the file path and try again.")
        return

    print("Extracted Metadata:", optimizer.metadata)

    # If altitude is missing, ask user to input it
    if 'altitude' not in optimizer.metadata:
        altitude = float(input("Enter the drone's altitude in meters: "))
        optimizer.metadata['altitude'] = altitude
    else:
        print(f"Altitude found in metadata: {optimizer.metadata['altitude']} meters")
        use_metadata = input("Do you want to use this altitude? (y/n): ").lower().strip()
        if use_metadata != 'y':
            altitude = float(input("Enter the drone's altitude in meters: "))
            optimizer.metadata['altitude'] = altitude

    # Calculate FOV from focal length
    if 'focal_length' in optimizer.metadata:
        optimizer.metadata['camera_fov'] = optimizer.estimate_fov(optimizer.metadata['focal_length'])
        print(f"Estimated field of view: {optimizer.metadata['camera_fov']:.2f} degrees")
    else:
        focal_length = float(input("Enter the camera's focal length in mm: "))
        optimizer.metadata['camera_fov'] = optimizer.estimate_fov(focal_length)
        print(f"Estimated field of view: {optimizer.metadata['camera_fov']:.2f} degrees")

    optimizer.display_image()
    if not optimizer.calculate_real_world_dimensions():
        print("Failed to calculate patio dimensions.")
        return

    if not optimizer.optimize_paver_layout():
        print("Failed to optimize paver layout.")
        return

    optimizer.display_results()
    optimizer.visualize_layout()

if __name__ == "__main__":
    main()