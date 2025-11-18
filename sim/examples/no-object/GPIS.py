import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle
import numpy as np

class InverseMultiquadricKernel:
    def __init__(self, c=2.0):
        self.c = c

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        dists = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
        K = 1.0 / np.sqrt(dists + self.c**2)
        return K
    
class GaussianProcessRegressor:
    def __init__(self, kernel, alpha=1e-10):
        self.kernel = kernel
        self.alpha = alpha
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        K = self.kernel(X, X) + self.alpha * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test, return_std=False):
        X_test = np.atleast_2d(X_test) 
        K_trans = self.kernel(X_test, self.X_train)

        y_mean = K_trans.dot(self.K_inv).dot(self.y_train)

        if return_std:
            K_test = self.kernel(X_test, X_test)
            y_var = K_test - K_trans.dot(self.K_inv).dot(K_trans.T)
            y_std = np.sqrt(np.diag(y_var))
            return y_mean, y_std
        else:
            return y_mean

class GPISModel:
    def __init__(self, x, y, yaw, laser1, laser2, value1, value2,
                boundary_sample_ratio=1, interior_sample_ratio=1, outerior_sample_ratio=1,
                kernel=None, alpha=1e-2, curvature_threshold=-1, exit_point = None):
        self.x = x
        self.y = y
        self.yaw = yaw
        x = np.array(x)
        y = np.array(y)
        yaw = np.array(yaw)
        laser1 = np.array(laser1)
        laser2 = np.array(laser2)
        value1 = np.array(value1)
        value2 = np.array(value2)

        self.x_wall = x[(value1 == 0) | (value2 == 0)]
        self.y_wall = y[(value1 == 0) | (value2 == 0)]
        self.x_inside = x[(value1 == -1) | (value2 == -1)]
        self.y_inside = y[(value1 == -1) | (value2 == -1)]
        self.x_outside = x[(value1 == 1) | (value2 == 1)]
        self.y_outside = y[(value1 == 1) | (value2 == 1)]

        laser1x = (laser1) * np.cos(yaw - 0.7853981)
        laser1y = (laser1) * np.sin(yaw - 0.7853981)
        laser2x = (laser2) * np.cos(yaw - 0.7853981)
        laser2y = (laser2) * np.sin(yaw - 0.7853981)

        laserx = np.where((value1 == 0) & (value2 == 0), (laser1x + laser2x) / 2, 
                        np.where(value1 == 0, laser1x, 
                                np.where(value2 == 0, laser2x, 0)))
        lasery = np.where((value1 == 0) & (value2 == 0), (laser1y + laser2y) / 2, 
                        np.where(value1 == 0, laser1y, 
                                np.where(value2 == 0, laser2y, 0)))

        self.laser1x_wall = laserx[(value1 == 0) | (value2 == 0)]
        self.laser1y_wall = lasery[(value1 == 0) | (value2 == 0)]
        self.laser1x_inside = laserx[(value1 == -1) | (value2 == -1)]
        self.laser1y_inside = lasery[(value1 == -1) | (value2 == -1)]
        self.laser1x_outside = laserx[(value1 == 1) | (value2 == 1)]
        self.laser1y_outside = lasery[(value1 == 1) | (value2 == 1)]
        print(self.laser1x_wall)

        print(f"x_wall size: {self.x_wall.size}")
        print(f"y_wall size: {self.y_wall.size}")
        print(f"x_inside size: {self.x_inside.size}")
        print(f"y_inside size: {self.y_inside.size}")
        print(f"laser1x_wall size: {self.laser1x_wall.size}")
        print(f"laser1y_wall size: {self.laser1y_wall.size}")

        self.X_boundary = np.vstack([self.x_wall+self.laser1x_wall, self.y_wall + self.laser1y_wall]).T
        self.y_boundary = np.zeros(len(self.x_wall))
        self.X_interior = np.vstack([self.x_inside, self.y_inside]).T
        self.y_interior = -np.ones(len(self.x_inside))
        self.X_outerior = np.vstack([self.x_outside, self.y_outside]).T
        self.y_outerior = np.ones(len(self.x_outside))
        self.boundary_sample_ratio = boundary_sample_ratio
        self.interior_sample_ratio = interior_sample_ratio
        self.outerior_sample_ratio = outerior_sample_ratio
        self.kernel = kernel if kernel else InverseMultiquadricKernel(c=1.5)
        self.alpha = alpha
        self.curvature_threshold = curvature_threshold
        
        self.X_train = None
        self.y_train = None
        self.gp = None
        self.Z = None
        self.sigma = None
        self.contour_points = None
        self.penalized_uncertainty_grid = None
        self.original_uncertainty_grid = None
        self.significant_points = None
        self.exit_point = exit_point
        self.contour_sigma_penalized = None
        self.max_uncertainty_point = None
        self.weights = None
        self.curvature = None
        self.uncertainty_retained_percentage = None
        self.contour_points_all = None
    
    def sample_data(self):
        num_boundary_samples = int(len(self.X_boundary) * self.boundary_sample_ratio)
        boundary_indices = np.random.choice(len(self.X_boundary), num_boundary_samples, replace=False)
        X_boundary_sampled = self.X_boundary[boundary_indices]
        y_boundary_sampled = self.y_boundary[boundary_indices]
        
        num_interior_samples = int(len(self.X_interior) * self.interior_sample_ratio)
        interior_indices = np.random.choice(len(self.X_interior), num_interior_samples, replace=False)
        X_interior_sampled = self.X_interior[interior_indices]
        y_interior_sampled = self.y_interior[interior_indices]

        num_outerior_samples = int(len(self.X_outerior) * self.outerior_sample_ratio)
        outerior_indices = np.random.choice(len(self.X_outerior), num_outerior_samples, replace=False)
        X_outerior_sampled = self.X_outerior[outerior_indices]
        y_outerior_sampled = self.y_outerior[outerior_indices]

        self.X_train = np.vstack([X_boundary_sampled, X_interior_sampled, X_outerior_sampled])
        self.y_train = np.concatenate([y_boundary_sampled, y_interior_sampled, y_outerior_sampled])
    
    def train_model(self):
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.gp.fit(self.X_train, self.y_train)
    
    def predict(self):
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)
        X_test = np.vstack([X.ravel(), Y.ravel()]).T
        y_pred, sigma = self.gp.predict(X_test, return_std=True)
        self.Z = y_pred.reshape(X.shape)
        self.sigma = sigma.reshape(X.shape)
        line_segments = self._marching_squares(100, self.Z.ravel(), self.sigma.ravel(), -6, 12/99, -6, 12/99)
        self.contour_points_all = self._connect_contour_segments(line_segments, len(line_segments))
        x_vals = [point.x for point in self.contour_points_all]
        y_vals = [point.y for point in self.contour_points_all]
        self.contour_points = np.column_stack((x_vals, y_vals))
        self.weights = self.gp.K_inv.dot(self.y_train) 
        self.curvature = self._compute_curvature_kernel(self.contour_points, self.weights, self.X_train, self.kernel)
        print(f"curvatures: {self.curvature}")
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T

        contour_sigma_interp = [point.y_std for point in self.contour_points_all]

        self.significant_points = self._find_high_curvature_clusters_using_curvature(self.contour_points, self.curvature, self.curvature_threshold)
        if self.exit_point is not None:
            self.significant_points = np.vstack([self.significant_points, self.exit_point])
            print(f"exit_points: {self.exit_point}")
        print(f"significant_points: {self.significant_points}")
        print(f"significant_points shape: {self.significant_points.shape}")
        penalty = self._potential_function(grid_points, self.significant_points, c=0.3)
        penalty_contour = self._potential_function(self.contour_points, self.significant_points, c=0.3)

        original_uncertainty = sigma.ravel()
        penalized_uncertainty = original_uncertainty + penalty
        self.original_uncertainty_grid = original_uncertainty.reshape(X.shape)
        self.penalized_uncertainty_grid = penalized_uncertainty.reshape(X.shape)
        self.contour_sigma_penalized = contour_sigma_interp + penalty_contour
        mask = y_pred <= 0.1  
        uncertainty_selected = original_uncertainty[mask]
        num_points = np.sum(mask)

        if num_points > 0:
            retained_uncertainty = np.sum(uncertainty_selected)
            self.uncertainty_retained_percentage = (retained_uncertainty / num_points) * 100
        else:
            self.uncertainty_retained_percentage = 0.0  



    class Point:
        def __init__(self, x, y, y_std):
            self.x = x
            self.y = y
            self.y_std = y_std

    class LineSegment:
        def __init__(self, start, end):
            self.start = start
            self.end = end   
    def _save_ordered_contour_point(self, ordered_contour_points, x, y, y_std):
        ordered_contour_points.append(self.Point(x, y, y_std))             
    def _marching_squares(self, grid_size, y_preds, y_stds, x_min, x_step, y_min, y_step):
        line_segments = []
    
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                idx_00 = i * grid_size + j   
                idx_01 = i * grid_size + (j + 1)  
                idx_10 = (i + 1) * grid_size + j  
                idx_11 = (i + 1) * grid_size + (j + 1) 

                top_left = y_preds[idx_00] > 0
                top_right = y_preds[idx_01] > 0
                bottom_left = y_preds[idx_10] > 0
                bottom_right = y_preds[idx_11] > 0

                cell_index = (top_left << 3) | (top_right << 2) | (bottom_right << 1) | bottom_left

                if cell_index == 1 or cell_index == 14:  # 0001 or 1110
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_10]))
                    x1 = x_min + j * x_step
                    y1 = y_min + i * y_step + t1 * y_step
                    y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_10] - y_stds[idx_00])

                    t2 = abs(y_preds[idx_10]) / (abs(y_preds[idx_10]) + abs(y_preds[idx_11]))
                    x2 = x_min + j * x_step + t2 * x_step
                    y2 = y_min + (i + 1) * y_step
                    y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 2 or cell_index == 13:  # 0010 or 1101
                    t1 = abs(y_preds[idx_01]) / (abs(y_preds[idx_01]) + abs(y_preds[idx_11]))
                    x1 = x_min + (j + 1) * x_step
                    y1 = y_min + i * y_step + t1 * y_step
                    y_std1 = y_stds[idx_01] + t1 * (y_stds[idx_11] - y_stds[idx_01])

                    t2 = abs(y_preds[idx_10]) / (abs(y_preds[idx_10]) + abs(y_preds[idx_11]))
                    x2 = x_min + j * x_step + t2 * x_step
                    y2 = y_min + (i + 1) * y_step
                    y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 3 or cell_index == 12:  # 0011 or 1100
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_10]))
                    x1 = x_min + j * x_step
                    y1 = y_min + i * y_step + t1 * y_step
                    y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_10] - y_stds[idx_00])

                    t2 = abs(y_preds[idx_01]) / (abs(y_preds[idx_01]) + abs(y_preds[idx_11]))
                    x2 = x_min + (j + 1) * x_step
                    y2 = y_min + i * y_step + t2 * y_step
                    y_std2 = y_stds[idx_01] + t2 * (y_stds[idx_11] - y_stds[idx_01])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 4 or cell_index == 11:  # 0100 or 1011
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_01]))
                    x1 = x_min + j * x_step + t1 * x_step
                    y1 = y_min + i * y_step
                    y_std1 = y_stds[idx_01] + t1 * (y_stds[idx_01] - y_stds[idx_00])

                    t2 = abs(y_preds[idx_01]) / (abs(y_preds[idx_01]) + abs(y_preds[idx_11]))
                    x2 = x_min + (j + 1) * x_step
                    y2 = y_min + i * y_step + t2 * y_step
                    y_std2 = y_stds[idx_01] + t2 * (y_stds[idx_11] - y_stds[idx_01])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 5 or cell_index == 10:
                    # Case for cellIndex == 5 or 10, which doesn't require handling in your original code
                    pass

                elif cell_index == 6 or cell_index == 9:  # 0110 or 1001
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_01]))
                    x1 = x_min + j * x_step + t1 * x_step
                    y1 = y_min + i * y_step
                    y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_01] - y_stds[idx_00])

                    t2 = abs(y_preds[idx_10]) / (abs(y_preds[idx_11]) + abs(y_preds[idx_10]))
                    x2 = x_min + j * x_step + t2 * x_step
                    y2 = y_min + (i + 1) * y_step
                    y_std2 = y_stds[idx_10] + t2 * (y_stds[idx_11] - y_stds[idx_10])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 7 or cell_index == 8:  # 0111 or 1000
                    t1 = abs(y_preds[idx_00]) / (abs(y_preds[idx_01]) + abs(y_preds[idx_00]))
                    x1 = x_min + j * x_step + t1 * x_step
                    y1 = y_min + i * y_step
                    y_std1 = y_stds[idx_00] + t1 * (y_stds[idx_00] - y_stds[idx_10])

                    t2 = abs(y_preds[idx_00]) / (abs(y_preds[idx_00]) + abs(y_preds[idx_10]))
                    x2 = x_min + j * x_step
                    y2 = y_min + i * y_step + t2 * y_step
                    y_std2 = y_stds[idx_00] + t2 * (y_stds[idx_10] - y_stds[idx_00])

                    line_segments.append(self.LineSegment(self.Point(x1, y1, y_std1), self.Point(x2, y2, y_std2)))

                elif cell_index == 0 or cell_index == 15:
                    pass

                else:
                    pass

        return line_segments
    def _connect_contour_segments(self, line_segments, segment_count):
        visited = [False] * segment_count
        ordered_contour_points = []

        for i in range(segment_count):
            if visited[i]:
                continue

            self._save_ordered_contour_point(ordered_contour_points, line_segments[i].start.x, line_segments[i].start.y, line_segments[i].start.y_std)
            self._save_ordered_contour_point(ordered_contour_points, line_segments[i].end.x, line_segments[i].end.y, line_segments[i].end.y_std)
            visited[i] = True

            current_end = line_segments[i].end
            found_start = True
            
            while found_start:
                found_start = False
                for j in range(segment_count):
                    if not visited[j]:
                        if (abs(line_segments[j].start.x - current_end.x) < 1e-6 and 
                            abs(line_segments[j].start.y - current_end.y) < 1e-6):
                            self._save_ordered_contour_point(ordered_contour_points, line_segments[j].end.x, line_segments[j].end.y, line_segments[j].end.y_std)
                            current_end = line_segments[j].end
                            visited[j] = True
                            found_start = True
                            break
                        elif (abs(line_segments[j].end.x - current_end.x) < 1e-6 and 
                            abs(line_segments[j].end.y - current_end.y) < 1e-6):
                            self._save_ordered_contour_point(ordered_contour_points, line_segments[j].start.x, line_segments[j].start.y, line_segments[j].start.y_std)
                            current_end = line_segments[j].start
                            visited[j] = True
                            found_start = True
                            break

        return ordered_contour_points   
    def _potential_function(self, x, significant_points, c):
        distances = np.linalg.norm(x[:, np.newaxis, :] - significant_points[np.newaxis, :, :], axis=2)
        P = -np.exp(-distances**2 / (2 * c**2))
        return np.min(P, axis=1)
    
    
    def _find_high_curvature_clusters_using_curvature(self, contour_points, curvatures, curvature_threshold=0.5):
    
        clusters = []
        current_cluster = []
        n = len(contour_points)
    
        for i in range(n):
            if curvatures[i] < curvature_threshold:
                current_cluster.append(contour_points[i])
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = []
    
        if current_cluster:
            clusters.append(current_cluster)
    
        if clusters and len(clusters) > 1:
            if curvatures[0] < curvature_threshold and curvatures[-1] < curvature_threshold:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop(-1)
    
        significant_points = []
        for cluster in clusters:
            cluster = np.array(cluster)
            centroid = np.mean(cluster, axis=0)
            significant_points.append(centroid)
    
        return np.array(significant_points)

    def _compute_curvature_kernel(self, contour_points, weights, X_train, kernel):
        c2 = kernel.c**2
        curvatures = []

        for x in contour_points:
            diffs = x - X_train
            d2 = np.sum(diffs**2, axis=1)  # ||x - xi||^2
            d2_c2 = d2 + c2

            grad = np.sum(weights[:, None] * (-diffs / d2_c2[:, None]**(3 / 2)), axis=0)

            hessian = np.zeros((2, 2))
            for i, diff in enumerate(diffs):
                outer = np.outer(diff, diff)
                hessian += weights[i] * (3 * outer / d2_c2[i]**(5 / 2) - np.eye(2) / d2_c2[i]**(3 / 2))

            grad_norm = np.linalg.norm(grad)
            if grad_norm == 0:
                curvatures.append(0)
                continue
            tr_hessian = np.trace(hessian)
            numerator = grad @ hessian @ grad - grad_norm**2 * tr_hessian
            curvature = numerator / grad_norm**3
            curvatures.append(curvature)

        return np.array(curvatures)
    
    def compute_normal(self, next_point, weights, X_train, kernel):
        c2 = kernel.c**2
        curvatures = []

        diffs = next_point - X_train
        d2 = np.sum(diffs**2, axis=1)  # ||x - xi||^2
        d2_c2 = d2 + c2

        # gradient
        grad = np.sum(weights[:, None] * (-diffs / d2_c2[:, None]**(3 / 2)), axis=0)
        horizontal_grad = grad[:2] 
        normal = np.arctan2(horizontal_grad[1], horizontal_grad[0])  
        return normal
    
    def find_max_uncertainty_point(self):
        max_uncertainty_index = np.argmax(self.contour_sigma_penalized)
        self.max_uncertainty_point = self.contour_points[max_uncertainty_index]
        return self.max_uncertainty_point

    def plot_results(self, filename=None):
        x = np.linspace(-6, 6, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(14, 6))

        cubes = [
            {"name": "fancy_cube1", "position": np.array([-1.5, -1.1, 0]), "scale": np.array([1, 0.2, 2])},
            {"name": "fancy_cube2", "position": np.array([0.5, -2.1, 0]), "scale": np.array([3, 0.2, 2])},
            {"name": "fancy_cube3", "position": np.array([-1.1, -1.7, 0]), "scale": np.array([0.2, 1, 2])},
            {"name": "fancy_cube4", "position": np.array([-2.1, 0.5, 0]), "scale": np.array([0.2, 3.4, 2])},
            {"name": "fancy_cube5", "position": np.array([2.1, 0.0, 0]), "scale": np.array([0.2, 4.4, 2])},
            {"name": "fancy_cube6", "position": np.array([-0.6, 2.1, 0]), "scale": np.array([2.8, 0.2, 2])}
        ]

        ax1 = plt.subplot(1, 2, 1)
        cf1 = ax1.contourf(X, Y, self.Z, levels=np.linspace(self.Z.min(), self.Z.max(), 100), cmap="viridis")
        plt.colorbar(cf1, ax=ax1, label='GPIS Value')
        ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap="coolwarm", edgecolor="none", s=10)
        if self.max_uncertainty_point is not None:
            ax1.scatter(self.max_uncertainty_point[0], self.max_uncertainty_point[1], color='red', s=100, edgecolor='black', label='Max Uncertainty Point')
        ax1.contour(X, Y, self.Z, levels=[0], colors='red')
        ax1.scatter(self.significant_points[:, 0], self.significant_points[:, 1], c='white', s=30, label='Significant Curvature Points')
        if self.exit_point is not None:
            ax1.scatter(self.exit_point[0], self.exit_point[1], c='gray', s=30, label='Exit Points')



        ax1.set_title("2D GPIS with RBF Kernel")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.legend()

        for cube in cubes:
            pos = cube["position"]*1.33
            scale = cube["scale"]*1.33
            bottom_left = pos[:2] - scale[:2] / 2
            width, height = scale[0], scale[1]
            rect = Rectangle(bottom_left, width, height, linewidth=1.5, edgecolor='blue', facecolor='skyblue', alpha=0.5)
            ax1.add_patch(rect)
            ax1.text(pos[0], pos[1], cube["name"], fontsize=6, ha='center', va='center', color='black')

        ax2 = plt.subplot(1, 2, 2)
        cf2 = ax2.contourf(X, Y, self.penalized_uncertainty_grid, levels=np.linspace(self.penalized_uncertainty_grid.min(), self.penalized_uncertainty_grid.max(), 100), cmap="viridis")
        plt.colorbar(cf2, ax=ax2, label='Penalized Uncertainty (Std)')
        ax2.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap="coolwarm", edgecolor="none", s=10)
        ax2.set_title("Uncertainty (Std) with Penalty")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_aspect('equal')
        ax2.grid(True)

        for cube in cubes:
            pos = cube["position"]*1.33
            scale = cube["scale"]*1.33
            bottom_left = pos[:2] - scale[:2] / 2
            width, height = scale[0], scale[1]
            rect = Rectangle(bottom_left, width, height, linewidth=1.5, edgecolor='blue', facecolor='skyblue', alpha=0.5)
            ax2.add_patch(rect)
            ax2.text(pos[0], pos[1], cube["name"], fontsize=6, ha='center', va='center', color='black')

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()
