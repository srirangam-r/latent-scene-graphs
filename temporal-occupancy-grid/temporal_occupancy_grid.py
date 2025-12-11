import open3d as o3d
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy


class VoxelTemporalMap:
    """3D Voxel-based Temporal Occupancy Map"""
    
    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size
        self.all_voxels = []
        self.observation_id = 0
        
    def add_observation(self, point_cloud, timestamp, is_static=False, label="object"):
        """Add point cloud observation to temporal map"""
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            point_cloud, voxel_size=self.voxel_size)
        
        voxels = voxel_grid.get_voxels()
        
        for voxel in voxels:
            self.all_voxels.append({
                'coord': tuple(voxel.grid_index),
                'timestamp': timestamp,
                'static': is_static,
                'color': np.asarray(voxel.color) if hasattr(voxel, 'color') else [0.7, 0.7, 0.7],
                'label': label,
                'id': self.observation_id
            })
        
        self.observation_id += 1
    
    def get_confidence(self, timestamp, last_seen_time, is_static, decay_rate=0.15):
        """Calculate confidence with exponential decay"""
        time_diff_hours = (timestamp - last_seen_time).total_seconds() / 3600.0
        if is_static:
            return 1.0
        else:
            return np.exp(-decay_rate * time_diff_hours)
    
    def get_temporal_pointcloud(self, current_time, decay_rate=0.15, confidence_threshold=0.05):
        """Generate point cloud with temporal confidence colors"""
        points = []
        colors = []
        confidences = []
        labels = []
        
        for voxel_data in self.all_voxels:
            confidence = self.get_confidence(
                current_time,
                voxel_data['timestamp'],
                voxel_data['static'],
                decay_rate
            )
            
            if confidence < confidence_threshold:
                continue
            
            pos = (np.array(voxel_data['coord']) + 0.5) * self.voxel_size
            points.append(pos)
            confidences.append(confidence)
            labels.append(voxel_data['label'])
            
            if voxel_data['static']:
                color = [0, 0, 1]
            else:
                if voxel_data['label'] == 'drawer_open':
                    color = [1 - confidence, confidence, 0]
                elif voxel_data['label'] == 'drawer_closed':
                    color = [0, confidence, confidence]
                else:
                    color = [1 - confidence, confidence, 0]
            
            colors.append(color)
        
        pcd = o3d.geometry.PointCloud()
        if len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd, confidences, labels


class QuasiDynamicPOMDP:
    """POMDP for decision making with quasi-dynamic objects"""
    
    def __init__(self, temporal_map, decay_rate=0.15):
        self.map = temporal_map
        self.decay_rate = decay_rate
        self.robot_position = np.array([0, 0, 0.5])
        self.goal_position = np.array([0, 0, -0.5])
        
    def compute_belief_state(self, current_time):
        """Compute belief state from temporal map"""
        beliefs = {}
        voxel_coords = {}
        
        for voxel_data in self.map.all_voxels:
            label = voxel_data['label']
            
            confidence = self.map.get_confidence(
                current_time,
                voxel_data['timestamp'],
                voxel_data['static'],
                self.decay_rate
            )
            
            if label not in beliefs:
                beliefs[label] = []
                voxel_coords[label] = []
            
            beliefs[label].append(confidence)
            voxel_coords[label].append(voxel_data['coord'])
        
        belief_summary = {}
        for label, confs in beliefs.items():
            avg_confidence = np.mean(confs) if confs else 0
            belief_summary[label] = {
                'confidence': avg_confidence,
                'num_voxels': len(confs),
                'entropy': self._compute_entropy(avg_confidence)
            }
        
        return belief_summary, voxel_coords
    
    def _compute_entropy(self, confidence):
        """Shannon entropy for binary distribution"""
        if confidence <= 0.01 or confidence >= 0.99:
            return 0
        p = confidence
        return -(p * np.log2(p) + (1-p) * np.log2(1-p))
    
    def evaluate_action_value(self, action, belief_state, current_time):
        """Evaluate expected value of action"""
        drawer_open_belief = belief_state.get('drawer_open', {'confidence': 0, 'entropy': 0})
        drawer_closed_belief = belief_state.get('drawer_closed', {'confidence': 0, 'entropy': 0})
        
        total_entropy = drawer_open_belief['entropy'] + drawer_closed_belief['entropy']
        uncertainty = total_entropy / 2
        
        if action == 'navigate_direct':
            p_path_clear = drawer_open_belief['confidence']
            p_path_blocked = drawer_closed_belief['confidence']
            expected_value = p_path_clear * 95 + p_path_blocked * (-200)
            
        elif action == 'navigate_around':
            expected_value = 80
            
        elif action == 'reobserve':
            expected_value = -10 + uncertainty * 30 + 95
        else:
            expected_value = -float('inf')
        
        return expected_value
    
    def select_optimal_action(self, belief_state, current_time):
        """Select action maximizing expected value"""
        actions = ['navigate_direct', 'navigate_around', 'reobserve']
        
        action_values = {}
        for action in actions:
            action_values[action] = self.evaluate_action_value(action, belief_state, current_time)
        
        best_action = max(action_values, key=action_values.get)
        return best_action, action_values
    
    def explain_decision(self, action, action_values, belief_state):
        """Generate decision explanation"""
        drawer_open = belief_state.get('drawer_open', {'confidence': 0, 'entropy': 0})
        drawer_closed = belief_state.get('drawer_closed', {'confidence': 0, 'entropy': 0})
        
        explanation = "\n" + "="*70 + "\n"
        explanation += "POMDP Decision Analysis\n"
        explanation += "="*70 + "\n\n"
        
        explanation += "Belief State:\n"
        explanation += f"  Drawer OPEN:   confidence={drawer_open['confidence']:.2f}, entropy={drawer_open['entropy']:.2f}\n"
        explanation += f"  Drawer CLOSED: confidence={drawer_closed['confidence']:.2f}, entropy={drawer_closed['entropy']:.2f}\n\n"
        
        explanation += "Action Values:\n"
        for act, val in action_values.items():
            marker = ">" if act == action else " "
            explanation += f"  {marker} {act:20s}: {val:+7.1f}\n"
        
        explanation += f"\nOptimal Action: {action}\n"
        explanation += "="*70 + "\n"
        
        return explanation


def align_point_clouds(source, target):
    """Align source to target using ICP"""
    source_down = source.voxel_down_sample(voxel_size=0.05)
    target_down = target.voxel_down_sample(voxel_size=0.05)
    
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    threshold = 0.05
    trans_init = np.identity(4)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    source_aligned = source.transform(reg_p2p.transformation)
    return source_aligned, reg_p2p.transformation


def create_pomdp_decision_figure(pomdp, times, decisions, belief_history):
    """Visualize POMDP decisions over time"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('POMDP Decision Making for Quasi-Dynamic Navigation', 
                 fontsize=16, fontweight='bold')
    
    time_labels = [f"T{i}" for i in range(len(times))]
    open_confidences = [b.get('drawer_open', {'confidence': 0})['confidence'] for b in belief_history]
    closed_confidences = [b.get('drawer_closed', {'confidence': 0})['confidence'] for b in belief_history]
    open_entropies = [b.get('drawer_open', {'entropy': 0})['entropy'] for b in belief_history]
    closed_entropies = [b.get('drawer_closed', {'entropy': 0})['entropy'] for b in belief_history]
    
    # Confidence
    ax1 = axes[0]
    x = np.arange(len(times))
    width = 0.35
    
    ax1.bar(x - width/2, open_confidences, width, label='Drawer Open', color='orange', alpha=0.7)
    ax1.bar(x + width/2, closed_confidences, width, label='Drawer Closed', color='cyan', alpha=0.7)
    ax1.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax1.set_title('Belief State Evolution', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(time_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Uncertainty
    ax2 = axes[1]
    total_entropy = [o + c for o, c in zip(open_entropies, closed_entropies)]
    
    ax2.plot(x, total_entropy, 'ro-', linewidth=2, markersize=8, label='Total Uncertainty')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Entropy', fontsize=12, fontweight='bold')
    ax2.set_title('Uncertainty Level', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(time_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Actions
    ax3 = axes[2]
    action_colors = {'navigate_direct': 'green', 'navigate_around': 'orange', 'reobserve': 'red'}
    
    for i, (time_label, decision) in enumerate(zip(time_labels, decisions)):
        action = decision['action']
        color = action_colors.get(action, 'gray')
        ax3.barh(i, 1, color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.text(0.5, i, action.replace('_', ' ').title(), ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    ax3.set_yticks(x)
    ax3.set_yticklabels(time_labels)
    ax3.set_xlim([0, 1])
    ax3.set_xlabel('POMDP Decision', fontsize=12, fontweight='bold')
    ax3.set_title('Optimal Actions', fontsize=13)
    ax3.set_xticks([])
    
    legend_elements = [
        Patch(facecolor='green', label='Navigate Direct'),
        Patch(facecolor='orange', label='Navigate Around'),
        Patch(facecolor='red', label='Re-observe')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('pomdp_decisions.png', dpi=200, bbox_inches='tight')
    plt.show()


def visualize_sequence(point_clouds, titles, descriptions):
    """
    Interactive visualization - navigate with keyboard
    Press N for next, P for previous, Q to quit
    """
    print("\n" + "="*80)
    print("Interactive Visualization")
    print("="*80)
    print("Controls:")
    print("  N or SPACE - Next")
    print("  P or B - Previous")  
    print("  Q or ESC - Quit")
    print("="*80 + "\n")
    
    current_idx = 0
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=titles[current_idx], width=1400, height=900)
    
    # Render options
    opt = vis.get_render_option()
    opt.point_size = 5.0
    opt.background_color = np.array([0.05, 0.05, 0.05])
    
    # Add first point cloud
    vis.add_geometry(point_clouds[current_idx])
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    
    def next_view(vis):
        nonlocal current_idx
        vis.clear_geometries()
        current_idx = (current_idx + 1) % len(point_clouds)
        vis.add_geometry(point_clouds[current_idx])
        
        # Update window title
        print(f"\n[{current_idx}] {titles[current_idx]}")
        print(f"    {descriptions[current_idx]}")
        
        return False
    
    def prev_view(vis):
        nonlocal current_idx
        vis.clear_geometries()
        current_idx = (current_idx - 1) % len(point_clouds)
        vis.add_geometry(point_clouds[current_idx])
        
        # Update window title
        print(f"\n[{current_idx}] {titles[current_idx]}")
        print(f"    {descriptions[current_idx]}")
        
        return False
    
    def quit_view(vis):
        vis.destroy_window()
        return True
    
    # Register callbacks
    vis.register_key_callback(ord('N'), next_view)
    vis.register_key_callback(ord(' '), next_view)  # Space key
    vis.register_key_callback(ord('P'), prev_view)
    vis.register_key_callback(ord('B'), prev_view)
    vis.register_key_callback(ord('Q'), quit_view)
    vis.register_key_callback(256, quit_view)  # ESC key
    
    print(f"[{current_idx}] {titles[current_idx]}")
    print(f"    {descriptions[current_idx]}")
    
    vis.run()
    vis.destroy_window()


def create_synthetic_desk_data():
    """Generate synthetic desk geometry"""
    desk = o3d.geometry.TriangleMesh.create_box(width=1.2, height=0.05, depth=0.8)
    desk.translate([-0.6, 0, -0.4])
    desk.paint_uniform_color([0.6, 0.4, 0.2])
    
    drawer_open = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.2, depth=0.4)
    drawer_open.translate([-0.25, -0.35, -0.2])
    drawer_open.paint_uniform_color([0.8, 0.6, 0.3])
    
    drawer_closed = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.2, depth=0.4)
    drawer_closed.translate([-0.25, -0.05, -0.2])
    drawer_closed.paint_uniform_color([0.8, 0.6, 0.3])
    
    scene_open = desk + drawer_open
    scene_closed = desk + drawer_closed
    
    pcd_open = scene_open.sample_points_uniformly(number_of_points=8000)
    pcd_closed = scene_closed.sample_points_uniformly(number_of_points=8000)
    
    return pcd_open, pcd_closed


def align_point_clouds(source, target):
    """Align source to target using ICP"""
    source_down = source.voxel_down_sample(voxel_size=0.05)
    target_down = target.voxel_down_sample(voxel_size=0.05)
    
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    threshold = 0.05
    trans_init = np.identity(4)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    source_aligned = source.transform(reg_p2p.transformation)
    return source_aligned, reg_p2p.transformation


def create_pomdp_decision_figure(times, decisions, belief_history):
    """Visualize POMDP decisions over time"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('POMDP Decision Making for Quasi-Dynamic Navigation', 
                 fontsize=16, fontweight='bold')
    
    time_labels = [f"T{i}" for i in range(len(times))]
    open_confidences = [b.get('drawer_open', {'confidence': 0})['confidence'] for b in belief_history]
    closed_confidences = [b.get('drawer_closed', {'confidence': 0})['confidence'] for b in belief_history]
    open_entropies = [b.get('drawer_open', {'entropy': 0})['entropy'] for b in belief_history]
    closed_entropies = [b.get('drawer_closed', {'entropy': 0})['entropy'] for b in belief_history]
    
    # Confidence
    ax1 = axes[0]
    x = np.arange(len(times))
    width = 0.35
    
    ax1.bar(x - width/2, open_confidences, width, label='Drawer Open', color='orange', alpha=0.7)
    ax1.bar(x + width/2, closed_confidences, width, label='Drawer Closed', color='cyan', alpha=0.7)
    ax1.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax1.set_title('Belief State Evolution', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(time_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Uncertainty
    ax2 = axes[1]
    total_entropy = [o + c for o, c in zip(open_entropies, closed_entropies)]
    
    ax2.plot(x, total_entropy, 'ro-', linewidth=2, markersize=8, label='Uncertainty')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Entropy', fontsize=12, fontweight='bold')
    ax2.set_title('Uncertainty Level', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(time_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Actions
    ax3 = axes[2]
    action_colors = {'navigate_direct': 'green', 'navigate_around': 'orange', 'reobserve': 'red'}
    
    for i, (time_label, decision) in enumerate(zip(time_labels, decisions)):
        action = decision['action']
        color = action_colors.get(action, 'gray')
        ax3.barh(i, 1, color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.text(0.5, i, action.replace('_', ' ').title(), ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    ax3.set_yticks(x)
    ax3.set_yticklabels(time_labels)
    ax3.set_xlim([0, 1])
    ax3.set_xlabel('POMDP Decision', fontsize=12, fontweight='bold')
    ax3.set_title('Optimal Actions', fontsize=13)
    ax3.set_xticks([])
    
    legend_elements = [
        Patch(facecolor='green', label='Navigate Direct'),
        Patch(facecolor='orange', label='Navigate Around'),
        Patch(facecolor='red', label='Re-observe')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('pomdp_decisions.png', dpi=200, bbox_inches='tight')
    plt.show()


def run_demo():
    """Execute demo"""
    print("="*80)
    print("Quasi-Dynamic Mapping with POMDP")
    print("="*80)
    
    vtm = VoxelTemporalMap(voxel_size=0.02)
    pomdp = QuasiDynamicPOMDP(vtm, decay_rate=0.15)
    t0 = datetime.now()
    
    # Load data
    try:
        desk_open = o3d.io.read_point_cloud("desk_open.ply")
        desk_closed = o3d.io.read_point_cloud("desk_closed.ply")
        
        desk_open = desk_open.voxel_down_sample(voxel_size=0.01)
        desk_closed = desk_closed.voxel_down_sample(voxel_size=0.01)
        
        desk_closed, _ = align_point_clouds(desk_closed, desk_open)
        
        print("Loaded real scans")
        
    except Exception as e:
        print(f"Using synthetic data: {e}")
        desk_open, desk_closed = create_synthetic_desk_data()
    
    # Build temporal map and collect visualizations
    print("\nBuilding temporal map")
    print("="*80)
    
    vtm.add_observation(desk_open, t0, is_static=False, label="drawer_open")
    
    times = []
    decisions = []
    belief_history = []
    point_clouds = []
    titles = []
    descriptions = []
    
    # T0
    print("[T0]")
    belief_state, _ = pomdp.compute_belief_state(t0)
    action, action_values = pomdp.select_optimal_action(belief_state, t0)
    print(pomdp.explain_decision(action, action_values, belief_state))
    
    times.append(t0)
    decisions.append({'action': action, 'values': action_values})
    belief_history.append(belief_state)
    
    pcd_t0, _, _ = vtm.get_temporal_pointcloud(t0)
    point_clouds.append(pcd_t0)
    titles.append("T")
    descriptions.append(f"POMDP Action: {action} | All observations fresh (green)")
    
    # T1
    t1 = t0 + timedelta(hours=1)
    print("\n[T1]")
    belief_state, _ = pomdp.compute_belief_state(t1)
    action, action_values = pomdp.select_optimal_action(belief_state, t1)
    print(pomdp.explain_decision(action, action_values, belief_state))
    
    times.append(t1)
    decisions.append({'action': action, 'values': action_values})
    belief_history.append(belief_state)
    
    pcd_t1, _, _ = vtm.get_temporal_pointcloud(t1)
    point_clouds.append(pcd_t1)
    titles.append("T1")
    descriptions.append(f"POMDP Action: {action} | Confidence decaying (yellow-green)")
    
    # T2
    t2 = t0 + timedelta(hours=3)
    print("\n[T2]")
    belief_state, _ = pomdp.compute_belief_state(t2)
    action, action_values = pomdp.select_optimal_action(belief_state, t2)
    print(pomdp.explain_decision(action, action_values, belief_state))
    
    times.append(t2)
    decisions.append({'action': action, 'values': action_values})
    belief_history.append(belief_state)
    
    pcd_t2, _, _ = vtm.get_temporal_pointcloud(t2)
    point_clouds.append(pcd_t2)
    titles.append("T2")
    descriptions.append(f"POMDP Action: {action} | Low confidence (orange-red)")
    
    # T3
    t3 = t0 + timedelta(hours=5)
    print("\n[T3]")
    vtm.add_observation(desk_closed, t3, is_static=False, label="drawer_closed")
    
    belief_state, _ = pomdp.compute_belief_state(t3)
    action, action_values = pomdp.select_optimal_action(belief_state, t3)
    print(pomdp.explain_decision(action, action_values, belief_state))
    
    times.append(t3)
    decisions.append({'action': action, 'values': action_values})
    belief_history.append(belief_state)
    
    pcd_t3, _, _ = vtm.get_temporal_pointcloud(t3)
    point_clouds.append(pcd_t3)
    titles.append("T3")
    descriptions.append(f"POMDP Action: {action} | Old position (red) + New position (cyan)")
    
    # T4
    t4 = t3 + timedelta(hours=1)
    print("\n[T4]")
    belief_state, _ = pomdp.compute_belief_state(t4)
    action, action_values = pomdp.select_optimal_action(belief_state, t4)
    print(pomdp.explain_decision(action, action_values, belief_state))
    
    times.append(t4)
    decisions.append({'action': action, 'values': action_values})
    belief_history.append(belief_state)
    
    pcd_t4, _, _ = vtm.get_temporal_pointcloud(t4)
    point_clouds.append(pcd_t4)
    titles.append("T4")
    descriptions.append(f"POMDP Action: {action} | Old observations fading away")
    
    # Interactive visualization
    print("\n" + "="*80)
    print("Starting interactive viewer")
    print("="*80)
    
    visualize_sequence(point_clouds, titles, descriptions)
    
    # Generate POMDP figure
    print("\nGenerating POMDP decision figure")
    create_pomdp_decision_figure(times, decisions, belief_history)
    
    print("\nDemo complete")
    print(f"Voxels tracked: {len(vtm.all_voxels)}")
    print(f"Temporal states: {len(times)}")


def main():
    print("\nQuasi-Dynamic Mapping + POMDP Demo")
    print("Press Enter to start\n")
    input()
    
    run_demo()
    
    print("\nGenerated: pomdp_decisions.png")


if __name__ == "__main__":
    main()
