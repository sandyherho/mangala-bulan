import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Animator:
    @staticmethod
    def create_gif(result: dict, filename: str, config: dict, output_dir: str = "outputs"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        x = result['x']
        t = result['t']
        T = result['T']
        
        n_frames = min(config.get('n_frames', 200), len(t))
        frame_indices = np.linspace(0, len(t)-1, n_frames, dtype=int)
        
        print(f"    Creating 3D animation ({n_frames} frames)...")
        
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        colormap = config.get('colormap', 'twilight')
        cmap = plt.get_cmap(colormap)
        
        ax.set_xlabel('Position [mm]', fontsize=14, color='white', labelpad=12)
        ax.set_ylabel('Time [s]', fontsize=14, color='white', labelpad=12)
        ax.set_zlabel('O₂ Concentration [mg/ml]', fontsize=14, color='white', labelpad=12)
        
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, t[-1])
        ax.set_zlim(np.min(T)*0.9, np.max(T)*1.1)
        
        ax.tick_params(colors='white', labelsize=12)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.grid(True, alpha=0.2, color='white', linestyle='--')
        
        # Title
        title_text = ax.text2D(0.5, 0.97, config.get('scenario_name', 'Oxygen Diffusion'),
                              transform=ax.transAxes, fontsize=16, color='white',
                              weight='bold', ha='center', va='top',
                              bbox=dict(boxstyle='round', facecolor='black',
                                      alpha=0.8, edgecolor='cyan', linewidth=2))
        
        # Time display
        time_text = ax.text2D(0.02, 0.97, '', transform=ax.transAxes,
                              fontsize=18, color='yellow', weight='bold',
                              ha='left', va='top',
                              bbox=dict(boxstyle='round', facecolor='black',
                                      alpha=0.9, edgecolor='yellow', linewidth=2))
        
        # Parameters display
        params_text = f"V_max={config.get('V_max', 2e-4):.1e} mg/ml/s | K_m={config.get('K_m', 1.0):.1f} mg/ml"
        params_display = ax.text2D(0.5, 0.02, params_text, transform=ax.transAxes,
                                   fontsize=11, color='lime', ha='center', va='bottom',
                                   bbox=dict(boxstyle='round', facecolor='black',
                                           alpha=0.8, edgecolor='lime', linewidth=1.5))
        
        # Create colorbar
        dummy_surf = ax.plot_surface(np.zeros((2,2)), np.zeros((2,2)), 
                                    np.zeros((2,2)), cmap=colormap,
                                    vmin=np.min(T), vmax=np.max(T))
        cbar = fig.colorbar(dummy_surf, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label('O₂ [mg/ml]', color='white', fontsize=14)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=11)
        plt.setp(cbar.ax.get_yticklabels(), color='white')
        dummy_surf.remove()
        
        def animate(frame_num):
            ax.clear()
            ax.set_facecolor('black')
            
            frame = frame_indices[frame_num]
            
            # Surface plot up to current time
            T_subset = T[:frame+1].T
            t_subset = t[:frame+1]
            T_mesh, X_mesh = np.meshgrid(t_subset, x)
            
            surf = ax.plot_surface(X_mesh, T_mesh, T_subset,
                                  cmap=colormap, alpha=0.85,
                                  linewidth=0, antialiased=True,
                                  vmin=np.min(T), vmax=np.max(T))
            
            # Highlight current profile
            ax.plot(x, [t[frame]]*len(x), T[frame],
                   color='cyan', linewidth=3, alpha=1.0, zorder=10)
            
            ax.set_xlabel('Position [mm]', fontsize=14, color='white', labelpad=12)
            ax.set_ylabel('Time [s]', fontsize=14, color='white', labelpad=12)
            ax.set_zlabel('O₂ [mg/ml]', fontsize=14, color='white', labelpad=12)
            
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, t[-1])
            ax.set_zlim(np.min(T)*0.9, np.max(T)*1.1)
            
            ax.view_init(elev=25, azim=45 + frame_num*0.3)
            
            ax.tick_params(colors='white', labelsize=12)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')
            ax.grid(True, alpha=0.2, color='white', linestyle='--')
            
            time_text.set_text(f't = {t[frame]:.3f} s')
            
            return [surf, title_text, time_text, params_display]
        
        print(f"    Rendering GIF...")
        anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                     interval=1000/config.get('fps', 30), blit=False)
        
        writer = animation.PillowWriter(fps=config.get('fps', 30))
        
        with tqdm(total=n_frames, desc="    Progress", unit="frame",
                 bar_format='{desc}: {percentage:3.00f}%|{bar}| {n:.00f}/{total:.00f} [{elapsed}<{remaining}]') as pbar:
            def progress_callback(current_frame, total_frames):
                pbar.n = current_frame + 1
                pbar.refresh()
            anim.save(filepath, writer=writer, dpi=config.get('dpi', 150),
                     progress_callback=progress_callback)
        
        plt.close(fig)
        print(f"    ✓ Animation saved: {filepath}")
