import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
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
        
        print(f"{'':>15}Creating 1D animation ({n_frames} frames)...")
        
        # Create figure with black background
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor('black')
        
        # Create axes
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        
        # Style settings
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='white', labelsize=11)
        ax.grid(True, alpha=0.15, color='white', linestyle='--')
        
        # Labels
        ax.set_xlabel('Position [mm]', fontsize=13, color='white')
        ax.set_ylabel('O₂ Concentration [mg/ml]', fontsize=13, color='white')
        
        # Set limits
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(np.min(T)*0.95, np.max(T)*1.05)
        
        # Initialize plot elements
        lines = []
        
        # Main concentration line
        line_main, = ax.plot([], [], 'c-', linewidth=2.5, alpha=1.0, 
                             label='Current', zorder=10)
        lines.append(line_main)
        
        # Trail lines for history
        n_trails = 5
        for i in range(n_trails):
            alpha = 0.3 * (1 - i/n_trails)
            line, = ax.plot([], [], color='cyan', linewidth=1.5, 
                           alpha=alpha, zorder=5-i)
            lines.append(line)
        
        # Gradient fill
        gradient_fill = None
        
        # Title
        title_text = fig.text(0.5, 0.95, config.get('scenario_name', 'Oxygen Diffusion'),
                              fontsize=16, color='white', weight='bold',
                              ha='center', va='top')
        
        # Time display
        time_text = fig.text(0.05, 0.92, '', fontsize=14, color='yellow',
                             weight='bold', ha='left', va='top')
        
        # Method display
        method_text = fig.text(0.05, 0.05, f"Method: {result['method']}", 
                               fontsize=11, color='lime', ha='left', va='bottom')
        
        # Parameters display
        params_text = f"V_max={config.get('V_max', 2e-4):.1e} mg/ml/s | K_m={config.get('K_m', 1.0):.1f} mg/ml | D_mb={config.get('D_mb', 3e-8):.1e} cm²/s"
        params_display = fig.text(0.95, 0.05, params_text, fontsize=10,
                                  color='cyan', ha='right', va='bottom')
        
        # Progress bar background
        progress_bg = Rectangle((0.35, 0.02), 0.3, 0.015, transform=fig.transFigure,
                                facecolor='gray', alpha=0.3, edgecolor='white', linewidth=1)
        fig.patches.append(progress_bg)
        
        # Progress bar
        progress_bar = Rectangle((0.35, 0.02), 0, 0.015, transform=fig.transFigure,
                                 facecolor='lime', alpha=0.8)
        fig.patches.append(progress_bar)
        
        def animate(frame_num):
            frame = frame_indices[frame_num]
            
            # Update main line
            lines[0].set_data(x, T[frame])
            
            # Update trail lines
            for i in range(1, min(n_trails+1, len(lines))):
                trail_idx = max(0, frame - i*10)
                if trail_idx < len(T):
                    lines[i].set_data(x, T[trail_idx])
            
            # Update gradient fill
            nonlocal gradient_fill
            if gradient_fill:
                gradient_fill.remove()
            gradient_fill = ax.fill_between(x, 0, T[frame], 
                                           color='cyan', alpha=0.1)
            
            # Add glow effect to boundaries
            ax.axvline(x[0], color='red', alpha=0.3, linewidth=3, linestyle='-')
            ax.axvline(x[-1], color='blue', alpha=0.3, linewidth=3, linestyle='-')
            
            # Update time text
            time_text.set_text(f't = {t[frame]:.4f} s')
            
            # Update progress bar
            progress = frame_num / n_frames
            progress_bar.set_width(0.3 * progress)
            
            # Add concentration indicators
            max_idx = np.argmax(T[frame])
            min_idx = np.argmin(T[frame])
            
            # Highlight max/min points
            ax.plot(x[max_idx], T[frame, max_idx], 'ro', markersize=8, 
                   alpha=0.8, markeredgecolor='white', markeredgewidth=1)
            ax.plot(x[min_idx], T[frame, min_idx], 'bo', markersize=8,
                   alpha=0.8, markeredgecolor='white', markeredgewidth=1)
            
            return lines + [gradient_fill, time_text, title_text, method_text, 
                          params_display, progress_bar]
        
        print(f"{'':>10}Rendering GIF...")
        anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                     interval=1000/config.get('fps', 30), blit=False)
        
        writer = animation.PillowWriter(fps=config.get('fps', 30))
        
        with tqdm(total=n_frames, desc=" "*10 + "Progress", unit="frame",
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}]') as pbar:
            def progress_callback(current_frame, total_frames):
                pbar.n = current_frame + 1
                pbar.refresh()
            anim.save(filepath, writer=writer, dpi=config.get('dpi', 150),
                     progress_callback=progress_callback)
        
        plt.close(fig)
        print(f"{'':>10}✓ Animation saved: {filepath}")
