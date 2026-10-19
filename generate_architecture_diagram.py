import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    # Setup high-res figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Helper function for drawing rounded boxes
    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, title_color="white", sub_color="#E2E8F0"):
        # Box shadow
        shadow = patches.FancyBboxPatch(
            (x + 0.06, y - 0.06), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor="#CBD5E1", edgecolor="none", alpha=0.5
        )
        ax.add_patch(shadow)

        # Main box
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=bg_color, edgecolor=border_color, linewidth=1.5
        )
        ax.add_patch(box)

        # Title & Subtitle
        ax.text(x + w / 2, y + h * 0.65, title, color=title_color, fontsize=11, fontweight='bold',
                ha='center', va='center', fontfamily='sans-serif')
        ax.text(x + w / 2, y + h * 0.32, subtitle, color=sub_color, fontsize=8.5,
                ha='center', va='center', fontfamily='sans-serif')

    # Draw Title
    ax.text(7, 7.5, "Driver Drowsiness Detection System Architecture", 
            fontsize=17, fontweight='bold', color='#0F172A', ha='center', va='center', fontfamily='sans-serif')
    ax.text(7, 7.15, "End-to-End Pipeline: Video Acquisition -> Computer Vision ROI -> Deep Learning Inference -> Alert Decision", 
            fontsize=9.5, color='#64748B', ha='center', va='center', fontfamily='sans-serif')

    # Block 1: Video Input
    draw_box(0.5, 4.8, 2.2, 1.2, "1. Video Stream", "Webcam / IR Camera\nReal-time RGB Input", "#1E293B", "#334155")

    # Block 2: Face & Eye Detection
    draw_box(3.2, 4.8, 2.3, 1.2, "2. Face & Eye ROI", "OpenCV Haar Cascades\n+ ROI Persistence Memory", "#1D4ED8", "#2563EB")

    # Block 3: Preprocessing
    draw_box(5.9, 4.8, 2.3, 1.2, "3. Preprocessing", "Crop Left & Right Eyes\nResize (64x64) & Normalize", "#0284C7", "#0EA5E9")

    # Block 4: Deep Learning Model
    draw_box(8.6, 4.8, 2.4, 1.2, "4. Deep Classifier", "Custom CNN (111K params)\nor MobileNetV2 (2.4M)", "#4F46E5", "#6366F1")

    # Block 5: Decision / Prediction
    draw_box(11.4, 4.8, 2.1, 1.2, "5. Prediction", "Softmax Probability\n(Open vs Closed)", "#7C3AED", "#8B5CF6")

    # Lower Flow: Branches
    # Safe State (Left branch downwards)
    draw_box(11.4, 2.5, 2.1, 1.1, "Driver Alert", "Reset Clock Timer\nSafe Driver State", "#059669", "#10B981")

    # Timer Accumulator
    draw_box(7.5, 2.5, 2.6, 1.1, "Drowsiness Timer", "Wall-Clock Time Accumulator\nContinuous Closed State", "#D97706", "#F59E0B")

    # Alert System
    draw_box(2.0, 2.3, 3.8, 1.5, "EMERGENCY ALERT TRIGGER", "• Continuous Audio Alarm (winsound buzzer)\n• Visual Warning Banner (Red HUD Alert)\n• Picture-in-Picture Eye View", "#DC2626", "#EF4444")

    # Connecting Arrows
    def draw_arrow(x1, y1, x2, y2, label="", label_pos=(0, 0), color="#475569"):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=15))
        if label:
            lx = (x1 + x2) / 2 + label_pos[0]
            ly = (y1 + y2) / 2 + label_pos[1]
            ax.text(lx, ly, label, fontsize=8.5, fontweight='bold', color=color,
                    ha='center', va='center', bbox=dict(boxstyle="square,pad=0.2", facecolor="#F8FAFC", edgecolor="none"))

    # Horizontal Top Pipeline Arrows
    draw_arrow(2.7, 5.4, 3.2, 5.4)
    draw_arrow(5.5, 5.4, 5.9, 5.4)
    draw_arrow(8.2, 5.4, 8.6, 5.4)
    draw_arrow(11.0, 5.4, 11.4, 5.4)

    # Prediction down branches
    # If Open Eye -> Alert Driver (Reset)
    draw_arrow(12.45, 4.8, 12.45, 3.6, label="Open Eye", label_pos=(0.4, 0), color="#059669")

    # If Closed Eye -> Timer Accumulator
    ax.annotate('', xy=(10.1, 3.05), xytext=(12.45, 4.8),
                arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=2, mutation_scale=15, connectionstyle="arc3,rad=0.2"))
    ax.text(11.6, 3.9, "Closed Eye", fontsize=8.5, fontweight='bold', color="#D97706",
            ha='center', va='center', bbox=dict(boxstyle="square,pad=0.2", facecolor="#F8FAFC", edgecolor="none"))

    # Timer -> Threshold Check -> Alert
    draw_arrow(7.5, 3.05, 5.8, 3.05, label="Duration > 1.5s", label_pos=(0, 0.25), color="#DC2626")

    # Reset loop arrow (Safe driver loops back to Video Stream)
    ax.annotate('', xy=(1.6, 4.8), xytext=(12.45, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.5, ls="--", mutation_scale=12,
                                connectionstyle="arc3,rad=-0.15"))
    ax.text(7.0, 1.5, "Continuous Frame Loop (30-60 FPS)", fontsize=8, color="#64748B",
            ha='center', va='center')

    # Save as JPG and PNG
    plt.tight_layout()
    plt.savefig("system_architecture.jpg", format='jpg', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig("system_architecture.png", format='png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print("Architecture diagram successfully saved as system_architecture.jpg and system_architecture.png")

if __name__ == "__main__":
    create_architecture_diagram()
