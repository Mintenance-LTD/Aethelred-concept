"""2D battlefield visualization using matplotlib."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from aethelred.config.settings import RenderConfig
from aethelred.core.enums import DroneRole
from aethelred.core.models import BattlefieldState


class BattlefieldRenderer:
    """2D visualization of the battlefield."""

    ROLE_COLORS: ClassVar[dict[DroneRole, str]] = {
        DroneRole.RECON: "#00BFFF",     # deep sky blue
        DroneRole.ENGAGE: "#FF4444",     # red
        DroneRole.EW: "#FFD700",         # gold
        DroneRole.RELAY: "#00FF7F",      # spring green
        DroneRole.MOTHER: "#FFFFFF",     # white
    }

    def __init__(self, config: RenderConfig) -> None:
        self.config = config
        self._fig: Any = None
        self._ax: Any = None
        self._initialized = False

    def render(
        self,
        state: BattlefieldState,
        terrain: np.ndarray | None = None,
        mode: str = "human",
    ) -> np.ndarray | None:
        """Render the current battlefield state."""
        import matplotlib
        if mode != "human":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._initialized:
            self._fig, self._ax = plt.subplots(1, 1, figsize=(8, 8))
            self._initialized = True
            if mode == "human":
                plt.ion()

        ax = self._ax
        ax.clear()

        # Draw terrain background
        if terrain is not None:
            ax.imshow(
                terrain,
                extent=[0, 1000, 0, 1000],
                origin="lower",
                cmap="terrain",
                alpha=0.4,
            )

        # Draw threats
        for threat in state.threats:
            if not threat.is_active:
                continue
            marker = "^" if threat.category.value == "kinetic" else "s"
            ax.plot(
                threat.position.x, threat.position.y,
                marker=marker, color="#FF0000", markersize=10,
                markeredgecolor="white", markeredgewidth=0.5,
            )
            # Threat range circle
            if self.config.show_ranges:
                circle = plt.Circle(
                    (threat.position.x, threat.position.y),
                    threat.estimated_range,
                    color="red", fill=False, alpha=0.2, linestyle="--",
                )
                ax.add_patch(circle)

        # Draw friendly units
        for drone in state.friendly_units:
            if not drone.is_alive():
                # Draw destroyed marker
                ax.plot(
                    drone.position.x, drone.position.y,
                    marker="x", color="gray", markersize=6, alpha=0.5,
                )
                continue

            color = self.ROLE_COLORS.get(drone.role, "#AAAAAA")
            size = 12 if drone.role == DroneRole.MOTHER else 8

            ax.plot(
                drone.position.x, drone.position.y,
                marker="o", color=color, markersize=size,
                markeredgecolor="white", markeredgewidth=0.5,
            )

            # Health bar
            bar_w = 12
            bar_x = drone.position.x - bar_w / 2
            bar_y = drone.position.y + 12
            ax.barh(
                bar_y, drone.health * bar_w, height=3,
                left=bar_x, color="lime" if drone.health > 0.5 else "red",
                alpha=0.7,
            )

            # Heading indicator
            import math
            dx = math.cos(drone.heading) * 15
            dy = math.sin(drone.heading) * 15
            ax.arrow(
                drone.position.x, drone.position.y, dx, dy,
                head_width=4, head_length=3, fc=color, ec=color, alpha=0.6,
            )

        # Draw objectives
        for obj in state.objectives:
            color = "#00FF00" if obj.is_completed else "#FFFF00"
            ax.plot(
                obj.position.x, obj.position.y,
                marker="*", color=color, markersize=15,
                markeredgecolor="white",
            )

        # Info panel
        num_alive = len([d for d in state.friendly_units if d.is_alive()])
        num_total = len(state.friendly_units)
        num_threats = len([t for t in state.threats if t.is_active])
        info_text = (
            f"Step: {state.timestep}  |  "
            f"Swarm: {num_alive}/{num_total}  |  "
            f"Threats: {num_threats}  |  "
            f"Comms: {1.0 - state.global_comms_degradation:.0%}"
        )
        ax.set_title(info_text, fontsize=10, color="white", pad=10)

        # Style
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_facecolor("#1a1a2e")
        self._fig.patch.set_facecolor("#0f0f1a")
        ax.set_aspect("equal")

        if self.config.show_grid:
            ax.grid(True, alpha=0.1, color="white")

        # Legend
        legend_elements = []
        import matplotlib.lines as mlines
        for role, color in self.ROLE_COLORS.items():
            if role == DroneRole.MOTHER:
                continue
            legend_elements.append(
                mlines.Line2D([], [], color=color, marker="o", linestyle="None",
                              markersize=6, label=role.value)
            )
        legend_elements.append(
            mlines.Line2D([], [], color="red", marker="^", linestyle="None",
                          markersize=6, label="threat")
        )
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
                  facecolor="#1a1a2e", edgecolor="gray", labelcolor="white")

        if mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            import matplotlib.pyplot as plt
            plt.pause(0.01)
            return None
        else:
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._initialized = False
