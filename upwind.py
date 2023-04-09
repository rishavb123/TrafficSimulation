import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")


class Simulation:
    def __init__(
        self,
        x_space: np.array = np.arange(-5, 5, 0.01),
        t_space: np.array = np.arange(0, 120, 0.1),
        rho_max: float = 160,
        v_max: float = 120 / 3600,
    ) -> None:
        self.x_space = x_space
        self.t_space = t_space

        self.rho_max = rho_max
        self.v_max = v_max

        self.s = Simulation.compute_differences(self.x_space)
        self.h = Simulation.compute_differences(self.t_space)

    def run_simulation(self, init_rho: np.array, update_step, boundary_value):
        sol = np.empty((*self.x_space.shape, *self.t_space.shape))
        sol[:, 0] = init_rho
        for j in range(1, len(self.t_space)):
            update_step(sol, j)
            sol[sol < 0] = 0
            sol[sol > self.rho_max] = self.rho_max
            boundary_value(sol, j)
        return sol

    def plot_density(self, rho, t=None, fig_ax=None):
        if fig_ax is None:
            fig_ax = plt.subplots()
        _, ax = fig_ax
        title = f"Density vs x{f' (t={t})' if t is not None else ''}"
        ax.set_title(title)
        ax.plot(self.x_space, rho)

    def plot_density_heatmap(self, sol, fig_ax=None):
        if fig_ax is None:
            fig_ax = plt.subplots()
        _, ax = fig_ax
        title = f"Density over space and time"
        ax.set_title(title)
        ax.imshow(sol.T, cmap="gray", vmin=0, vmax=self.rho_max)
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")

    def plot_density_animation(self, sol, fig_ax=None, log_t=False):
        if fig_ax is None:
            fig_ax = plt.subplots()
        fig, ax = fig_ax

        (ln,) = ax.plot(self.x_space, sol[:, 0], label="rho")
        (ln2,) = ax.plot(self.x_space, self.compute_f(sol[:, 0]), label="f")
        ax.legend()

        def init():
            ax.set_xlim(self.x_space[0], self.x_space[-1])
            ax.set_ylim(-20, self.rho_max + 20)
            ax.set_title("Density / Flow vs x")
            ax.set_xlabel("x")
            ax.set_ylabel("rho / f")
            return (ln, ln2)

        def update(frame_t):
            ln.set_ydata(sol[:, frame_t])
            ln2.set_ydata(self.compute_f(sol[:, frame_t]))
            if log_t:
                print(f"t={np.sum(self.h[:frame_t + 1]):.1f} s {' ' * 40}", end="\r")
            return (ln, ln2)

        anim = FuncAnimation(
            fig,
            update,
            frames=range(len(self.t_space)),
            init_func=init,
            blit=True,
            interval=self.h[0] * 1000,
        )

        return anim

    def update_textbook(self, sol, j):
        rho = sol[:, j - 1]
        f = self.compute_f(rho)
        sol[:-1, j] = rho[:-1] - self.h[j - 1] / self.s * (f[1:] - f[:-1])

    def update_lax_friedrichs(self, sol, j):
        rho = sol[:, j - 1]
        f = self.compute_f(rho)
        sol[1:-1, j] = (
            (rho[2:] + rho[:-2]) / 2 
            - self.h[j - 1] / self.s[1:] * (
                f[2:] - f[:-2]
            ) / 2
        )

    def right_circular_boundary_value(self, sol, j):
        sol[-1, j] = sol[0, j]

    def circular_boundary_values(self, sol, j):
        sol[-1, j] = sol[1, j]
        sol[0, j] = sol[-2, j]
    
    def compute_f(self, rho: np.array) -> np.array:
        f = self.v_max * rho * (1 - rho / self.rho_max)
        return f

    @staticmethod
    def compute_differences(space: np.array) -> np.array:
        diff = space[1:] - space[:-1]
        return diff


def main():
    s = Simulation()

    x = s.x_space

    init_rho = np.zeros_like(x)
    init_rho[x < -0.25] = 80
    init_rho[(x >= -0.25) & (x < 0)] = 160


    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    sol = s.run_simulation(init_rho, s.update_textbook, s.right_circular_boundary_value)
    s.plot_density_heatmap(sol, fig_ax=(fig, axes[0][1]))
    s.plot_density_animation(sol, fig_ax=(fig, axes[0][0]))

    sol = s.run_simulation(init_rho, s.update_lax_friedrichs, s.circular_boundary_values)
    s.plot_density_heatmap(sol, fig_ax=(fig, axes[1][1]))
    s.plot_density_animation(sol, fig_ax=(fig, axes[1][0]), log_t=True)

    plt.show()


if __name__ == "__main__":
    main()
