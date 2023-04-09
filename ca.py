import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Simulation:
    def __init__(
        self,
        v_max: int = 5,
        road_length: int = 300,
        t_max: int = 150,
        include_randomization: bool = False,
    ) -> None:
        self.v_max = v_max
        self.road_length = road_length
        self.t_max = t_max
        self.include_randomization = include_randomization

    def run_simulation(self, init_dens: float = 0.1):
        road = np.zeros(self.road_length)
        num_cars = int(np.round(init_dens * self.road_length))
        road[np.random.choice(self.road_length, size=(num_cars, ), replace=False)] = 1

        velocities = np.random.choice(self.v_max + 1, size=(self.road_length, ), replace=True)
        velocities[road == 0] = -1

        sol = np.zeros((self.road_length, self.t_max))
        sol[:, 0] = velocities
        
        operations = [self.accelerate, self.decelerate]
        
        if self.include_randomization:
            operations.append(self.randomize)
            
        operations.append(self.move)

        for t in range(1, self.t_max):
            sol[:, t] = self.apply(operations, sol[:, t - 1])
            
        return sol
        
    def apply(self, operations, inp):
        z = inp
        for op in operations:
            z = op(z)
        return z
            
    def accelerate(self, vel):
        vel = vel + 1
        vel[vel == 0] = -1
        vel[vel > self.v_max] = self.v_max
        return vel

    def decelerate(self, vel):
        next_idx = -np.ones_like(vel)
        last_idx = -1
        first_idx = -1
        for i in range(self.road_length - 1, -1, -1):
            if vel[i] >= 0:
                if first_idx == -1:
                    first_idx = i
                next_idx[i] = last_idx
                last_idx = i
        next_idx[first_idx] = self.road_length + last_idx       
        
        for i in range(self.road_length):
            if vel[i] >= 0:
                vel[i] = min(next_idx[i] - i - 1, vel[i])
        return vel
        
    def randomize(self, vel, p=0.2):
        vel[np.random.random(size=(self.road_length)) < p] -= 1
        return vel

    def move(self, vel):
        next_vel = -np.ones_like(vel)
        for i in range(self.road_length):
            if vel[i] >= 0:
                next_vel[int((i + vel[i]) % self.road_length)] = vel[i]
        return next_vel
        
    def plot_velocity_heatmap(self, sol, fig_ax=None, cmap="gray"):
        if fig_ax is None:
            fig_ax = plt.subplots()
        _, ax = fig_ax
        title = f"Cars over space and time"
        ax.set_title(title)
        ax.imshow(sol.T, cmap=cmap, vmin=-1, vmax=self.v_max)
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")
                
    def plot_velocity_animation(self, sol, fig_ax=None, log_t=False):
        if fig_ax is None:
            fig_ax = plt.subplots()
        fig, ax = fig_ax

        (ln,) = ax.plot(np.arange(self.road_length) , sol[:, 0], label="rho")

        ax.legend()

        def init():
            ax.set_xlim(0, self.road_length)
            ax.set_ylim(-2, self.v_max + 1)
            ax.set_title("Velocity vs x")
            ax.set_xlabel("x")
            ax.set_ylabel("velocity")
            return (ln, )

        def update(frame_t):
            ln.set_ydata(sol[:, frame_t])
            if log_t:
                print(f"t={np.sum(self.h[:frame_t + 1]):.1f} s {' ' * 40}", end="\r")
            return (ln, )

        anim = FuncAnimation(
            fig,
            update,
            frames=range(self.t_max),
            init_func=init,
            blit=True,
        )

        return anim



def main():
    s = Simulation(v_max=5, road_length=300, t_max=100, include_randomization=True)
    sol = s.run_simulation(init_dens=0.1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    s.plot_velocity_heatmap(sol, fig_ax=(fig, axes[0]))
    s.plot_velocity_animation(sol, fig_ax=(fig, axes[1]))
    
    plt.show()


if __name__ == "__main__":
    main()
