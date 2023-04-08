import numpy as np


class Simulation:
    def __init__(
        self,
        v_max: float = 5 * 7.5 / 1000,
        road_length: float = 300 * 7.5 / 1000,
        cell_length: float = 7.5 / 1000,
    ) -> None:
        self.cell_length = cell_length
        self.v_max = v_max // self.cell_length
        self.road_length = road_length // self.cell_length

    def run_simulation(self, update_velocity, init_dens: float = 0.1, max_t: float = 150):
        road = np.zeros(self.road_length)
        num_cars = np.round(init_dens * self.road_length)
        road[np.random.choice(self.road_length, size=(num_cars, ), replace=False)] = 1

        velocities = np.random.choice(self.v_max + 1, size=(self.road_length, ), replace=True)
        velocities[road == 0] = -1

        sol = np.zeros((self.road_length, max_t))
        sol[:, 0] = velocities

        for t in range(1, max_t):
            
            sol[:, t] = 

    def update_velocity(vel):

    def apply_velocity(vel):
        for i in range(len(vel) - 1, -1, -1):
            



def main():
    pass


if __name__ == "__main__":
    main()
