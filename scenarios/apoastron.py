import rebound
import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the apoastron of the system's orbit."""
        final_answer_units = "m"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=True, return_empirical=False) -> float:
        """
        Return the true answer for the environment.
        
        Args:
            N_obs: Number of observations to use (if None, use all)
            verification: Whether to verify values match
            return_empirical: If True, return the empirically derived value;
                              if False, return the value inputted into the simulation or using Rebound simulated details typically hidden
        """
        # Load the simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)
        # Calculate masses using task_utils
        m1, m2 = task_utils.star_masses(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        # Calculate semi-major axis and eccentricity using task_utils
        a, _, _ = task_utils.calculate_semi_major_axes(df, m1, m2, self.binary_sim, verification=verification, return_empirical=return_empirical)
        e = task_utils.calculate_eccentricity(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        # Calculate apoapsis
        apoapsis = a * (1 + e)
        

        # Rebound verification
        sim = rebound.Simulation()
        sim.units = self.binary_sim.units
        sim.add(m=self.binary_sim.star1_mass, x=self.binary_sim.star1_pos[0], y=self.binary_sim.star1_pos[1], z=self.binary_sim.star1_pos[2], 
                vx=self.binary_sim.star1_momentum[0] / self.binary_sim.star1_mass, vy=self.binary_sim.star1_momentum[1] / self.binary_sim.star1_mass, vz=self.binary_sim.star1_momentum[2] / self.binary_sim.star1_mass)
        sim.add(m=self.binary_sim.star2_mass, x=self.binary_sim.star2_pos[0], y=self.binary_sim.star2_pos[1], z=self.binary_sim.star2_pos[2], 
                vx=self.binary_sim.star2_momentum[0] / self.binary_sim.star2_mass, vy=self.binary_sim.star2_momentum[1] / self.binary_sim.star2_mass, vz=self.binary_sim.star2_momentum[2] / self.binary_sim.star2_mass)
        orb = sim.particles[1].orbit(primary=sim.particles[0])
        apoapsis_rebound = orb.a * (1 + orb.e)
        if verification:
            assert abs(apoapsis - apoapsis_rebound) < 0.02 * apoapsis_rebound, f"{apoapsis} and {apoapsis_rebound} are not within 2% of each other"
        
        if return_empirical:
            return apoapsis  # Return the calculated apoapsis if empirical value is requested
        else:
            return apoapsis_rebound  # Return the rebound calculated apoapsis if not requesting empirical value

