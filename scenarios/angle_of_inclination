import rebound
import numpy as np
import pandas as pd
import scripts.task_utils as task_utils

#Angle of inclination, as included in rebound, calculataed as the angle between orbital plane and xy-reference plane measured from the positive x-axis
#Angle are in radians
#Assumes that inclination is constant throughout

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the angle of inclination of system's orbit."""
        final_answer_units = "rad"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=False, return_empirical=False) -> float:
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

        # Calculate the unit vector of the orbital plane with angular momentum vector
        # Calculate relative positions
        df['rel_x'] = df['star2_x'] - df['star1_x']
        df['rel_y'] = df['star2_y'] - df['star1_y']
        df['rel_z'] = df['star2_z'] - df['star1_z']
        
        # Calculate relative velocities using task_utils
        _, _, _, star2_vx, star2_vy, star2_vz = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        star1_vx, star1_vy, star1_vz, _, _, _ = task_utils.calculate_velocities(df, self.binary_sim, verification=verification, return_empirical=return_empirical)
        
        df['rel_vx'] = star2_vx - star1_vx
        df['rel_vy'] = star2_vy - star1_vy
        df['rel_vz'] = star2_vz - star1_vz
        
        # Compute the specific angular momentum components
        h_x = (df['rel_y'] * df['rel_vz'] - df['rel_z'] * df['rel_vy']).mean()
        h_y  = (df['rel_z'] * df['rel_vx'] - df['rel_x'] * df['rel_vz']).mean()
        h_z = (df['rel_x'] * df['rel_vy'] - df['rel_y'] * df['rel_vx']).mean()
        h_magnitude = np.sqrt(h_x**2 + h_y**2 + h_z**2)
        h = np.array([h_x, h_y, h_z]) / h_magnitude

        # Calculate the inclination angle
        empirical_inc = np.arctan2(h[2], h[1]) % (2 * np.pi) # Ensures radian is positive, measured from the positive x-axis


        # verification, inclinaiton is usually constant throughout the simulation, so we can use the first value
        inc_rebound = df['inclination'].iloc[0]
        if verification:
            assert abs(empirical_inc - inc_rebound) < 0.01 * inc_rebound, f"{empirical_inc} and {inc_rebound} are not within 1% of each other"
        
        if return_empirical:
            return empirical_inc  # Return the calculated inclination if empirical value is requested
        else:
            return inc_rebound  # Return the rebound calculated inclination if not requesting empirical value



#        # Projection approach if there is no rotation of orbital plane about the z axis
#        df['empirical_inc'] = np.arctan2(df['star1_z'] - df['star2_z'], df['star1_x'] - df['star2_x'])
#
#        #Ensures radian is positive, measured from the positive x-axis
#        df['empirical_inc'] = df['empirical_inc'] % (2 * np.pi)
#        mean_empirical_inc = df['empirical_inc'].mean()
#        
