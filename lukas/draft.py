# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import numpy.lib.scimath as sm
import threading
import time

# Class definition for the black hole model
class BH:
    def __init__(self, f, g, h, param):
        """
        Initialize the black hole model with given functions and parameters.

        Parameters:
        - f, g, h: Functions describing the system's behavior.
        - param: Parameters used by the functions f, g, and h.
        """
        # Store parameters and function definitions
        self.param = param
        self.safe_f = f
        self.safe_g = g
        self.safe_h = h

        # Create lambda functions for f, g, h using provided parameters
        self.f = lambda r: self.safe_f(r, param)
        self.g = lambda r: self.safe_g(r, param)
        self.h = lambda r: self.safe_h(r, param)

    def get_paramters(self, param):
        """
        Return the given parameters.
        """
        return param 

    def set_paramters(self, param):
        """
        Set the parameters for the system, updating the function definitions.

        Parameters:
        - param: New set of parameters for the system.
        """
        self.param = param

        self.f = lambda r: self.safe_f(r, param)
        self.g = lambda r: self.safe_g(r, param)
        self.h = lambda r: self.safe_h(r, param)

    # Effective Potential
    def V_eff(self, r, sigma=0, L=1):
        """
        Compute the effective potential for a given radial coordinate 'r'.

        Parameters:
        - r: Radial coordinate.
        - sigma: Optional parameter, default is 0, -1 for partical.
        - L: Orbital angular momentum, default is 1.
        
        Returns:
        - V: Effective potential at the given r.
        """
        V = self.f(r) * (sigma - (L ** 2) / self.h(r))
        return V

    # Equations of Motion (Solve Differential-Algebraic Equations)
    def solve_DAE(self, tau, tau_span, r_0, t_0=0, phi_0=0, sigma=0, L=1, E=1, R_s=2):
        """
        Solve the system of differential-algebraic equations (DAE) for motion around the black hole.

        Parameters:
        - tau: Time steps at which the solution is evaluated.
        - tau_span: Tuple containing the time span for the integration.
        - r_0: Initial radial position.
        - t_0: Initial time, default is 0.
        - phi_0: Initial angular position, default is 0.
        - sigma, L, E, R_s: Physical parameters used in the equations.
        
        Returns:
        - result_p: Solution for positive time direction.
        - result_n: Solution for negative time direction.
        - Falls_in: Boolean indicating if the solution falls inside the event horizon.
        """
        # Define the system of differential equations
        def DAE(tau, y, delta):
            t, r, phi = y
            """
            Delta is used to account for the fact that the sqrt of Rr dot can be positive or negative.
            Considering that an object can only escape from orbit if E < 0, delta also in other equations to make them fit my approch.
            """
            
            # Define the differential equations for t, r, and phi
            dtdtau    = -delta * E / self.f(r)
            dphidtau  = -delta * L / self.h(r)
            argument  = 1 / self.g(r) * ((E ** 2) / self.f(r) + sigma - (L ** 2) / self.h(r))

            # Check if the argument for radial motion is positive or negative
            if argument >= 0:
                drdtau = delta * np.sqrt(argument)
            else:
                drdtau = delta * np.emath.sqrt(argument)  # Uses complex numbers if argument is negative
                print("Warning: E < V_eff at r", r)

            return [dtdtau, drdtau, dphidtau]
        
        # Initial conditions: time, radial position, and angular position
        initial_conditions = [t_0, r_0, phi_0]

        # Solve the differential equations for both forward and backward time directions
        sol_p = solve_ivp(DAE, tau_span, initial_conditions, t_eval=tau, args=[1], method='RK45')
        sol_n = solve_ivp(DAE, tau_span, initial_conditions, t_eval=tau, args=[-1], method='RK45')

        Falls_in = False

        # Function to check if the solution falls inside the event horizon
        def Falls_in_BH(arr):
            # Find the index where the radial coordinate goes below the Schwarzschild radius
            index = np.argmax(arr.y[1] < R_s)

            # Check if the radial coordinate falls inside the black hole
            if sol_n.y[1][index] < R_s:
                result = arr.y[:, :index]  # Return the solution up to the event horizon
                Falls_in = True
            else:
                result = arr.y[:]  # Return the full solution

            return result

        # Get the solutions for both directions
        result_p = Falls_in_BH(sol_p)
        result_n = Falls_in_BH(sol_n)

        return result_p, result_n, Falls_in

    '''
    From here on, the code is not really finished.
    '''

    # Light deflection (gravitational lensing)
    def alpha(self, r_0=1, n=1):
        """
        Calculate the deflection angle for light passing near the black hole.

        Parameters:
        - r_0: Reference radial distance, default is 1.
        - n: Multiplicative factor, default is 1.
        
        Returns:
        - alpha: The deflection angle.
        """
        # Define the integrand for the deflection angle
        def integrand(r):
            return 1 / np.sqrt(self.h(r) / self.g(r) * (self.h(r) / self.h(r_0) * self.f(r_0) / self.f(r) - 1))

        # Perform the integration
        alpha, error = integrate.quad(integrand, r_0, float('inf'), epsabs=1e-12, epsrel=1e-12)

        return 2 * alpha - n * np.pi

    # Perihelion shift
    def D_Phi(self, r1, r2, L=0, E=1):
        """
        Calculate the perihelion shift between two radial positions.

        Parameters:
        - r1, r2: The radial positions between which the perihelion shift is calculated.
        - L: Orbital angular momentum, default is 0.
        - E: Energy, default is 1.
        
        Returns:
        - Phi: The perihelion shift.
        """
        # Define the integrand for the perihelion shift
        def integrand(r):
            return 1 / np.sqrt(((self.E(r, L=L)) ** 2 + self.V_eff(r, L=L)) / (self.f(r) * self.g(r)))

        # Perform the integration to calculate the perihelion shift
        Phi, error = integrate.quad(integrand, r1, r2, epsabs=1e-12, epsrel=1e-12)

        return Phi