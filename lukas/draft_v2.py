# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
import numpy.lib.scimath as sm
from scipy.optimize import fminbound
from scipy.integrate import quad
import OGRePy as gr
import sympy as sp
from scipy.optimize import fsolve

from OGRePy.abc import t, phi, theta
r = gr.sym("r", nonnegative=True)

# Definition of spherical coordinates
Spherical = gr.Coordinates(t, r, theta, phi)

# Class definition for the Metric model
class MetricSystem:
    def __init__(self, f, g, h, param, M=1):
        """
        Initialize the  Metric model with given functions and parameters.

        Parameters:
        - f, g, h: Functions describing the system's behavior.
        - param: Parameters used by the functions f, g, and h.
        """
        # Store parameters and function definitions
        self.param = param
        self.safe_f = f
        self.safe_g = g
        self.safe_h = h

        # Create functions for f, g, h using provided parameters
        self.f = self.safe_f(r, param)
        self.g = self.safe_g(r, param)
        self.h = self.safe_h(r, param)

        #self.M = param[0] # Mass of the black hole
        self.R_s = 2 * M # Schwarzschild radius

        self.z = 1
        self.rsafe=0
        self.drsafe=0

    def metric(self):
        """
        Define the black hole metric in spherical coordinates using the functions f, g, and h.

        Returns:
        - return_metric: A Metric object representing the black hole geometry.
        """
        # Create metric components
        return_metric = gr.Metric(
            coords=Spherical,
            components=gr.diag(-self.f, self.g, self.h, sp.sin(theta) ** 2 * self.h),
            symbol="eta",
        )
        return return_metric

    def __call__(self):
        return self.metric()

    def get_parameters(self, param):
        """
        Return the given parameters.

        Parameters:
        - param: Parameters to be returned.

        Returns:
        - param: The input parameters.
        """
        return param

    def set_parameters(self, param):
        """
        Set the parameters for the system, updating the function definitions.

        Parameters:
        - param: New set of parameters for the system.
        """
        self.param = param
        self.f = self.safe_f(r, param)
        self.g = self.safe_g(r, param)
        self.h = self.safe_h(r, param)

    def V_eff(self, r_val=None, sigma=0, L=1):
        """
        Compute the effective potential for a given radial coordinate 'r'.

        Parameters:
        - r_val: Radial coordinate.
        - sigma: Optional parameter, default is 0, -1 for particle.
        - L: Orbital angular momentum, default is 1.

        Returns:
        - V: Effective potential at the given r.
        """
        # Effective potential formula
        V = - self.f * (sigma - (L ** 2) / self.h)

        # Handle both symbolic and numerical inputs for r_val
        if isinstance(r_val, np.ndarray):
            V_num = sp.lambdify(r, V, "numpy")
            V_return = V_num(r_val)
        elif r_val is not None:
            V_return = V.subs(r, r_val)
        else:
            V_return = V

        return V_return

    def min_max_V_eff(self, sigma=0, L=1, M=1):
        """
        Find the minimum and maximum values of the effective potential V_eff
        within a given radial span.

        Parameters:
        - sigma: Optional parameter for the effective potential (default is 0).
        - L: Orbital angular momentum (default is 1).

        Returns:
        - min: The radial position r_min where the effective potential is minimized.
        - max: The radial position r_max where the effective potential is maximized.
        """
        V = self.V_eff(sigma=sigma, L=L)

        # Calculate the first and second derivatives of the effective potential
        dV_dr_1 = sp.diff(V, r)
        dV_dr_2 = sp.diff(dV_dr_1, r)

        # Solve for critical points (extremes)
        dV_dr_1_simplified = sp.simplify(dV_dr_1)
        extrem = sp.nsolve(dV_dr_1_simplified, r, 2.3)

        # Initialize lists to store minimum and maximum values
        min, max = [], []
        
        if isinstance(extrem, (np.ndarray, list, tuple)):
            for i in range(len(extrem)):
                # Check if the second derivative is positive (min) or negative (max)
                if dV_dr_2.subs(r, extrem[i]) > 0:
                    min.append(float(extrem[i]))
                elif dV_dr_2.subs(r, extrem[i]) < 0:
                    max.append(float(extrem[i]))
        else: 
            #min.append()
            max.append(extrem)
        # Return both the minimum and maximum values
        return min, max
    
    #debuging
    def min_max_V_eff_debug(self, sigma=0, L=1, M=1):
            """
            Find the minimum and maximum values of the effective potential V_eff
            within a given radial span.

            Parameters:
            - sigma: Optional parameter for the effective potential (default is 0).
            - L: Orbital angular momentum (default is 1).

            Returns:
            - min: The radial position r_min where the effective potential is minimized.
            - max: The radial position r_max where the effective potential is maximized.
            """
            V = self.V_eff(sigma=sigma, L=L)

            # Calculate the first and second derivatives of the effective potential
            dV_dr_1 = sp.diff(V, r)
            dV_dr_2 = sp.diff(dV_dr_1, r)

            dV_dr_1_simplified = sp.simplify(dV_dr_1)

            try:
                max = sp.nsolve(dV_dr_1_simplified, r, 2*M+0.1)
            except ValueError as e:
                max=None
            
            try:
                min= sp.nsolve(dV_dr_1_simplified, r, 2*max)
            except ValueError as e:
                min = None


            


            return min, max

    def dphi_dr(self, r_val=None, sigma=0, L=1, E=1):
        """
        Calculate the derivative of the angle phi with respect to the radial distance r.

        Parameters:
        - r: The radial distance at which the derivative is evaluated.
        - sigma: Optional parameter for the effective potential (default is 0).
        - L: Orbital angular momentum (default is 1).
        - E: Energy (default is 1).

        Returns:
        - The derivative of phi with respect to r at the specified radial distance.
        """
        # Formula for the derivative of phi with respect to r
        dphi_dr_sym = (L / self.h) * sp.sqrt(self.f * self.g / (E**2 + self.V_eff(sigma=sigma, L=L)))

        # Handle both symbolic and numerical inputs for r_val
        if isinstance(r_val, np.ndarray):
            dphi_dr_num = sp.lambdify(r, dphi_dr_sym, "numpy")
            dphi_dr_return = dphi_dr_num(r_val)
        elif r_val is not None:
            dphi_dr_return = dphi_dr_sym.subs(r, r_val)
        else:
            dphi_dr_return = sp.simplify(dphi_dr_sym)

        return dphi_dr_return

    def phi(self, r_val, r_span, phi_0=0, sigma=0, L=1, E=1):
        """
        Calculate the angle phi for a list of radial distances r_list by integrating the derivative dphi/dr.

        Parameters:
        - r_list: A list or array of radial distances where phi is to be evaluated.
        - phi_0: The reference radial position (default is 0), where phi(r0) is set to 0.
        - sigma: Optional parameter for the effective potential (default is 0).
        - L: Orbital angular momentum (default is 1).
        - E: Energy (default is 1).

        Returns:
        - An array of phi values corresponding to each radial distance in r_list.
        """
        # Getting the derivative dphi/dr
        d_phi_dr = self.dphi_dr(r_val=None, sigma=sigma, L=L, E=E)

        # Perform symbolic integration
        phi = sp.integrate(d_phi_dr, r)

        if sp.Integral(d_phi_dr, r) == phi:
            # Lambdify the derivative to be used in the solver
            d_phi_dr_func = sp.lambdify(r, d_phi_dr, "numpy")

            # Define the ODE system for numerical integration
            def ode_system(r, phi):
                return d_phi_dr_func(r)

            # Use scipy's solve_ivp to numerically integrate dphi/dr
            phi_list = solve_ivp(ode_system, r_span, [phi_0], t_eval=r_val, method='RK45')
            phi_return = phi_list.y[0]
        else:
            # Return the symbolic result if no numerical integration is needed
            phi_return = phi

        return phi_return 

    def solve_DAE(self, tau, tau_span, r_0, t_0=0, phi_0=0, sigma=0, L=1, E=1, debug=False):

        if abs(E**2) < self.V_eff(r_0, sigma=sigma, L=L): 
            print('E < V_eff(r_0)')
            return
        """
        Solve the system of differential-algebraic equations (DAE) for motion around the black hole.

        Parameters:
        - tau: Time parameter (proper time) for the solution.
        - tau_span: The span of tau for integration.
        - r_0: Initial radial position.
        - t_0: Initial time (default is 0).
        - phi_0: Initial angle (default is 0).
        - sigma: Optional parameter for the effective potential (default is 0).
        - L: Orbital angular momentum (default is 1).
        - E: Energy (default is 1).
        - R_s: Schwarzschild radius (default is 2).

        Returns:
        - result_p: Solution for the forward direction.
        - result_n: Solution for the reverse direction.
        - Falls_in: Boolean flag indicating if the object falls into the black hole.
        """
        # Lambdify the functions f, g, and h for symbolic expressions
        f = sp.lambdify(r, self.f, "numpy")
        g = sp.lambdify(r, self.g, "numpy")
        h = sp.lambdify(r, self.h, "numpy")

        #r_test = sp.solve(self.V_eff(sigma=sigma, L=L) - E ** 2, r)
        Circulare = self.V_eff(L=L, sigma=sigma, r_val=r_0)!= E**2

        self.z = 1
        # Define the system of differential equations for the DAE
        def DAE(tau, y, delta):
            #if Circulare:
            t, r, phi = y
            r = max(r, self.R_s+0.1)

            argument = 1 / g(r) * ((E ** 2) / f(r) + sigma - (L ** 2) / h(r))

            if argument < 0: 
                self.z = -1 * self.z

            drdtau = self.z * delta * np.sqrt(np.abs(argument))   
            #else:
            #    drdtau=0
            #    r= r_0

            dtdtau = delta * E / f(r)
            dphidtau = delta * L / h(r)

            if debug: print(fr'z: {self.z}, tau {tau}, r: {r} and dr/dtau: {drdtau}, argument: {argument}')

            return [dtdtau, drdtau, dphidtau]


        # Initial conditions for the differential equations
        initial_conditions = [t_0, r_0, phi_0]

        # Solve the differential equations for both directions (forward and reverse)
        sol_p = solve_ivp(DAE, tau_span, initial_conditions, t_eval=tau, args=[1], method='DOP853',  max_step=0.05)
        self.z = 1
        sol_n = solve_ivp(DAE, tau_span, initial_conditions, t_eval=tau, args=[-1], method='DOP853',  max_step=0.05)

        Falls_in = False

        # Function to check if the solution falls inside the event horizon
        def Falls_in_BH(arr):
            index = np.where(arr.y[1] < self.R_s)[0]
            if index.size > 0:
                Falls_in = True
                result = arr.y[:, :index[0]]  # Return solution up to event horizon
            else:
                result = arr.y[:]  # Return full solution

            return result
        
        # Get results for both directions
        result_p = Falls_in_BH(sol_p)
        result_n = Falls_in_BH(sol_n)

        return result_p, result_n, Falls_in

# Class for the Black Hole
class BH:
    def __init__(self, M, Metric_sys_list, Metric_Name_list):
        self.M = M
        self.R_s = 2 * M  # Schwarzschild radius (R_s)

        # Lists to store MetricSystem instances and their corresponding names
        self.Metric_sys_list = Metric_sys_list
        self.Metric_Name_list = Metric_Name_list

        # Add MetricSystem instances dynamically based on the provided names
        for Metric_sys, Metric_Name in zip(self.Metric_sys_list, self.Metric_Name_list):
            self.add_Metric_sys(Metric_sys, Metric_Name)

    def add_Metric_sys(self, Metric_sys, Metric_Name):
        """Dynamically adds a new MetricSystem instance to the BH object."""
        # Dynamically set the attribute with the given name
        #Metric_sys.set_parameters([self.M, Metric_sys.set_parameters[1:]])
        setattr(self, Metric_Name, Metric_sys)
      
