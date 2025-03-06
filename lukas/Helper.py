import numpy as np
import sympy as sp

from IPython.display import display, Math

from itertools import product
import matplotlib.pyplot as plt


class SymbolWithValue(sp.Symbol):
    """
    A Symbol subclass that extends sympy.Symbol by allowing it to store a default value.

    Parameters:
    - name: The name of the symbol.
    - default_value: An optional default value for the symbol. Defaults to None.
    - kwargs: Any additional keyword arguments passed to sympy.Symbol constructor.
    """
    
    def __new__(cls, name, default_value=None, **kwargs):
        """
        Create a new SymbolWithValue object, which is a subclass of sympy.Symbol.
        
        This method ensures that the symbol is created and that a default value is attached if provided.
        
        Parameters:
        - name (str): The name of the symbol.
        - default_value (optional): The default value to associate with the symbol.
        - kwargs: Additional keyword arguments for sympy.Symbol constructor.
        
        Returns:
        - SymbolWithValue instance: A new instance of the class with the default value attached.
        """
        # Create the Symbol object using sympy's __new__ method
        obj = sp.Symbol.__new__(cls, name, **kwargs)
        
        # Attach the default_value if provided
        obj.value = default_value
        
        return obj


class SymbolicFunction:
    """
    A class to represent a symbolic function where the variables are either sympy.Symbol or instances of SymbolWithValue.
    It allows evaluation of the function both symbolically and numerically.
    """

    def __init__(self, function, variables):
        """
        Initializes the symbolic function.
        
        Parameters:
        - function: A sympy function (e.g., a sympy expression).
        - variables: A list of variables (either sympy.Symbol or SymbolWithValue).
        """
        self.function = function
        self.variables = variables

    def symbolic(self):
        """
        Returns the symbolic function.
        """
        return self.function
    
    def values(self):
        """
        Evaluates the function by substituting the variable values into the symbolic expression.
        Any variables with None as value are excluded from the evaluation.
        
        Returns:
        - The numerical result of the function with variables substituted (excluding unassigned ones).
        """
        subs = {}
        for var in self.variables:
            if isinstance(var, SymbolWithValue) and var.value is not None:
                subs[var] = var.value  # Use the value of SymbolWithValue if defined
            elif isinstance(var, sp.Symbol):
                subs[var] = var  # Keep sympy.Symbols as they are (for symbolic calculations)
        
        # Substitute the values into the function and return the result
        return self.function.subs(subs)
    
    def __call__(self):
        """
        Allows the symbolic function to be called directly as a function, 
        returning the symbolic expression when called.
        """
        return self.function
    
r = sp.Symbol('r')  # Define the symbol 'r'

class SymbolicFunction2(sp.Function):
    # This class defines a custom symbolic function, inheriting from sympy.Function.
    @classmethod
    def __new__(cls, expr, func, variables=[], dependence=[r], name='f', *args):
        # The function accepts an expression (e.g., x**2 + y) and returns a functional expression.
        obj = sp.Function.__new__(sp.Function, name)  # Create a new symbolic function object

        subs = {}  # Dictionary to hold substitution values for variables
        
        # Loop over the given variables and prepare the substitutions
        for var in variables:
            if isinstance(var, SymbolWithValue) and var.value is not None:
                # If the variable is an instance of SymbolWithValue and has a defined value,
                # substitute the value into the function.
                subs[var] = var.value
            elif isinstance(var, sp.Symbol):
                # If the variable is a sympy.Symbol, keep it for symbolic calculations.
                subs[var] = var

        # Substitute the values into the function expression and return the result
        safe = obj  # Save the reference to the original function object for later use
        obj = sp.Lambda(dependence, func.subs(subs))  # Create a Lambda function with the substituted expression
        obj.sym = sp.simplify(sp.Lambda(dependence, func).args[1])  # Simplify the symbolic function (no substitutions)
        obj.num = sp.lambdify(dependence, sp.Lambda(dependence, func.subs(subs)).args[1], "numpy")  # Create a numerical function for evaluation with numpy
        obj.f = safe  # Store the original function reference
        
        return obj  # Return the customized function object

# Function to display mathematical expressions in LaTeX format
def printTeX(argument, Text=''):
    # Display the argument with the provided text, formatted using LaTeX
    display(Math(fr'\begin{{align}} {Text} {sp.latex(argument)} \end{{align}}'))
    return

# Function to create an RGB image from a list of values
def Color_img(v_list):
    # Create an empty RGB image with dimensions (len(v_list), len(v_list), 3)
    # The third dimension is 3 because we're creating an RGB image (3 color channels: R, G, B)
    rgb_image = np.zeros((len(v_list), len(v_list), 3))  
    
    # Set the red and green channels to zero (no red and no green)
    rgb_image[:, :, 0] = 0  # Red channel
    rgb_image[:, :, 1] = 0  # Green channel 
    
    # Set the blue channel to the values from v_list
    # This means the image will have varying shades of blue based on v_list
    rgb_image[:, :, 2] = v_list 
    
    # Return the generated RGB image and the first row of the image (rgb_image[0])
    return rgb_image, rgb_image[0]

# Function to display a color spectrum on the given axis (ax)
def color_spec(ax, y=0.1, rgb_image=None, v_list=[0], grad=None):
    # If no RGB image is provided, generate one using the provided v_list
    if rgb_image is None: 
        rgb_image = Color_img(v_list)[0]
    
    # If grad is specified (gradient value), create a gradient from 0 to 1 and generate the RGB image
    if grad is not None: 
        rgb_image = Color_img(np.linspace(0, 1, grad))[0] 
    
    # Display the RGB image on the axis ax using imshow
    ax.imshow(rgb_image)

    # Set the y-axis limits (this controls the height of the color spectrum)
    ax.set_ylim(0, len(rgb_image[0]) * y)
    
    # If the RGB image has more than 1 column (if there's more than one color)
    if len(rgb_image[0]) > 1:
        # If there are fewer than 10 columns, set the x-ticks based on the number of columns
        if len(rgb_image[0]) - 1 < 10: 
            ax.set_xticks(np.linspace(0, len(rgb_image[0]) - 1, len(rgb_image[0])), 
                          np.round(np.linspace(0, 1, len(rgb_image[0])), 2))
        # If there are more than 10 columns, set 10 x-ticks equally spaced and labeled between 0 and 1
        else: 
            ax.set_xticks(np.linspace(0, len(rgb_image[0]) - 1, 10), 
                          np.round(np.linspace(0, 1, 10), 2))
    # If the image only has one column, set x-ticks at 0
    else: 
        ax.set_xticks([0, 0])  # Only show one tick (0) for a single column image
    
    # Remove y-axis ticks because they're not necessary for the color spectrum
    ax.set_yticks([])

    # Set the label for the x-axis
    ax.set_xlabel('$Q_b$')  # Use LaTeX formatting for the label

    # Set the title of the plot
    ax.set_title('$Q_b$ colourspecktrum')  # Again, using LaTeX for formatting
    
    # Return the axis with the color spectrum
    return ax


# Function to plot the effective potential V_eff on a given axis
def Veff_graph(ax, Metric, param, sigma=-1, L=1, M=1, r_int=[2, 10], bool_legend=True):
    # Generate a list of r values (radius) from r_int[0]*M to r_int[1]*M, with 10 points per unit
    r_list = np.linspace(r_int[0] * M, r_int[1] * M, r_int[1] * 10)
    
    V_eff_list = []  # List to store the computed V_eff values

    # Compute V_eff for each set of parameters and store the results
    for i in range(len(param)):
        # Set the parameters for the Metric object
        Metric.set_parameters(param[i])
        # Compute V_eff for the current r range and append it to the list
        V_eff_list.append(Metric.V_eff(r_list, sigma=sigma, L=L))

    # Plot the V_eff for each set of parameters
    for i in range(len(param)):
        # Plot V_eff for the current set of parameters
        if bool_legend: 
            ax.plot(r_list, V_eff_list[i], color=(0, 0, param[i][0]), label=fr'$V_{{eff}}$ with Q_b={param[i][0]}')
        else: 
            ax.plot(r_list, V_eff_list[i], color=(0, 0, param[i][0]))
    
    # Set the title of the plot and the axis labels
    ax.set_title(fr'$V_{{eff}}(r)$  with $\sigma={sigma}$, $M={M}$ and $L={L}$')
    ax.set_ylabel('$V_eff$')  # Y-axis label
    ax.set_xlabel('r [$R_s$]')  # X-axis label
    ax.grid()  # Show the grid on the plot
    
    return ax  # Return the axis with the plot

# Function to plot the maxima and minima of the effective potential V_eff
def MaxMin(ax, Metric, param, sigma=-1, L=1, M=1):
    points_x, points_y=[],[]  # Lists to store the x and y coordinates of maxima
    points_x_min, points_y_min=[],[]  # Lists to store the x and y coordinates of minima
    for i in range(len(param)):
        # Set the parameters for the Metric object
        Metric.set_parameters(param[i])
        # Get the locations of the minima and maxima of V_eff
        min_store, max_store = Metric.min_max_V_eff_debug(sigma=sigma, L=L)
        
        points_x.append(max_store)  # Store the x-coordinate of the maximum
        points_y.append(Metric.V_eff(max_store, sigma=sigma, L=L))  # Store the corresponding V_eff value at the maximum

        # If a minimum exists (min is not equal to max), store it
        if min_store != max_store and min_store is not None:
            points_x_min.append(min_store)  # Store the x-coordinate of the minimum
            points_y_min.append(Metric.V_eff(min_store, sigma=sigma, L=L))  # Store the corresponding V_eff value at the minimum
            # Plot the minimum on the graph with a specific color and marker style
            if len(param) < 4: ax.scatter(min_store, Metric.V_eff(min_store, sigma=sigma, L=L), color=(0, 0, param[i][0]), 
                       label=f'Minimum for $Q_b={param[i][0]}$', marker='o', edgecolors=(1, 0.5, 0), linewidth=1, zorder=5)

        # Plot the maximum if the number of parameters is less than 4
        if len(param) < 4:
            ax.scatter(points_x[i], points_y[i], color=(0, 0, param[i][0]), label=f'Maximum for $Q_b={param[i][0]}$', 
                       marker='o', edgecolors=(1, 0, 0), linewidth=1, zorder=5)
    
    # If there are at least 3 parameters, plot the maxima and minima lines
    if len(param) >= 3:
        ax.plot(points_x, points_y, color=(1, 0, 0), label='Maxima', zorder=4)
        ax.plot(points_x_min, points_y_min, color=(1, 0.5, 0), label='Minima', zorder=4)

# Function to create and display the full plot for V_eff
def Veff_plt(Metric, param=[0, 0, 2], sigma=-1, L=1, rgb_image=None, grad=None, M=1, r_int=[2, 10], maxmin=True):
    
    # If a gradient image (rgb_image) is given, create the color spectrum and adjust the parameters
    if grad is not None: 
        # Create the color image and beta list (colors) with a linear gradient from 0 to 1
        rgb_image, beta_list = Color_img(np.linspace(0, 1, grad))
        # Adjust the parameters by converting the color values to parameters
        param = [(float(beta_list[i][2]), 0, 2 * M) for i in range(len(beta_list))]

    # Create a new figure with size 10x8
    fig = plt.figure(figsize=(10, 8))

    # Add a gridspec to control the layout of the plots (2 rows, 1 column)
    gs = fig.add_gridspec(2, 1) 

    # Create the first subplot for the V_eff plot (upper half)
    V_plt = fig.add_subplot(gs[0, 0])  
    # Call Veff_graph to create the V_eff plot
    if maxmin: 
        MaxMin(V_plt, Metric, param=param, sigma=sigma, L=L)
    Veff_graph(V_plt, Metric, param, L=L, sigma=sigma, bool_legend=rgb_image is None, r_int=r_int, M=M)
    
    V_plt.legend()  # Display the legend for the plot

    # If rgb_image is provided (i.e., color spectrum is present), create the color spectrum plot
    if rgb_image is not None: 
        # Create the second subplot for the color spectrum (lower half)
        spec = fig.add_subplot(gs[1, :])  
        # Call color_spec to display the color spectrum
        color_spec(spec, rgb_image=rgb_image, y=0.05)

    # Adjust the layout to avoid overlaps
    plt.tight_layout()
    
    # Display the plot
    plt.show()


def trPlot(Metric, param=[(0, 0, 2)], r_0=5, sigma=-1, L=1, E=1.4, end=10, M=1, grad=None, debug=False):
    # If a gradient image (rgb_image) is given, create the color spectrum and adjust the parameters
    if grad is not None: 
        # Create the color image and beta list (colors) with a linear gradient from 0 to 1
        rgb_image, beta_list = Color_img(np.linspace(0, 1, grad))
        # Adjust the parameters by converting the color values to parameters
        param = [(float(beta_list[i][2]), 0, 2 * M) for i in range(len(beta_list))]

    tau_span = [0, end]
    tau_list = np.linspace(tau_span[0], tau_span[1], 10*end)  # Time array for simulation

    Res=[]  # To store results from each calculation
        
    # Loop through the parameters and solve the differential equations for each
    for i in range(len(param)):
        Metric.set_parameters(param[i])  # Set the parameters for the Metric object
        sp, sn, f = Metric.solve_DAE(tau_list, tau_span, r_0, sigma = sigma, L = L, E = E, debug=debug)
        Res.append(sp)  # Store the solution for positive (sp) trajectory
        Res.append(sn)  # Store the solution for negative (sn) trajectory

    fig = plt.figure(figsize=(18, 13))  # Create a new figure with a specific size
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1, 0.1])  # Create a grid layout for subplots

    # Define the subplots
    Orbit = fig.add_subplot(gs[0, 2])  # Subplot for the orbit (trajectory)
    V_eff = fig.add_subplot(gs[0, 0:2])  # Subplot for the effective potential

    Beta = fig.add_subplot(gs[1, 0])  # Subplot for Beta vs time
    RadiusBeta = fig.add_subplot(gs[1, 1])  # Subplot for Beta vs radius
    Radius = fig.add_subplot(gs[1, 2])  # Subplot for radius vs time
    
    # Loop through results and plot them
    for i in range(len(Res)):
        # Choose the color based on the parameter for the current trajectory
        if i % 2 == 0: 
            Clr = (0, 0, param[int(i/2)][0])  # Color for positive trajectory
        else: 
            Clr = (0, 0, param[int((i-1)/2)][0])  # Color for negative trajectory

        # Plot the results in the respective subplots
        Orbit.plot(Res[i][1] * np.cos(Res[i][2]), Res[i][1] * np.sin(Res[i][2]), c=Clr)
        Radius.plot(Res[i][0], Res[i][1], c=Clr)  
        Beta.plot(Res[i][0], Res[i][2], c=Clr) 
        RadiusBeta.plot(Res[i][1], Res[i][2], c=Clr) 

    Orbit.set_aspect('equal', adjustable='box')  # Set the aspect ratio for the orbit plot

    # Plot the event horizon in the orbit plot
    Phi_event_horizon = np.linspace(0, 2 * np.pi, 50)
    R__event_horizon = [2 * M] * len(Phi_event_horizon)  # Event horizon radius
    Orbit.fill_between(R__event_horizon * np.cos(Phi_event_horizon), 
                       R__event_horizon * np.sin(Phi_event_horizon), 
                       color=(0, 0, 0), alpha=1, label='event horizon', zorder=5)

    # Set titles, labels, and legends for the subplots
    Orbit.set_title('Trajectory of particle')
    Orbit.set_xlabel('x [$R_s$]')
    Orbit.set_ylabel('y [$R_s$]')
    Orbit.legend()

    Radius.set_title('r(t) of particle') 
    Radius.set_xlabel('t')
    Radius.set_ylabel('r [$R_s$]') 

    Beta.set_title(rf'$\beta$(t) of particle') 
    Beta.set_xlabel('t')
    Beta.set_ylabel(rf'$\beta$') 

    RadiusBeta.set_title(rf'$\beta$(r) of particle') 
    RadiusBeta.set_xlabel('r [$R_s$]')
    RadiusBeta.set_ylabel(rf'$\beta$')     

    # Calculate the maximum radius from the first result
    max_r = Res[0][1][np.argmax(Res[0][1])]
    if max_r < (2*r_0-2  ): max_r = 2*r_0-2  
    # Plot the effective potential graph
    Veff_graph(V_eff, Metric, param, sigma=sigma, L=L, M=M, r_int=[2, int(max_r)], bool_legend=False)
    
    # Plot maximum and minimum effective potential
    MaxMin(V_eff, Metric, param=param, sigma=sigma, L=L)

    V_eff.grid(False)  # Turn off the grid for the effective potential plot

    MaxLine = E**2  # Maximum value of the potential (Energy squared)

    Metric.set_parameters(param[0])  # Set the parameters for the Metric again

    # Plot the critical radius r_0 and energy E_0
    V_eff.plot([r_0, r_0], np.linspace(Metric.V_eff(2*M, sigma=sigma, L=L), MaxLine, 2), c='grey', zorder=4, label='$r_0$')
    V_eff.plot(np.linspace(2*M, int(max_r), 2), [abs(E**2), abs(E**2)], c='grey', zorder=4, label='$E_0$')
    V_eff.legend()  # Display the legend

    # Add a color spectrum to the figure (for gradient)
    if len(param)>1: 
        spec = fig.add_subplot(gs[2, :])  
        if grad is not None: color_spec(spec, y=0.025, grad=grad)  # Color specification for the bottom spectrum plot
        else: color_spec(spec, y=0.025, v_list= [row[0] for row in param]) 

    plt.tight_layout()  # Adjust the layout to avoid overlapping
    plt.show()  # Display the plot

