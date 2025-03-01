import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
from tkinter.simpledialog import askfloat
from tkinter import messagebox
import numpy as np
import plasma
import collision
import jets
import matplotlib
from matplotlib import style
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # , NavigationToolbar2Tk
# import matplotlib.animation as animation
# import matplotlib.colors as colors
from utilities import round_decimals_up, round_decimals_down
import timekeeper
import config

matplotlib.use('TkAgg')  # Use proper matplotlib backend
style.use('Solarize_Light2')


# Define fonts
LARGE_FONT = ('Verdana', 10)

#####################
# Generic Functions #
#####################


# Function to slap together a label string to display from calculated moment data.
def moment_label(moment, angleDeflection, k=0, label='Total'):
    string = 'k = ' + str(k) + ' ' + str(label) + ' Jet Drift Moment: ' + str(moment) + ' GeV\n' \
             + 'Angular Deflection: ' + str(angleDeflection) + ' deg'
    return string


###########################
# Main Application Object #
###########################
# Define the main application object, inheriting from tk.Tk object type.
class PlasmaInspector(tk.Tk):
    # Initialization. Needs self. *args allows any number of parameters to be passed in,  **kwargs allows dictionaries.
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)  # Initializes tkinter business passed into the PlasmaInspector.
        # tk.Tk.iconbitmap(self, default='icon.ico')  # Sets the application icon - not working on Mephisto
        tk.Tk.wm_title(self, 'Plasma Inspector')
        # Create menu bar
        self.menubar = tk.Menu(self)
        tk.Tk.config(self, menu=self.menubar)

        # Dictionary of frames
        # Top one is active frame, can hold many frames for cool stuff to do
        self.frames = {}

        # Instantiate each frame and add them into our dictionary of frames
        for frame in [MainPage]:

            current_frame = frame(self)  # Instantiate frame

            self.frames[frame] = current_frame  # Add to dictionary

            # Sets where the frame goes
            # sticky -> North South East West... Defines which directions things stretch in. This is all directions
            # You can stick stuff wherever you want, you don't need to predefine grid size and shape
            current_frame.grid()

        self.show_frame(MainPage)

    # Method to show a particular frame
    def show_frame(self, framekey):

        # Selects frame to be raised to be frame within frames dictionary with key framekey
        frame = self.frames[framekey]

        # Raises the selected frame to the front
        frame.tkraise()


#############
# Main Page #
#############
# Define the main page, inheriting from the tk.Frame class
class MainPage(tk.Frame):

    def __init__(self, parent):
        # initialize frame inheritence
        tk.Frame.__init__(self, parent)

        # Set initialization flags
        self.file_selected = False

        # ##############################
        # Initialize adjustable values #
        ################################
        # Physical options
        self.time = tk.DoubleVar()  # Holds a float; default value 0.0
        self.phi0 = tk.DoubleVar()
        self.x0 = tk.DoubleVar()
        self.y0 = tk.DoubleVar()
        self.parton_E = tk.DoubleVar()
        self.parton_E.set(3)
        self.nth = tk.IntVar()
        self.nth.set(10)
        self.calculated = tk.BooleanVar()
        self.calculated.set(False)
        self.drift = tk.BooleanVar()
        self.drift.set(True)
        self.fgqhat = tk.BooleanVar()
        self.fgqhat.set(False)
        self.fg = tk.BooleanVar()
        self.fg.set(False)
        self.el = tk.BooleanVar()
        self.el.set(True)
        self.cel = tk.BooleanVar()
        self.cel.set(False)
        self.el_model = tk.StringVar()
        self.el_model.set('num_GLV')

        # Integration options
        self.tempHRG = tk.DoubleVar()
        self.tempHRG.set(config.jet.T_HRG)
        self.tempUnhydro = tk.DoubleVar()
        self.tempUnhydro.set(config.jet.T_UNHYDRO)

        # Plotting options
        self.plot_temp = tk.BooleanVar()
        self.plot_temp.set(True)
        self.plot_vel = tk.BooleanVar()
        self.plot_vel.set(True)
        self.plot_grad = tk.BooleanVar()
        self.plot_grad.set(False)
        self.velocityType = tk.StringVar()
        self.velocityType.set('stream')
        self.gradientType = tk.StringVar()
        self.gradientType.set('stream')
        self.contourNumber = tk.IntVar()
        self.contourNumber.set(15)
        self.tempType = tk.StringVar()
        self.tempType.set('contour')
        self.propPlotRes = tk.DoubleVar()
        self.propPlotRes.set(0.2)
        self.plotColors = tk.BooleanVar()
        self.plotColors.set(True)
        self.zoom = tk.DoubleVar()
        self.zoom.set(1)

        # Moment variables
        self.parton_dataframe = None
        self.parton_xarray = None
        self.moment = 0
        self.angleDeflection = 0
        self.momentPlasma = 0
        self.angleDeflectionPlasma = 0
        self.momentHRG = 0
        self.angleDeflectionHRG = 0
        self.K = 0
        self.momentDisplay = tk.StringVar()
        self.momentDisplay.set(moment_label(moment=None, angleDeflection=None,
                                            k=self.K, label='Total'))
        self.ELDisplay = tk.StringVar()
        self.ELDisplay.set('...')
        self.momentHRGDisplay = tk.StringVar()
        self.momentHRGDisplay.set('...')
        self.momentUnhydroDisplay = tk.StringVar()
        self.momentUnhydroDisplay.set('...')

        ############################
        # Event management objects #
        ############################
        self.hydro_file = None
        self.current_event = None
        self.temp_max = None
        self.temp_min = None

        #############################
        # Parton management objects #
        #############################
        self.current_parton = None

        ################
        # Plot Objects #
        ################

        # Create the QGP Plot that will dynamically update and set its labels
        self.plasmaFigure = plt.figure(num=0)
        self.plasmaAxis = self.plasmaFigure.add_subplot(1, 1, 1)

        # Define colorbar objects with "1" scalar mappable object so they can be manipulated.
        self.tempcb = 0
        self.velcb = 0
        self.gradcb = 0

        # Define plots
        self.tempPlot = None
        self.velPlot = None
        self.gradPlot = None

        # Create the medium property plots that will dynamically update
        # Constrained layout does some magic to organize the plots
        self.propertyFigure, self.propertyAxes = plt.subplots(3, 4, constrained_layout=True, num=1)

        # Create canvases and show the empty plots
        self.canvas = FigureCanvasTkAgg(self.plasmaFigure, master=self)
        self.canvas.draw()
        self.canvas1 = FigureCanvasTkAgg(self.propertyFigure, master=self)
        self.canvas1.draw()

        # Make our cute little toolbar for the plasma plot
        # self.toolbar = NavigationToolbar2Tk(self.canvas1, self)
        # self.toolbar.update()

        # Set up the moment display
        self.momentLabel = tk.Label(self, textvariable=self.momentDisplay, font=LARGE_FONT)
        self.momentPlasmaLabel = tk.Label(self, textvariable=self.ELDisplay, font=LARGE_FONT)
        self.momentHRGLabel = tk.Label(self, textvariable=self.momentHRGDisplay, font=LARGE_FONT)
        self.momentUnhydroLabel = tk.Label(self, textvariable=self.momentUnhydroDisplay, font=LARGE_FONT)

        ############
        # Controls #
        ############
        # Create button to change pages
        #buttonPage1 = ttk.Button(self, text='Go to Page 1', command=lambda: parent.show_frame(PageOne))

        # Create button to update plots
        self.update_button = ttk.Button(self, text='Update Plots',
                                   command=self.update_plots)

        # Create button to calculate and display moment
        self.run_parton_button = ttk.Button(self, text='Run Parton',
                                            command=self.run_parton)

        # Create button to sample the event and set parton initial conditions.
        self.sample_button = ttk.Button(self, text='Sample',
                                        command=self.sample_event)



        # Create time slider
        self.timeSlider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.time,
                                   from_=0, to=10, length=200, resolution=0.1, label='time (fm)')
        # Create x0 slider
        self.x0Slider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.x0,
                                 from_=0, to=10, length=200, resolution=0.1, label='x0 (fm)')
        # Create y0 slider
        self.y0Slider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.y0,
                                 from_=0, to=10, length=200, resolution=0.1, label='y0 (fm)')
        # Create phi0 slider
        self.phi0Slider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.phi0,
                                   from_=0, to=2*np.pi, length=200, resolution=0.01, label='phi0 (rad)')
        # Create parton energy slider
        self.parton_E_slider = tk.Scale(self, orient=tk.HORIZONTAL, variable=self.parton_E,
                                        from_=0.1, to=100, length=200, resolution=0.1, label='Parton E (GeV)')
        # Create tempHRG slider
        self.tempCutoffSlider = tk.Scale(self, orient=tk.HORIZONTAL,
                                         variable=self.tempHRG, from_=0, to=1, length=200, resolution=0.01,
                                         label='Had. Temp (GeV)')
        # Create tempHRG slider
        self.tempUnhydroSlider = tk.Scale(self, orient=tk.HORIZONTAL,
                                          variable=self.tempUnhydro, from_=0, to=1, length=200, resolution=0.01,
                                          label='Unhydro Temp (GeV)')
        # Create zoom slider
        self.zoomSlider = tk.Scale(self, orient=tk.HORIZONTAL,
                                          variable=self.zoom, from_=0, to=1, length=200, resolution=0.01,
                                          label='Zoom')


        # Register update ON RELEASE - use of command parameter applies action immediately
        #self.update_button.bind("<ButtonRelease-1>", self.update_plots)
        self.timeSlider.bind("<ButtonRelease-1>", self.update_parton)
        self.x0Slider.bind("<ButtonRelease-1>", self.update_parton)
        self.y0Slider.bind("<ButtonRelease-1>", self.update_parton)
        self.phi0Slider.bind("<ButtonRelease-1>", self.update_parton)
        self.parton_E_slider.bind("<ButtonRelease-1>", self.update_parton)
        self.tempCutoffSlider.bind("<ButtonRelease-1>", self.update_parton)
        self.tempUnhydroSlider.bind("<ButtonRelease-1>", self.update_parton)
        self.zoomSlider.bind("<ButtonRelease-1>", self.update_parton)

        #########
        # Menus #
        #########
        # Create file menu cascade
        # ---
        fileMenu = tk.Menu(parent.menubar, tearoff=0)
        fileMenu.add_command(label='Select File', command=self.select_file)
        fileMenu.add_command(label='Save Parton Record', command=self.save_record)
        fileMenu.add_command(label='Woods-Saxon', command=self.woods_saxon)
        fileMenu.add_command(label='Optical Glauber', command=self.optical_glauber)
        fileMenu.add_command(label='Log(Mult) Temp Optical Glauber', command=self.lmt_optical_glauber)
        fileMenu.add_command(label='\"new\" Optical Glauber', command=self.new_optical_glauber)
        fileMenu.add_command(label='Exit', command=self.quit)
        parent.menubar.add_cascade(label='File', menu=fileMenu)
        # ---

        # Create physics menu cascade
        # ---
        physicsMenu = tk.Menu(parent.menubar, tearoff=0)
        # scale submenu
        scale_menu = tk.Menu(physicsMenu, tearoff=0)
        physicsMenu.add_cascade(label='dtau', menu=scale_menu)
        scale_menu.add_command(label='Set dtau', command=self.select_dtau)
        scale_menu.add_command(label='Set hard process time', command=self.select_tau_prod)
        # drift submenu
        driftMenu = tk.Menu(physicsMenu, tearoff=0)
        physicsMenu.add_cascade(label='Flow Drift', menu=driftMenu)
        driftMenu.add_radiobutton(label='On', variable=self.drift, value=True,
                                command=self.not_calculated)
        driftMenu.add_radiobutton(label='Off', variable=self.drift, value=False,
                                  command=self.not_calculated)
        # Flow-Gradient q-hat mod submenu
        flowgradqhatMenu = tk.Menu(physicsMenu, tearoff=0)
        physicsMenu.add_cascade(label='Flow-Gradient qhat Mod', menu=flowgradqhatMenu)
        flowgradqhatMenu.add_radiobutton(label='On', variable=self.fgqhat, value=True,
                                  command=self.not_calculated)
        flowgradqhatMenu.add_radiobutton(label='Off', variable=self.fgqhat, value=False,
                                  command=self.not_calculated)

        # Flow-Gradients submenu
        flowgradMenu = tk.Menu(physicsMenu, tearoff=0)
        physicsMenu.add_cascade(label='Flow-Gradient Drift', menu=flowgradMenu)
        flowgradMenu.add_radiobutton(label='On', variable=self.fg, value=True,
                                 command=self.not_calculated)
        flowgradMenu.add_radiobutton(label='Off', variable=self.fg, value=False,
                                 command=self.not_calculated)

        # EL submenu
        elMenu = tk.Menu(physicsMenu, tearoff=0)
        celMenu = tk.Menu(physicsMenu, tearoff=0)
        el_model_menu = tk.Menu(elMenu, tearoff=0)
        elMenu.add_cascade(label='Model', menu=el_model_menu)
        physicsMenu.add_cascade(label='Radiative Energy Loss', menu=elMenu)
        elMenu.add_radiobutton(label='On', variable=self.el, value=True,
                                 command=self.not_calculated)
        elMenu.add_radiobutton(label='Off', variable=self.el, value=False,
                                 command=self.not_calculated)
        physicsMenu.add_cascade(label='Collisional Energy Loss', menu=celMenu)
        celMenu.add_radiobutton(label='On', variable=self.cel, value=True,
                               command=self.not_calculated)
        celMenu.add_radiobutton(label='Off', variable=self.cel, value=False,
                               command=self.not_calculated)
        el_model_menu.add_radiobutton(label='Numerical GLV', variable=self.el_model, value='num_GLV',
                                      command=self.not_calculated)
        el_model_menu.add_radiobutton(label='Analytic Approx. GLV', variable=self.el_model, value='GLV',
                               command=self.not_calculated)
        el_model_menu.add_radiobutton(label='BBMG', variable=self.el_model, value='BBMG',
                               command=self.not_calculated)
        parent.menubar.add_cascade(label='Physics', menu=physicsMenu)

        # Create plasma plot menu cascade
        # ---
        plasmaMenu = tk.Menu(parent.menubar, tearoff=0)
        # Parton submenu
        parton_menu = tk.Menu(plasmaMenu, tearoff=0)
        plasmaMenu.add_cascade(label='Parton', menu=parton_menu)
        parton_menu.add_radiobutton(label='Coarse (Every 20th)', variable=self.nth, value=20,
                                    command=self.update_plots)
        parton_menu.add_radiobutton(label='Medium-Coarse (Every 15th)', variable=self.nth, value=15,
                                    command=self.update_plots)
        parton_menu.add_radiobutton(label='Medium (Every 10th)', variable=self.nth, value=10,
                                    command=self.update_plots)
        parton_menu.add_radiobutton(label='Medium-Fine (Every 5th)', variable=self.nth, value=5,
                                    command=self.update_plots)
        parton_menu.add_radiobutton(label='Fine (Every 2nd)', variable=self.nth, value=2,
                                    command=self.update_plots)
        parton_menu.add_radiobutton(label='Ultra-Fine (1)', variable=self.nth, value=1,
                                    command=self.update_plots)
        # Temperature submenu
        tempMenu = tk.Menu(plasmaMenu, tearoff=0)
        plasmaMenu.add_cascade(label='Temperatures', menu=tempMenu)
        # Contours sub-submenu
        contourMenu = tk.Menu(tempMenu, tearoff=0)
        contourMenu.add_radiobutton(label='Coarse (10)', variable=self.contourNumber, value=10,
                                command=self.update_plots)
        contourMenu.add_radiobutton(label='Medium-Coarse (15)', variable=self.contourNumber, value=15,
                                 command=self.update_plots)
        contourMenu.add_radiobutton(label='Medium (20)', variable=self.contourNumber, value=20,
                                 command=self.update_plots)
        contourMenu.add_radiobutton(label='Medium-Fine (25)', variable=self.contourNumber, value=25,
                                 command=self.update_plots)
        contourMenu.add_radiobutton(label='Fine (30)', variable=self.contourNumber, value=30,
                                 command=self.update_plots)
        contourMenu.add_radiobutton(label='Ultra-Fine (40)', variable=self.contourNumber, value=40,
                                 command=self.update_plots)
        tempMenu.add_cascade(label='No. Contours', menu=contourMenu)
        # Plot type sub-submenu
        tempTypeMenu = tk.Menu(tempMenu, tearoff=0)
        tempMenu.add_cascade(label='Plot Style', menu=tempTypeMenu)
        tempTypeMenu.add_radiobutton(label='Contour', variable=self.tempType, value='contour',
                                    command=self.update_plots)
        tempTypeMenu.add_radiobutton(label='Density', variable=self.tempType, value='density',
                                    command=self.update_plots)
        tempTypeMenu.add_radiobutton(label='Enabled', variable=self.plot_temp, value=True,
                                     command=self.update_plots)
        tempTypeMenu.add_radiobutton(label='Disabled', variable=self.plot_temp, value=False,
                                     command=self.update_plots)



        # Velocity submenu
        velMenu = tk.Menu(plasmaMenu, tearoff=0)
        velTypeMenu = tk.Menu(velMenu)
        velTypeMenu.add_radiobutton(label='Stream Velocities', variable=self.velocityType, value='stream',
                              command=self.update_plots)
        velTypeMenu.add_radiobutton(label='Quiver Velocities', variable=self.velocityType, value='quiver',
                              command=self.update_plots)
        velTypeMenu.add_radiobutton(label='Enabled', variable=self.plot_vel, value=True,
                                    command=self.update_plots)
        velTypeMenu.add_radiobutton(label='Disabled', variable=self.plot_vel, value=False,
                                    command=self.update_plots)
        velMenu.add_cascade(label='Plot Style', menu=velTypeMenu)
        plasmaMenu.add_cascade(label='Velocities', menu=velMenu)
        parent.menubar.add_cascade(label='Plasma Plot', menu=plasmaMenu)

        # Gradient submenu
        gradMenu = tk.Menu(plasmaMenu, tearoff=0)
        gradTypeMenu = tk.Menu(gradMenu)
        gradTypeMenu.add_radiobutton(label='Stream Gradients', variable=self.gradientType, value='stream',
                                    command=self.update_plots)
        gradTypeMenu.add_radiobutton(label='Quiver Gradients', variable=self.gradientType, value='quiver',
                                    command=self.update_plots)
        gradTypeMenu.add_radiobutton(label='Enabled', variable=self.plot_grad, value=True,
                                     command=self.update_plots)
        gradTypeMenu.add_radiobutton(label='Disabled', variable=self.plot_grad, value=False,
                                     command=self.update_plots)
        gradMenu.add_cascade(label='Plot Style', menu=gradTypeMenu)
        plasmaMenu.add_cascade(label='Gradients', menu=gradMenu)

        # Property plot menu
        propMenu = tk.Menu(parent.menubar, tearoff=0)
        propMenu.add_checkbutton(label='Color Plots for Temp Cutoff', variable=self.plotColors, onvalue=1,
                                 offvalue=0, command=self.update_plots)
        propResMenu = tk.Menu(propMenu)
        propResMenu.add_radiobutton(label='0.1 fm', variable=self.propPlotRes, value=0.1,
                                    command=self.update_plots)
        propResMenu.add_radiobutton(label='0.2 fm', variable=self.propPlotRes, value=0.2,
                                    command=self.update_plots)
        propResMenu.add_radiobutton(label='0.3 fm', variable=self.propPlotRes, value=0.3,
                                    command=self.update_plots)
        propResMenu.add_radiobutton(label='0.4 fm', variable=self.propPlotRes, value=0.4,
                                    command=self.update_plots)
        propResMenu.add_radiobutton(label='0.5 fm', variable=self.propPlotRes, value=0.5,
                                    command=self.update_plots)
        propMenu.add_cascade(label='Plotting Time Resolution', menu=propResMenu)
        parent.menubar.add_cascade(label='Medium Properties', menu=propMenu)
        # ---

        ###########################
        # Organization and Layout #
        ###########################
        # Smash everything into the window
        self.update_button.grid(row=0, column=0)
        self.sample_button.grid(row=0, column=1)
        self.run_parton_button.grid(row=0, column=2)
        self.timeSlider.grid(row=1, column=0, columnspan=2)
        self.x0Slider.grid(row=1, column=2, columnspan=2)
        self.y0Slider.grid(row=1, column=4, columnspan=2)
        self.phi0Slider.grid(row=1, column=6, columnspan=2)
        self.parton_E_slider.grid(row=1, column=8, columnspan=2)
        self.zoomSlider.grid(row=1, column=10, columnspan=1)
        self.tempCutoffSlider.grid(row=5, column=8, columnspan=2)
        self.tempUnhydroSlider.grid(row=6, column=8, columnspan=2)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=6, rowspan=3)
        self.canvas1.get_tk_widget().grid(row=2, column=6, columnspan=6, rowspan=3)
        self.momentLabel.grid(row=5, column=0, columnspan=4)
        self.momentPlasmaLabel.grid(row=6, column=0, columnspan=4)
        self.momentHRGLabel.grid(row=5, column=4, columnspan=4)
        self.momentUnhydroLabel.grid(row=6, column=4, columnspan=4)
        # buttonPage1.grid()  # Unused second page

        # Create the parton object
        self.update_parton(0)

    # Define the select file function
    def select_file(self, value=None):
        hydro_file_path = askopenfilename(
            initialdir='/share/apps/Hybrid-Transport/hic-eventgen/results/')  # show an "Open" dialog box and return the path to the selected file
        self.file_selected = True
        print('Selected file: ' + str(hydro_file_path))

        # Open grid file as object
        self.hydro_file = plasma.osu_hydro_file(file_path=hydro_file_path)

        # Create plasma object from current_event object
        self.current_event = plasma.plasma_event(event=self.hydro_file)

        # Find current_event parameters
        maxTemp, minTemp, meanTemp, medianTemp, stdTemp = self.current_event.temp_stats()
        self.temp_max = maxTemp
        self.temp_min = minTemp
        self.tau0 = self.current_event.t0
        self.tauf = self.current_event.tf

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.tau0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.tauf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    # Define the save record function
    def save_record(self, value=None):
        filename = asksaveasfilename(initialdir='/share/apps/Hybrid-Transport/hic-eventgen/results/')
        self.parton_xarray.to_netcdf(filename + '.nc')

    # Define the woods-saxon selection
    def woods_saxon(self, value=None):
        # Ask user for optical glauber input parameters:
        b = askfloat("Input", "Enter impact parameter (b) in fm (max 2R): ", minvalue=0.0, maxvalue=2 * 6.62)
        T0 = askfloat("Input", "Enter temperature norm in GeV: ", minvalue=0.0, maxvalue=500)
        V0 = askfloat("Input", "Enter flow velocity norm in c: ", minvalue=0.0, maxvalue=1)
        alpha = askfloat("Input", "Enter expansion parameter: ", minvalue=0, maxvalue=2)
        self.file_selected = True  # Set that you have selected an event
        #print('Selected optical glauber:\nR = {}, b = {}, phi = {}, T0 = {}, V0 = {}'.format(R, b, phi, T0, V0))

        # Create plasma object
        rmax = 8
        self.current_event = collision.woods_saxon_plasma(b, resolution=5, rmax=rmax, alpha=1)

        # Find current_event parameters
        self.temp_max = self.current_event.max_temp()
        self.temp_min = self.current_event.min_temp()

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.current_event.t0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.current_event.tf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    # Define the optical glauber selection
    def optical_glauber(self, value=None):
        # Ask user for optical glauber input parameters:
        R = askfloat("Input", "Enter ion radius (R) in fm: ", minvalue=0.0, maxvalue=25)
        b = askfloat("Input", "Enter impact parameter (b) in fm (max 2R): ", minvalue=0.0, maxvalue=2*R)
        T0 = askfloat("Input", "Enter temperature norm in GeV: ", minvalue=0.0, maxvalue=500)
        V0 = askfloat("Input", "Enter flow velocity norm in c: ", minvalue=0.0, maxvalue=1)
        phi = askfloat("Input", "Enter reaction plane angle in rad: ", minvalue=0.0, maxvalue=2*np.pi)
        rmax = 15
        event_lifetime = 20
        self.file_selected = True  # Set that you have selected an event
        print('Selected optical glauber:\nR = {}, b = {}, phi = {}, T0 = {}, V0 = {}'.format(R, b, phi, T0, V0))

        # Generate optical glauber
        analytic_t, analytic_ux, analytic_uy, mult, e2 = collision.optical_glauber(R=R, b=b, phi=phi, T0=T0, U0=V0)

        # Create plasma object
        self.current_event = plasma.functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux,
                                                      y_vel_func=analytic_uy,
                                                      rmax=rmax, time=event_lifetime)

        # Find current_event parameters
        self.temp_max = self.current_event.max_temp()
        self.temp_min = self.current_event.min_temp()

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.current_event.t0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.current_event.tf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    # Define the optical glauber selection
    def lmt_optical_glauber(self, value=None):
        # Ask user for optical glauber input parameters:
        R = askfloat("Input", "Enter ion radius (R) in fm: ", minvalue=0.0, maxvalue=25)
        b = askfloat("Input", "Enter impact parameter (b) in fm (max 2R): ", minvalue=0.0, maxvalue=2 * R)
        T0 = askfloat("Input", "Enter temperature norm in GeV: ", minvalue=0.0, maxvalue=500)
        V0 = askfloat("Input", "Enter flow velocity norm in c: ", minvalue=0.0, maxvalue=1)
        phi = askfloat("Input", "Enter reaction plane angle in rad: ", minvalue=0.0, maxvalue=2 * np.pi)
        rmax = 15
        event_lifetime = 20
        self.file_selected = True  # Set that you have selected an event
        print('Selected lmt optical glauber:\nR = {}, b = {}, phi = {}, T0 = {}, V0 = {}'.format(R, b, phi, T0, V0))

        # Generate optical glauber
        analytic_t, analytic_ux, analytic_uy, mult, e2 = collision.optical_glauber_logT(R=R, b=b, phi=phi, T0=T0, U0=V0)

        # Create plasma object
        self.current_event = plasma.functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux,
                                                      y_vel_func=analytic_uy,
                                                      rmax=rmax, time=event_lifetime)

        # Find current_event parameters
        self.temp_max = self.current_event.max_temp()
        self.temp_min = self.current_event.min_temp()

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.current_event.t0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.current_event.tf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    # Define the optical glauber selection
    def new_optical_glauber(self, value=None):
        # Ask user for optical glauber input parameters:
        R = askfloat("Input", "Enter ion radius (R) in fm: ", minvalue=0.0, maxvalue=25)
        b = askfloat("Input", "Enter impact parameter (b) in fm (max 2R): ", minvalue=0.0, maxvalue=2 * R)
        T0 = askfloat("Input", "Enter temperature norm in GeV: ", minvalue=0.0, maxvalue=500)
        V0 = askfloat("Input", "Enter flow velocity norm in c: ", minvalue=0.0, maxvalue=1)
        phi = askfloat("Input", "Enter reaction plane angle in rad: ", minvalue=0.0, maxvalue=2 * np.pi)
        rmax = 15
        event_lifetime = 20
        self.file_selected = True  # Set that you have selected an event
        print('Selected "new" optical glauber:\nR = {}, b = {}, phi = {}, T0 = {}, V0 = {}'.format(R, b, phi, T0, V0))

        # Generate optical glauber
        analytic_t, analytic_ux, analytic_uy, mult, e2 = collision.optical_glauber_logT(R=R, b=b, phi=phi, T0=T0, U0=V0)

        # Create plasma object
        self.current_event = plasma.functional_plasma(temp_func=analytic_t, x_vel_func=analytic_ux,
                                                      y_vel_func=analytic_uy,
                                                      rmax=rmax, time=event_lifetime)

        # Find current_event parameters
        self.temp_max = self.current_event.max_temp()
        self.temp_min = self.current_event.min_temp()

        # Set sliders limits to match bounds of the event
        dec = 1  # number of decimals rounding to... Should match resolution of slider.
        self.timeSlider.configure(from_=round_decimals_up(self.current_event.t0, decimals=dec))
        self.timeSlider.configure(to=round_decimals_down(self.current_event.tf, decimals=dec))
        self.time.set(round_decimals_up(self.current_event.t0, decimals=dec))

        self.x0Slider.configure(from_=round_decimals_up(self.current_event.xmin, decimals=dec))
        self.x0Slider.configure(to=round_decimals_down(self.current_event.xmax, decimals=dec))
        self.x0.set(0)

        self.y0Slider.configure(from_=round_decimals_up(self.current_event.ymin, decimals=dec))
        self.y0Slider.configure(to=round_decimals_down(self.current_event.ymax, decimals=dec))
        self.y0.set(0)

        self.update_plots()

    def tempColoringFunc(self, t):
        colorArray = np.array([])
        for time in t:
            # Check temperature at the given time
            checkTemp = self.current_event.temp(self.current_parton.coords3(time))
            # Assign relevant color to plot
            if self.tempHRG.get() > checkTemp > self.tempUnhydro.get():
                # Hadron gas phase, hydrodynamic
                color = 'g'
            elif checkTemp < self.tempUnhydro.get():
                # Hadron gas phase, unhydrodynamic
                color = 'r'
            else:
                color = 'b'
            colorArray = np.append(colorArray, color)
        hydroHRGTimeIndexes = np.where(colorArray == 'g')
        hydroHRGTimes = np.array([])
        for index in hydroHRGTimeIndexes:
            time = t[index]
            hydroHRGTimes = np.append(hydroHRGTimes, time)
        unhydroHRGTimeIndexes = np.where(colorArray == 'r')
        unhydroHRGTimes = np.array([])
        for index in unhydroHRGTimeIndexes:
            time = t[index]
            unhydroHRGTimes = np.append(unhydroHRGTimes, time)
        return colorArray, hydroHRGTimes, unhydroHRGTimes

    # Define set not calculated function
    def not_calculated(self, value=None):
        self.calculated.set(False)

    # Define parton update function
    def update_parton(self, value=None):
        # Set current_parton object to current slider parameters
        self.current_parton = jets.parton(x_0=self.x0.get(), y_0=self.y0.get(),
                                          phi_0=self.phi0.get(), p_T0=self.parton_E.get())
        self.not_calculated(0)

    # Set dtau
    def select_dtau(self, value=None):
        config.jet.DTAU = askfloat("Input", "Enter parton prop. time step in fm: ", minvalue=0.0001, maxvalue=5)

    # Set production time of the hard process
    def select_tau_prod(self, value=None):
        config.jet.TAU_PROD = askfloat("Input", "Enter parton production. time in fm: ", minvalue=0, maxvalue=self.tauf)


    # Define the update function
    def update_plots(self, value=None):
        if self.file_selected:
            # # Clear all the plots and colorbars
            self.plasmaAxis.clear()
            try:
                self.tempcb.remove()
            except AttributeError:
                pass
            try:
                self.velcb.remove()
            except AttributeError:
                pass
            try:
                self.gradcb.remove()
            except AttributeError:
                pass
            #
            for axisList in self.propertyAxes:  # Medium property plots
                for axis in axisList:
                    axis.clear()

            # Create the QGP Plot that will dynamically update and set its labels
            self.plasmaFigure = plt.figure(num=0)
            self.plasmaAxis = self.plasmaFigure.add_subplot(1, 1, 1)

            # # Define colorbar objects with "1" scalar mappable object so they can be manipulated.
            # self.tempcb = 0
            # self.velcb = 0
            # self.gradcb = 0

            # Define plots
            self.tempPlot = None
            self.velPlot = None
            self.gradPlot = None

            # Create canvases and show the empty plots
            # self.canvas = FigureCanvasTkAgg(self.plasmaFigure, master=self)
            # self.canvas.draw()

            # Select QGP figure as current figure
            plt.figure(self.plasmaFigure.number)

            # Select QGP plot axis as current axis
            # plt.sca(self.plasmaAxis)

            # Plot new temperatures & velocities
            self.tempPlot, self.velPlot, self.gradPlot, self.tempcb, self.velcb, self.gradcb\
                = self.current_event.plot(self.time.get(),
                                          plot_temp=self.plot_temp.get(),
                                          plot_vel=self.plot_vel.get(),
                                          plot_grad=self.plot_grad.get(),
                                          temp_resolution=100,
                                          vel_resolution=30,
                                          gradtype=self.gradientType.get(),
                                          veltype=self.velocityType.get(),
                                          temptype=self.tempType.get(),
                                          numContours=self.contourNumber.get(),
                                          zoom=self.zoom.get())

            # Set moment display
            # !!!!!!!!!!!!! Currently Empty !!!!!!!!!!!!
            self.momentDisplay.set('...')
            self.ELDisplay.set('...')
            self.momentHRGDisplay.set('...')
            self.momentUnhydroDisplay.set('...')

            # Decide if you want to feed tempHRG to the integrand function to bring it to zero.
            if self.plotColors.get():
                decidedCut = 0
            else:
                decidedCut = self.tempHRG.get()

            # Set plot font hardcoded options
            plotFontSize = 8
            markSize = 2
            gridLineWidth = 1
            connectorLineStyle = '-'
            # connectorLineColor = 'black'  # Not set up

            #print(self.calculated.get())
            if self.calculated.get():

                # Plot parton trajectory
                # Select QGP figure as current figure
                plt.figure(self.plasmaFigure.number)
                # Plot initial trajectory
                d_x = np.cos(self.current_parton.phi_0) * self.current_parton.beta_0 * (0.5 * self.current_event.tf)
                d_y = np.sin(self.current_parton.phi_0) * self.current_parton.beta_0 * (0.5 * self.current_event.tf)
                self.plasmaAxis.arrow(self.current_parton.x_0, self.current_parton.y_0, d_x, d_y, color='white', width=0.15)
                # Get trajectory points
                time_array = self.parton_xarray['time'].to_numpy()
                xpos_array = self.parton_xarray['x'].to_numpy()
                ypos_array = self.parton_xarray['y'].to_numpy()
                q_drift_array = self.parton_xarray['q_drift'].to_numpy()
                q_EL_array = self.parton_xarray['q_el'].to_numpy()
                q_cel_array = self.parton_xarray['q_cel'].to_numpy()
                q_fg_utau_array = self.parton_xarray['q_fg_utau'].to_numpy()
                q_fg_uperp_array = self.parton_xarray['q_fg_uperp'].to_numpy()
                q_fg_total_array = q_fg_utau_array + q_fg_uperp_array
                q_fg_utau_qhat_array = self.parton_xarray['q_fg_utau_qhat'].to_numpy()
                q_fg_uperp_qhat_array = self.parton_xarray['q_fg_uperp_qhat'].to_numpy()
                pT_array = self.parton_xarray['pT'].to_numpy()
                temp_seen_array = self.parton_xarray['temp'].to_numpy()
                grad_perp_temp_array = self.parton_xarray['grad_perp_temp'].to_numpy()
                grad_perp_utau_array = self.parton_xarray['grad_perp_utau'].to_numpy()
                grad_perp_uperp_array = self.parton_xarray['grad_perp_uperp'].to_numpy()
                u_perp_array = self.parton_xarray['u_perp'].to_numpy()
                u_par_array = self.parton_xarray['u_par'].to_numpy()
                u_array = self.parton_xarray['u'].to_numpy()
                phase_array = self.parton_xarray['phase'].to_numpy()
                rpos_array = np.sqrt(xpos_array**2 + ypos_array**2)
                # Plot trajectory
                self.plasmaAxis.plot(xpos_array[::self.nth.get()], ypos_array[::self.nth.get()], marker=',',
                                     color='black', markersize=20)

                # Set moment display
                self.momentDisplay.set('Total F Drift: {} GeV'.format(np.sum(q_drift_array)))
                self.ELDisplay.set('Total EL: {} (el) \n+ {} (coll) \n+ {} (fg_utau) \n+ {} (fg_uperp) = {} GeV'.format(
                                                                       np.sum(q_EL_array),
                                                                       np.sum(q_cel_array),
                                                                       np.sum(q_fg_utau_qhat_array),
                                                                       np.sum(q_fg_uperp_qhat_array),
                                                                       (np.sum(q_EL_array)
                                                                        + np.sum(q_fg_utau_qhat_array)
                                                                        + np.sum(q_fg_uperp_qhat_array))))
                self.momentHRGDisplay.set('Total FG Drift: {} GeV'.format(np.sum(q_fg_utau_array)
                                                                          + np.sum(q_fg_uperp_array)))
                self.momentUnhydroDisplay.set('...')

                # Select medium properties figure as current figure
                plt.figure(self.propertyFigure.number)

                # Plot connector lines for properties
                self.propertyAxes[0, 0].plot(time_array, u_perp_array, ls=connectorLineStyle)
                self.propertyAxes[0, 1].plot(time_array, u_par_array, ls=connectorLineStyle)
                self.propertyAxes[1, 0].plot(time_array, q_fg_utau_qhat_array, ls=connectorLineStyle)
                self.propertyAxes[1, 1].plot(time_array, grad_perp_temp_array/(temp_seen_array**2), ls=connectorLineStyle)
                self.propertyAxes[1, 2].plot(time_array, q_EL_array, ls=connectorLineStyle)
                self.propertyAxes[2, 1].plot(time_array, grad_perp_utau_array/(u_par_array**2), ls=connectorLineStyle)
                self.propertyAxes[0, 2].plot(time_array, q_fg_uperp_qhat_array, ls=connectorLineStyle)
                self.propertyAxes[2, 2].plot(time_array, q_drift_array, ls=connectorLineStyle)
                self.propertyAxes[2, 0].plot(time_array, grad_perp_uperp_array/(u_perp_array**2), ls=connectorLineStyle)
                self.propertyAxes[0, 3].plot(time_array, pT_array, ls=connectorLineStyle)
                self.propertyAxes[1, 3].plot(time_array, q_fg_utau_array, ls=connectorLineStyle)
                self.propertyAxes[2, 3].plot(time_array, q_fg_uperp_array, ls=connectorLineStyle)

                if self.plotColors.get():
                    # Determine colors from temp seen by parton at each time.
                    color_array = np.array([])
                    for phase in phase_array:
                        if phase == 'qgp':
                            color_array = np.append(color_array, 'b')
                        elif phase == 'hrg':
                            color_array = np.append(color_array, 'g')
                        elif phase == 'unh':
                            color_array = np.append(color_array, 'r')
                        else:
                            color_array = np.append(color_array, 'black')

                    # Plot colored markers.
                    for i in range(0, len(time_array)):
                        self.propertyAxes[0, 0].plot(time_array[i], u_perp_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[0, 1].plot(time_array[i], u_par_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[1, 0].plot(time_array[i], q_fg_utau_qhat_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[1, 1].plot(time_array[i], grad_perp_temp_array[i]/(temp_seen_array[i]**2), 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[1, 2].plot(time_array[i], q_EL_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[2, 1].plot(time_array[i], grad_perp_utau_array[i]/(u_par_array[i]**2), 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[0, 2].plot(time_array[i], q_fg_uperp_qhat_array[i], 'o', color=color_array[i] , markersize=markSize)
                        self.propertyAxes[2, 2].plot(time_array[i], q_drift_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[2, 0].plot(time_array[i], grad_perp_uperp_array[i]/(u_perp_array[i]**2), 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[0, 3].plot(time_array[i], pT_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[1, 3].plot(time_array[i], q_fg_utau_array[i], 'o', color=color_array[i], markersize=markSize)
                        self.propertyAxes[2, 3].plot(time_array[i], q_fg_uperp_array[i], 'o', color=color_array[i], markersize=markSize)

                        # # PERFORMANCE ISSUES:
                        # # Fill under the drift integrand curve for qgp phase
                        # self.propertyAxes[2, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_drift_array,
                        #     where=(phase_array == 'qgp'),
                        #     color='b',
                        #     alpha=0.2)
                        # # Fill under the drift integrand curve for hydrodynamic hadron gas phase
                        # self.propertyAxes[2, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_drift_array,
                        #     where=(phase_array == 'hrg'),
                        #     color='g',
                        #     alpha=0.2)
                        # # Fill under the drift integrand curve for unhydrodynamic phase
                        # self.propertyAxes[2, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_drift_array,
                        #     where=(phase_array == 'unh'),
                        #     color='g',
                        #     alpha=0.2)
                        # # Fill under the EL integrand curve for qgp phase
                        # self.propertyAxes[1, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_EL_array,
                        #     where=(phase_array == 'qgp'),
                        #     color='b',
                        #     alpha=0.2)
                        # # Fill under the EL integrand curve for hydrodynamic hadron gas phase
                        # self.propertyAxes[1, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_EL_array,
                        #     where=(phase_array == 'hrg'),
                        #     color='g',
                        #     alpha=0.1)
                        # # Fill under the EL integrand curve for unhydrodynamic phase
                        # self.propertyAxes[1, 3].fill_between(
                        #     x=time_array,
                        #     y1=int_EL_array,
                        #     where=(phase_array == 'unh'),
                        #     color='g',
                        #     alpha=0.2)

            # Gridlines and ticks
            # Plot horizontal gridlines at y=0
            for axisList in self.propertyAxes:  # Medium property plots
                for axis in axisList:
                    axis.axhline(y=0, color='black', linestyle=':', lw=gridLineWidth)

            # Plot horizontal gridline at temp minTemp for temp plot
            #self.propertyAxes[1, 0].axhline(y=self.tempHRG.get(), color='black', linestyle=':', lw=gridLineWidth)

            # Plot vertical gridline at current time from slider
            for axisList in self.propertyAxes:  # Iterate through medium property plots
                for axis in axisList:
                    axis.axvline(x=self.time.get(), color='black', ls=':', lw=gridLineWidth)

            # Plot tick at temp minTemp for temp plot
            #self.propertyAxes[1, 0].set_yticks(list(self.propertyAxes[1, 0].get_yticks()) + [self.tempHRG.get()])

            # Plot property titles
            self.propertyAxes[0, 0].set_title(r"$u_\tau$", fontsize=plotFontSize)
            self.propertyAxes[0, 1].set_title(r"$u_\parallel$", fontsize=plotFontSize)
            self.propertyAxes[1, 0].set_title(r"$q_{\nabla_\perp u_\tau \hat{q}}$ (GeV)", fontsize=plotFontSize)
            self.propertyAxes[1, 1].set_title(r"$\nabla_{\perp} T / T^2$", fontsize=plotFontSize)
            self.propertyAxes[1, 2].set_title(r"$q_{EL}$", fontsize=plotFontSize)
            self.propertyAxes[2, 1].set_title(r"$\nabla_{\perp} u_{\tau} / u_{\tau}^2$", fontsize=plotFontSize)
            self.propertyAxes[0, 2].set_title(r"$q_{\nabla_\perp u_\perp \hat{q}}$", fontsize=plotFontSize)
            self.propertyAxes[2, 2].set_title(r"$q_{drift}$", fontsize=plotFontSize)
            self.propertyAxes[2, 0].set_title(r"$\nabla_{\perp} u_{\perp}/ u_{\perp}^2$", fontsize=plotFontSize)
            self.propertyAxes[0, 3].set_title(r"$p_T$", fontsize=plotFontSize)
            self.propertyAxes[1, 3].set_title(r"$q_{\nabla_\perp u_\tau}$", fontsize=plotFontSize)
            self.propertyAxes[2, 3].set_title(r"$q_{\nabla_\perp u_\perp}$", fontsize=plotFontSize)



        # If you've got no file loaded, just redraw the canvas.
        else:
            print('Select a file!!!')
        self.canvas.draw()
        self.canvas1.draw()

    # Method to animate plasma evolution
    def animate_plasma(self):
        return None

    def run_parton(self, value=None):
        if self.file_selected:
            # Update parton object
            self.update_parton(0)

            # Calculate the parton trajectory
            print('Calculating parton trajectory...')
            # Run the time loop
            self.parton_dataframe, self.parton_xarray = timekeeper.evolve(event=self.current_event,
                                                                          parton=self.current_parton,
                                                                          drift=self.drift.get(),
                                                                          el=self.el.get(),
                                                                          cel=self.cel.get(),
                                                                          fg=self.fg.get(),
                                                                          fgqhat=self.fgqhat.get(),
                                                                          temp_hrg=self.tempHRG.get(),
                                                                          temp_unh=self.tempUnhydro.get(),
                                                                          el_model=self.el_model.get())

            print('Parton trajectory complete.')
            self.calculated.set(True)
            # Update plots to set current parton and event business.
            self.update_plots()

        else:
            print('Select a file!!!')

    # Method to sample the event and return new parton production point
    def sample_event(self):
        if self.file_selected:
            # Sample T^6 dist. and get point
            sampledPoint = collision.generate_jet_seed_point(self.current_event, 1)

            # Uniform sample an angle
            sampledAngle = float(np.random.uniform(0, 2*np.pi, 1)[0])

            # Set sliders to point
            self.x0.set(sampledPoint[0])
            self.y0.set(sampledPoint[1])
            self.phi0.set(sampledAngle)

            # Update plots
            self.update_plots()
        else:
            print('Select a file!!!')



"""
# Unused second page with back button.
class PageOne(tk.Frame):

    def __init__(self, parent):
        # initialize frame inheritence
        tk.Frame.__init__(self, parent)

        # Define label and smash it in.
        label = tk.Label(self, text='Page 1', font=LARGE_FONT)
        label.grid()

        # Create button to change pages and smack it in
        backButton = ttk.Button(self, text='Back',
                                command=lambda: parent.show_frame(MainPage))
        backButton.grid()
"""







# Make the main application class
app = PlasmaInspector()

# Handle the window close
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        # Close matplotlib objects
        plt.close('all')
        # Close the application
        app.destroy()
        # Be polite. :)
        print('Have a lovely day!')

app.protocol("WM_DELETE_WINDOW", on_closing)

# Generate the window and let it hang out until closed
app.mainloop()


