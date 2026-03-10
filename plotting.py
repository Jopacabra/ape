import numpy as np
import matplotlib.lines as mlines
import pythia8
from matplotlib import pyplot as plt

import hard_particles


def plot_trajectories(hard_event : hard_particles.EventRecord, *, z_axis: str = "etas", color_by: str | None = "id", rap_max=1.0) -> None:
    """
    2D or 3D line plot of all Particle trajectories in this EventRecord.

    Assumes each particle has:
      - history: list of (tau, x, y, etas, px, py, pz)
      - id: PDG id (optional, for coloring/legend)

    Parameters
    ----------
    hard_event : hard_particles.EventRecord to plot trajectories from.
    z_axis : {"etas", "tau", "z", None}
        Which quantity to use as the Z axis.
        NOTE: if z_axis == "z", we plot z = tau * sinh(etas) in fm
              if z_axis == "etas", we plot z = etas.
              if z_axis == "tau", we plot z = tau.
        NOTE: If z_axis ==
    color_by : {"id", None}
        If "id", color trajectories by particle id (categorical colormap).
    """

    col_index = {"tau": 0, "x": 1, "y": 2, "etas": 3}
    if z_axis not in ("etas", "tau", "z", None):
        raise ValueError("z_axis must be 'etas', 'tau', 'z', or None for 2D")

    fig = plt.figure(figsize=(7, 7))
    if z_axis is None:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection="3d")

    # Fix aspect ratio.
    ax.set_aspect("equal")

    # Optional categorical coloring by particle id
    if color_by == "id":
        ids = np.array([])
        for p in hard_event.particles:
            if rap_max is not None and np.abs(p.rap) > rap_max:
                continue
            ids = np.append(ids, p.id)
        uniq_ids = np.unique(ids)
        cmap = plt.get_cmap("tab20", max(len(uniq_ids), 1))
        id_to_color = {pid: cmap(i) for i, pid in enumerate(uniq_ids)}
    else:
        id_to_color = {}

    any_plotted = False

    for idx, p in enumerate(hard_event.particles):
        hist = getattr(p, "history", None)
        if not hist:
            continue

        if rap_max is not None and np.abs(p.rap) > rap_max:
            continue

        traj = np.asarray(hist, dtype=float)
        if traj.ndim != 2 or traj.shape[1] < 4:
            raise ValueError(f"Particle {idx} has unexpected history shape: {traj.shape}")

        tau = traj[:, col_index["tau"]]
        x = traj[:, col_index["x"]]
        y = traj[:, col_index["y"]]
        if z_axis == "z":
            z = tau * np.sinh(traj[:, col_index["etas"]])  # z [fm] proxy
            z_label = "z = tau*sinh(etas) [fm]"
        elif z_axis == "etas":
            z = traj[:, col_index["etas"]]
            z_label = "etas"
        elif z_axis == "tau":
            z = tau
            z_label = "tau [fm]"
        else:
            z = None
            z_label = None

        pid = getattr(p, "id", None)
        color = id_to_color.get(pid, None)

        if z is None:
            ax.plot(x, y, lw=1.5, alpha=0.9, color=color)
        else:
            ax.plot(x, y, z, lw=1.5, alpha=0.9, color=color)
            ax.set_zlabel(z_label)
        any_plotted = True

    ax.set_xlabel("x [fm]")
    ax.set_ylabel("y [fm]")

    title = "Particle trajectories"
    if getattr(hard_event, "event_id", None) is not None:
        title += f" (event_id={hard_event.event_id})"
    ax.set_title(title)

    if not any_plotted:
        ax.text2D(0.05, 0.95, "No trajectories to plot (empty history).", transform=ax.transAxes)

    if color_by == "id" and len(id_to_color) <= 12:

        if z_axis is None:
            handles = [
                mlines.Line2D([], [], color=c, lw=2, label=f"id={pid}")
                for pid, c in id_to_color.items()
            ]
            ax.legend(handles=handles, loc="best")
        else:
            handles = [
                mlines.Line2D([], [], color=c, lw=2, label=f"id={pid}")
                for pid, c in id_to_color.items()
            ]
            ax.legend(handles=handles, loc="best")

    plt.tight_layout()
    plt.show()


def azimuthal_plot_pythia(pythia_event : pythia8.Event, bottom=8, max_height=4, N=80):
    """
    Radial histogram of energy deposition from particles in the event record.

    Parameters
    ----------
    pythia_event : pythia8.Event to plot trajectories from.
    bottom : The radius at which to start the histogram.
    max_height : The maximum height of the histogram.
    N : The number of bins in the histogram.
    """

    num_particles = pythia_event.size()

    weight_array = np.array([])  # Array for weight of particle
    phi_array = np.array([])  # Array for azimuthal coordinate of particle
    for i in range(num_particles):
        p = pythia_event[i]
        if p.isFinal():  # Only record final state particles
            phi_array = np.append(phi_array, p.phi())
            weight_array = np.append(weight_array, p.pT())



    phi_bins = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    counts, _ = np.histogram(phi_array, bins=phi_bins, weights=weight_array)
    radii = (max_height / np.amax(counts)) * counts
    width = (2 * np.pi) / N

    axis = plt.subplot(111, polar=True)
    bars = axis.bar((phi_bins[0:-1] + phi_bins[1:])/2, radii, width=width, bottom=bottom)

    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.8)

    return axis


def plot_parton_hadron(hard_event : hard_particles.EventRecord, hadrons : pythia8.Event, rap_max=1.0, N=80, max_height=3):
    # Create figure
    fig = plt.figure(figsize=(7, 7))

    ################
    # Trajectories #
    ################
    if hard_event is not None:
        # Create axis
        axis = fig.add_subplot(111, frameon=False)
        axis.grid(False)

        # Collect parton plot info and coloring
        col_index = {"tau": 0, "x": 1, "y": 2, "etas": 3}

        # Fix aspect ratio.
        axis.set_aspect("equal")

        # Categorical coloring by particle id
        ids = np.array([])
        for p in hard_event.particles:
            if rap_max is not None and np.abs(p.rap) > rap_max:
                continue
            ids = np.append(ids, p.id)
        uniq_ids = np.unique(ids)
        cmap = plt.get_cmap("tab20", max(len(uniq_ids), 1))
        id_to_color = {pid: cmap(i) for i, pid in enumerate(uniq_ids)}


        # Iterate over particles and plot
        rmax = 0
        for idx, p in enumerate(hard_event.particles):
            hist = getattr(p, "history", None)
            if not hist:
                continue

            if rap_max is not None and np.abs(p.rap) > rap_max:
                continue

            traj = np.asarray(hist, dtype=float)
            if traj.ndim != 2 or traj.shape[1] < 4:
                raise ValueError(f"Particle {idx} has unexpected history shape: {traj.shape}")

            x = np.array(traj[:, col_index["x"]]) - p.x_0
            y = np.array(traj[:, col_index["y"]]) - p.y_0

            pid = getattr(p, "id", None)
            color = id_to_color.get(pid, None)

            axis.plot(x, y, lw=1.5, alpha=0.9, color=color)
            any_plotted = True
            rmax = max(rmax, np.max(np.hypot(x, y)))

        # axis.set_xlabel("x [fm]")
        # axis.set_ylabel("y [fm]")

        # title = "Particle trajectories"
        # if getattr(hard_event, "event_id", None) is not None:
        #     title += f" (event_id={hard_event.event_id})"
        # axis.set_title(title)

        if not any_plotted:
            axis.text2D(0.05, 0.95, "No trajectories to plot (empty history).", transform=axis.transAxes)

        if len(id_to_color) <= 12:
            handles = [
                mlines.Line2D([], [], color=c, lw=2, label=f"id={pid}")
                for pid, c in id_to_color.items()
            ]
            axis.legend(handles=handles, loc="best")

    ##################
    # "Calorimeters" #
    ##################
    if hadrons is not None:

        num_particles = hadrons.size()
        if hard_event is not None:
            bottom = rmax
        else:
            bottom = 4

        weight_array = np.array([])  # Array for weight of particle
        phi_array = np.array([])  # Array for azimuthal coordinate of particle
        for i in range(num_particles):
            p = hadrons[i]
            if p.isFinal():  # Only record final state particles
                phi_array = np.append(phi_array, np.mod(p.phi(), 2*np.pi))
                weight_array = np.append(weight_array, p.pT())

        phi_bins = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        counts, _ = np.histogram(phi_array, bins=phi_bins, weights=weight_array)
        radii = (max_height / np.amax(counts)) * counts
        width = (2 * np.pi) / N

        paxis = fig.add_axes(111, polar=True, frameon=False)
        paxis.grid(False)
        bars = paxis.bar((phi_bins[0:-1] + phi_bins[1:]) / 2, radii, width=width, bottom=bottom)

        # Use custom colors and opacity
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.jet(r / 10.))
            bar.set_alpha(0.8)

    if hard_event is not None and hadrons is not None:
        # Align
        axis.set_xlim(-rmax-max_height, rmax+max_height)
        axis.set_ylim(-rmax-max_height, rmax+max_height)
        paxis.set_rlim(0,rmax+max_height)

        # Remove all ticks
        axis.set_xticks([])
        axis.set_yticks([])
        paxis.set_xticks([])
        paxis.set_yticks([])

    plt.show()
