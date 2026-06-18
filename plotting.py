import numpy as np
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
import pythia8
from matplotlib import pyplot as plt

import hard_particles
import config

id_color_dict = {21: "yellowgreen",
                 22: "m",
                 23: "orchid",
                 24: "fuchsia",
                 1: "black",
                 2: "black",
                 3: "black",
                 -1: "dimgrey",
                 -2: "dimgrey",
                 -3: "dimgrey",
                 2212: "crimson",
                 2112: "grey"}

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

    # Get colormap of appropriate size
    ids = np.array([])
    for p in hard_event.particles:
        if rap_max is not None and np.abs(p.rap) > rap_max:
            continue
        ids = np.append(ids, p.id)
    uniq_ids = np.unique(ids)
    cmap = plt.get_cmap("tab20", max(len(uniq_ids), 1))

    any_plotted = False
    id_to_color = {}
    j = 0
    for idx, p in enumerate(hard_event.particles):
        hist = getattr(p, "history", None)
        if p.status < 0:
            continue
        if not hist:
            continue
        if rap_max is not None and np.abs(p.rap) > rap_max:
            continue

        # Get color
        pid = p.id
        try:
            id_to_color[pid] = id_color_dict[int(pid)]
        except:
            id_to_color[pid] = cmap(j)
            j += 1

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
        if color_by == "id":
            color = id_to_color.get(pid, None)
        else:
            color = None

        if z is None:
            ax.plot(x, y, lw=1.5, alpha=0.9, color=color)
        else:
            ax.plot(x, y, z, lw=1.5, alpha=0.9, color=color)
            ax.set_zlabel(z_label)

            ax.plot(x[0], y[0], z[0], lw=0, marker="o", markersize=4, alpha=0.9, color="k")

            if p.thermalized:
                # Plot thermalization mark
                ax.plot(x[-1], y[-1], z[-1], "x", fillstyle="none", markersize=12, alpha=0.9, color=color)
        any_plotted = True

    ax.set_xlabel("x [fm]")
    ax.set_ylabel("y [fm]")

    title = "Particle trajectories"
    if getattr(hard_event, "event_id", None) is not None:
        title += f" (event_id={hard_event.event_id})"
    ax.set_title(title)

    if not any_plotted:
        ax.text(0.05, 0.95, "No trajectories to plot (empty history).", transform=ax.transAxes)

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


def plot_parton_hadron(hard_event: hard_particles.EventRecord, hadrons: pythia8.Event,
                       rap_max=1.0, N=80, max_height=3,
                       plasma_object=None, hydro_resolution=60, hydro_vel_resolution=20,
                       hydro_alpha=0.5, hydro_temp=True, hydro_flow=True):
    """
    Plot a 2D faux-collider view for the event in the xy-plane.

    Params:
        hard_event       : APE event record object -- holds particle trajectory information
        hadrons          : Pythia event object -- holds hadron information for calorimeter plots
        rap_max          : Maximum rapidity to plot. Can be None for no cut. (default: 1.0)
        plasma_object    : Optional plasma.plasma instance. If provided, temperature and/or
                           flow are rendered behind the trajectories at tau = sqrt(x^2 + y^2),
                           eta_s = 0, centred on (hard_event.event_x0, hard_event.event_y0).
        hydro_resolution : Number of grid points per side for the temperature background. (default: 60)
        hydro_vel_resolution : Number of grid points per side for the quiver flow overlay. (default: 20)
        hydro_alpha      : Opacity of the hydro background layer. (default: 0.5)
        hydro_temp       : Whether to plot the temperature background. (default: True)
        hydro_flow       : Whether to plot the flow velocity quiver. (default: True)
    """
    # Create figure
    fig = plt.figure(figsize=(7, 7))

    # Create the polar calorimeter axis first, before any colorbars are attached,
    # so that colorbar calls cannot shift the Cartesian axis and misalign the two.
    if hadrons is not None:
        paxis = fig.add_axes([0, 0, 1, 1], polar=True, frameon=False)
        paxis.grid(False)

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

        # Iterate over particles and find rmax
        rmax = 0
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

            x = np.array(traj[:, col_index["x"]]) - p.x_0
            y = np.array(traj[:, col_index["y"]]) - p.y_0
            rmax = max(rmax, np.max(np.hypot(x, y)))

        ##########################
        # Hydro background layer #
        ##########################
        if plasma_object is not None and (hydro_temp or hydro_flow):
            cx = hard_event.event_x0
            cy = hard_event.event_y0
            t_switch = config.soft_transport.hydro.T_SWITCH

            half = max(rmax, 1.0)

            # Clip mask: a filled circle of radius `half` centred at the origin.
            # All hydro artists are clipped to this circle so nothing renders outside rmax.
            hydro_clip_circle = mpatches.Circle(
                (0, 0), half, transform=axis.transData
            )

            # Temperature grid (evaluated first; mask is reused for flow)
            xs_t = np.linspace(-half, half, hydro_resolution)
            ys_t = np.linspace(-half, half, hydro_resolution)
            Xg, Yg = np.meshgrid(xs_t, ys_t)  # shape (Ny, Nx)

            tau_g = np.sqrt((Xg + cx) ** 2 + (Yg + cy) ** 2)
            tau_g = np.clip(tau_g, plasma_object.t0, plasma_object.tf)

            pts_t = np.stack([tau_g,
                              Xg + cx,
                              Yg + cy,
                              np.zeros_like(Xg)], axis=-1)
            temp_vals = plasma_object.temp(pts_t)  # shape (Ny, Nx)

            # Mask: cells where T < T_SWITCH render as white / hidden
            cold_mask = temp_vals < t_switch

            if hydro_temp:
                temp_max = plasma_object.max_temp()

                # Build a "plasma" colormap that maps masked (cold) cells to white
                base_cmap = mcm.get_cmap("plasma").copy()
                base_cmap.set_bad(color="white")

                temp_masked = np.ma.array(temp_vals, mask=cold_mask)

                num_levels = 15
                levels = np.linspace(t_switch, temp_max, num_levels)
                pcm = axis.contourf(
                    xs_t, ys_t, temp_masked,
                    levels=levels,
                    cmap=base_cmap,
                    norm=mcolors.Normalize(vmin=t_switch, vmax=temp_max),
                    alpha=hydro_alpha, zorder=0,
                )
                # Clip the entire contourf to the detector circle.
                # QuadContourSet.set_clip_path() propagates to all child artists
                # (collections were removed in Matplotlib 3.8).
                pcm.set_clip_path(hydro_clip_circle)
                # fig.colorbar(pcm, ax=axis, label="Temperature (GeV)",
                #              fraction=0.035, pad=0.02)

            # Flow quiver grid (coarser resolution)
            if hydro_flow:
                xs_v = np.linspace(-half, half, hydro_vel_resolution)
                ys_v = np.linspace(-half, half, hydro_vel_resolution)
                Xv, Yv = np.meshgrid(xs_v, ys_v)

                tau_v = np.sqrt((Xv + cx) ** 2 + (Yv + cy) ** 2)
                tau_v = np.clip(tau_v, plasma_object.t0, plasma_object.tf)

                pts_v = np.stack([tau_v,
                                  Xv + cx,
                                  Yv + cy,
                                  np.zeros_like(Xv)], axis=-1)
                vx_vals = plasma_object.x_vel(pts_v)
                vy_vals = plasma_object.y_vel(pts_v)
                vmag = np.sqrt(vx_vals ** 2 + vy_vals ** 2)

                # Remap cold_mask onto the coarser quiver grid via nearest-neighbour
                # index lookup, then blank cold arrows with NaN so quiver skips them.
                xi = np.searchsorted(xs_t, xs_v).clip(0, hydro_resolution - 1)
                yi = np.searchsorted(ys_t, ys_v).clip(0, hydro_resolution - 1)
                cold_mask_v = cold_mask[np.ix_(yi, xi)]  # shape (Nv, Nv)

                vx_vals = np.where(cold_mask_v, np.nan, vx_vals)
                vy_vals = np.where(cold_mask_v, np.nan, vy_vals)
                vmag = np.where(cold_mask_v, np.nan, vmag)

                flow_cmap = mcm.get_cmap("cool").copy()
                flow_cmap.set_bad(color="white")

                qv = axis.quiver(
                    Xv, Yv, vx_vals, vy_vals, vmag,
                    cmap=flow_cmap, norm=mcolors.Normalize(vmin=0, vmax=1),
                    alpha=hydro_alpha, zorder=0.5,
                    scale=None, pivot="mid",
                )
                qv.set_clip_path(hydro_clip_circle)
                # fig.colorbar(qv, ax=axis, label="Flow velocity (c)",
                #              fraction=0.035, pad=0.06)

        # Iterate over particles and plot
        id_to_color = {}
        j = 0
        for idx, p in enumerate(hard_event.particles):
            hist = getattr(p, "history", None)
            if p.status < 0:
                continue
            if not hist:
                continue
            if rap_max is not None and np.abs(p.rap) > rap_max:
                continue

            # Get color
            pid = p.id
            try:
                id_to_color[pid] = id_color_dict[int(pid)]
            except:
                id_to_color[pid] = cmap(j)
                j += 1

            traj = np.asarray(hist, dtype=float)
            if traj.ndim != 2 or traj.shape[1] < 4:
                raise ValueError(f"Particle {idx} has unexpected history shape: {traj.shape}")

            tau = np.array(traj[:, col_index["tau"]])
            x = np.array(traj[:, col_index["x"]]) - hard_event.event_x0
            y = np.array(traj[:, col_index["y"]]) - hard_event.event_y0

            pid = getattr(p, "id", None)
            color = id_to_color.get(pid, None)

            axis.plot(x, y, lw=1.5, alpha=0.9, color=color, zorder=2)
            axis.plot(x[0], y[0], lw=0, marker="o", markersize=4, alpha=0.9, color="k", zorder=2)
            any_plotted = True

            if p.thermalized:
                # Plot thermalization mark
                axis.plot(x[-1], y[-1], "x", fillstyle="none", markersize=12, alpha=0.9, color=color, zorder=2.9)

            # Get momentum info and compute azimuthal position of detector interaction
            px = p.px
            py = p.py
            phi = np.arctan2(py, px)
            det_x = rmax * np.cos(phi)
            det_y = rmax * np.sin(phi)

            # Plot detector interaction
            axis.plot(det_x, det_y, "o", fillstyle="none", markersize=8, alpha=0.9, color=color, zorder=2)

        if not any_plotted:
            axis.text(0, 0, "No trajectories to plot.", transform=axis.transAxes)

        #################
        # Detector rings #
        #################
        # Two concentric rings at the trajectory boundary, spacing the parton
        # endpoint markers from the base of the calorimeter histogram bars.
        ring_gap = 0.15 * max_height  # gap between the two rings, in data units
        for r_ring in (rmax, rmax + ring_gap):
            ring = mpatches.Circle(
                (0, 0), r_ring,
                fill=False, edgecolor="black", linewidth=1.2,
                zorder=3, transform=axis.transData,
            )
            axis.add_patch(ring)

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
            bottom = rmax + ring_gap  # start bars at the outer detector ring
        else:
            bottom = 4

        weight_array = np.array([])  # Array for weight of particle
        phi_array = np.array([])  # Array for azimuthal coordinate of particle
        for i in range(num_particles):
            p = hadrons[i]
            if p.isFinal():  # Only record final state particles
                if np.abs(p.y()) < rap_max:   # Only record particles in rapidity range
                    phi_array = np.append(phi_array, np.mod(p.phi(), 2*np.pi))
                    weight_array = np.append(weight_array, p.pT())

        phi_bins = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        counts, _ = np.histogram(phi_array, bins=phi_bins, weights=weight_array)
        max_counts = np.amax(counts)
        if max_counts > 0:
            radii = (max_height / max_counts) * counts
        else:
            radii = np.zeros(counts.shape)
        width = (2 * np.pi) / N

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
