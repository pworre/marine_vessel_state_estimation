import numpy as np
import matplotlib.pyplot as plt


def plot_vessel_motion(states: np.ndarray) -> None:
    """
    Plot the motion of the vessel.

    Parameters
    ----------
    states : ndarray, shape (N, 6)
        History of vessel states:

        [x, y, psi, u, v, r]

        x, y   : position [m]
        psi    : heading [rad]
        u, v   : body-frame velocities [m/s]
        r      : yaw rate [rad/s]
    """

    states = np.asarray(states)

    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(
            "states must have shape (N, 6): "
            "[x, y, psi, u, v, r]"
        )

    # Extract states
    x = states[:, 0]
    y = states[:, 1]
    psi = states[:, 2]
    u = states[:, 3]
    v = states[:, 4]
    r = states[:, 5]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ---------------------------------------------------------------
    # Position / trajectory
    # ---------------------------------------------------------------

    axes[0, 0].plot(y, x)
    axes[0, 0].plot(y[0], x[0], "o", label="Start")
    axes[0, 0].plot(y[-1], x[-1], "x", label="End")

    axes[0, 0].set_xlabel("East [m]")
    axes[0, 0].set_ylabel("North [m]")
    axes[0, 0].set_title("Vessel trajectory")
    axes[0, 0].axis("equal")
    axes[0, 0].grid()
    axes[0, 0].legend()

    # ---------------------------------------------------------------
    # Heading
    # ---------------------------------------------------------------

    time = np.arange(len(states))

    axes[0, 1].plot(time, np.rad2deg(psi))

    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Heading [deg]")
    axes[0, 1].set_title("Heading")
    axes[0, 1].grid()

    # ---------------------------------------------------------------
    # Velocities
    # ---------------------------------------------------------------

    axes[1, 0].plot(time, u, label="u")
    axes[1, 0].plot(time, v, label="v")

    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Velocity [m/s]")
    axes[1, 0].set_title("Body-frame velocities")
    axes[1, 0].grid()
    axes[1, 0].legend()

    # ---------------------------------------------------------------
    # Yaw rate
    # ---------------------------------------------------------------

    axes[1, 1].plot(time, np.rad2deg(r))

    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].set_ylabel("Yaw rate [deg/s]")
    axes[1, 1].set_title("Yaw rate")
    axes[1, 1].grid()

    plt.tight_layout()
    plt.show()