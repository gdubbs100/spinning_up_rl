import time
import numpy as np
import matplotlib.pyplot as plt

class PID_controller:

    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp # proportional gain
        self.Ki = Ki # Integral gain
        self.Kd = Kd # Derivative gain

        self.integral = 0
        self.prev_error = 0
    
    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = (
            self.Kp * error + 
            self.Ki * self.integral + 
            self.Kd * derivative
        )

        self.prev_error = error
        return np.array([output]), error, self.integral, derivative


class ThermostatEnv:
    def __init__(self, ambient_temp=20.0, initial_temp=20.0, heater_power=0.1):
        self.ambient_temp = ambient_temp
        self.temp = initial_temp
        self.heater_power = heater_power  # Max heating effect per time step
        self.heat_loss_rate = 0.01        # How quickly the room cools to ambient

    def step(self, heater_output, dt):
        # Clamp the heater output to [0, 1]
        # heater_output = max(0.0, min(1.0, heater_output))

        # Heating effect
        heating = heater_output * self.heater_power

        # Cooling effect
        cooling = self.heat_loss_rate * (self.temp - self.ambient_temp)

        # Update room temperature
        self.temp += (heating - cooling) * dt
        return self.temp

if __name__ == "__main__":
    target_temp = 25.0
    dt = 0.1
    steps = 1000

    env = ThermostatEnv()
    pid = PID_controller(Kp=1.0, Ki=1.0, Kd=.1)

    temps = []
    control_signals = []
    errors = []
    integrals = []
    derivatives = []

    for _ in range(steps):
        current_temp = env.temp
        control_signal, error, integral, derivative = pid.update(target_temp, current_temp, dt)
        # heater_output = max(0.0, min(1.0, control_signal))
        temp = env.step(control_signal, dt)
        temps.append(temp)
        control_signals.append(control_signal)
        errors.append(error)
        integrals.append(integral)
        derivatives.append(derivative)

    # breakpoint()
    # Plot
    fig, ax = plt.subplots(2, 3, figsize = (10, 7), sharex=True)
    ax = ax.flatten()
    ax[0].plot(range(steps), temps, label = 'Temperature')
    ax[0].axhline(target_temp, color='r', linestyle='--', label="Target Temp")
    ax[0].set_ylabel("Temperature")
    ax[0].set_title("PID-Thermostat")
    ax[0].set_xlabel("Time Steps")
    ax[0].grid()

    ax[1].plot(range(steps), control_signals, label = 'control signals')
    ax[1].set_ylabel("Control Signals")
    ax[1].set_xlabel("Time Steps")
    ax[1].grid()

    ax[2].plot(range(steps), errors, label = 'errors')
    ax[2].set_ylabel("Errors")
    ax[2].set_xlabel("Time Steps")
    ax[2].grid()

    ax[3].plot(range(steps), integrals, label='integrals')
    ax[3].set_ylabel("Integrals")
    ax[3].set_xlabel("Time Steps")
    ax[3].grid()

    ax[4].plot(range(steps), derivatives, label = 'derivatives')
    ax[4].set_ylabel("Derivatives")
    ax[4].set_xlabel("Time Steps")
    ax[4].grid()
    
    # plt.title("PID-Controlled Thermostat")
    # plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.show()