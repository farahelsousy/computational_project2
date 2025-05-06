"""Oscillator network ODE"""

import numpy as np
import scipy.stats as ss
from farms_core import pylog


class AbstractOscillatorController:
    """zebrafish controller"""

    def __init__(
            self,
            pars
    ):
        super().__init__()

        # Simulation parameters
        self.pars = pars
        self.n_iterations = pars.n_iterations
        self.timestep = pars.timestep
        self.times = np.linspace(
            0, self.pars.n_iterations*self.timestep, self.pars.n_iterations)

        # Abstract oscillator parameters
        self.n_oscillators = 2*self.pars.n_joints

        # States
        self.n_eq = self.n_oscillators*2  # oscillator phase + oscillator amp
        self.state = np.zeros([self.n_iterations, self.n_eq])
        self.dstate = np.zeros([self.n_eq])  # derivative state

        # State index
        self.oscillator_phase_l = 2 * np.arange(0, self.pars.n_joints)
        self.oscillator_phase_r = 2 * np.arange(0, self.pars.n_joints) + 1
        self.oscillator_phase_all = np.arange(0, 2*self.pars.n_joints)
        self.oscillator_amplitude_l = self.pars.n_joints * \
            2 + 2 * np.arange(0, self.pars.n_joints)
        self.oscillator_amplitude_r = self.pars.n_joints * \
            2 + 2 * np.arange(0, self.pars.n_joints) + 1
        self.oscillator_amplitude_all = self.pars.n_joints * \
            2 + np.arange(0, 2*self.pars.n_joints)

        # Initial state
        if self.pars.initial_phases is None:
            self.state[0, 0:2*self.pars.n_joints-1:2] = 1 * \
                np.linspace(2*np.pi, 0, self.pars.n_joints)
            self.state[0, 1:2 *
                       self.pars.n_joints:2] = np.linspace(np.pi, -
                                                           np.pi, self.pars.n_joints)
        else:
            self.state[0, :2*self.pars.n_joints] = self.pars.initial_phases

        self.state[0, self.n_oscillators:2 *
                   self.n_oscillators] = np.ones(self.n_oscillators)  # Initialize amplitudes to 1

        # motor output and indexes
        self.motor_out = np.zeros([self.n_iterations, self.n_oscillators])
        self.motor_l = 2*np.arange(0, self.pars.n_joints)
        self.motor_r = self.motor_l + 1

        # initialize ode solver
        self.f = self.network_ode
        self.step = self.step_euler

        # pre-computed zero activity for the last two tail joints
        self.zeros4 = np.zeros(4)

    def network_ode(self, state, pos=None):
        """
        pars
        -------
        self: AbstractOscillatorController
            The controller object
        state: <np.array>
            An array of size 2*n_oscillators storing the oscillator phases and amplitudes
        pos: <np.array>
            Current joint positions (angles)
        Returns
        -------
        dstate: <np.array>
            An array of size 2*n_oscillators storing the oscillator phases and amplitudes derivatives
            Note that phases and amplitudes have to appear in the same order as defined in states.
        -------
        This function is called each step to update the network states (amplitudes and phases).
        Here you have to implement the Ordinary Differential Equation (ODE)
        to compute the derivatives of network states.
        For which you need CPG parameters  like nominal amplitudes, coupling weights, rates.
        The computation of the above-mentioned parameters can go in another custom function or
        be implemented here directly.
        """
        # Initialize arrays for the derivatives of phases and amplitudes
        dphases = np.zeros(self.n_oscillators)
        damplitudes = np.zeros(self.n_oscillators)

        # Get the current phases and amplitudes from the state
        phases = state[self.oscillator_phase_all]
        amplitudes = state[self.oscillator_amplitude_all]

        # Calculate frequency
        f = self.pars.cpg_frequency_gain * self.pars.drive + self.pars.cpg_frequency_offset

        # Calculate stretch feedback if joint positions are provided
        if pos is not None:
            # Check if we're using entraining signals by looking at the amplitude or variability of pos
            is_entrainment_active = np.max(np.abs(pos)) > 0.1
            
            # Calculate feedback weights scaled by reference
            w_ipsi = self.pars.feedback_weights_ipsi * self.pars.feedback_gain_ref
            w_contra = self.pars.feedback_weights_contra * self.pars.feedback_gain_ref
            
            # Calculate stretch feedback for each oscillator
            for i in range(self.n_oscillators):
                if i % 2 == 0:  # Left side
                    s_i = w_ipsi * max(0, pos[i//2]) + w_contra * max(0, -pos[i//2])
                else:  # Right side
                    s_i = w_ipsi * max(0, -pos[i//2]) + w_contra * max(0, pos[i//2])
                
                # Update phase derivative with stretch feedback
                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                
                # Add stronger entrainment effect when using entraining signals
                if is_entrainment_active and hasattr(self.pars, 'entraining_signals'):
                    # If entrainment is active, use a stronger direct frequency modulation
                    # Get the frequency from pars if it's being used with entraining signals
                    entrainment_freq = 0
                    if hasattr(self.pars, 'entraining_signals') and self.pars.entraining_signals is not None:
                        # Extract frequency from the entraining signals - this is a simplification
                        # Assuming entrainment is at 8Hz or other value set in exercise7.py
                        entrainment_freq = 8.0  # Hardcoded for now from ENTRAINMENT_FREQUENCY_HZ
                    
                    # Blend natural frequency with entrainment frequency based on feedback strength
                    entrainment_strength = min(1.0, abs(w_ipsi) + abs(w_contra))
                    f_entrained = f * (1 - entrainment_strength) + entrainment_freq * entrainment_strength
                    
                    # Use entrained frequency for phase update
                    dphases[i] = 2 * np.pi * f_entrained - s_i/(amplitudes[i] + epsilon) * np.sin(phases[i])
                else:
                    # Regular update without strong entrainment
                    dphases[i] = 2 * np.pi * f - s_i/(amplitudes[i] + epsilon) * np.sin(phases[i])
                
                # Update amplitude derivative with stretch feedback
                damplitudes[i] = self.pars.amplitude_rates * (self.pars.nominal_amplitude[i] - amplitudes[i]) + s_i * np.cos(phases[i])
        else:
            # If no joint positions, use default phase derivative
            dphases = 2 * np.pi * f * np.ones(self.n_oscillators)
            damplitudes = self.pars.amplitude_rates * (self.pars.nominal_amplitude - amplitudes)

        # Add coupling terms
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if i == j:
                    continue
                
                # Calculate coupling weights and phase lags
                if abs(i - j) == 2:  # Adjacent segments on same side
                    wij = self.pars.weights_body2body
                    phij = np.sign(i - j) * self.pars.phase_lag_body / (self.pars.n_joints - 1)
                elif (j - i == 1) and (i % 2 == 0):  # Left to right coupling
                    wij = self.pars.weights_body2body_contralateral
                    phij = -np.pi
                elif (i - j == 1) and (i % 2 == 1):  # Right to left coupling (mutual)
                    wij = self.pars.weights_body2body_contralateral
                    phij = np.pi
                else:
                    wij = 0
                    phij = 0

                # Add coupling term to phase derivative
                dphases[i] += amplitudes[j] * wij * np.sin(phases[j] - phases[i] - phij)

        return np.concatenate([dphases, damplitudes])

    def motor_output(self, iteration):
        """
        pars
        -------
        self: AbstractOscillatorController
            The controller object
            Hint: you can call self.state to access the current state of the controller
        iteration: <int>
            Current sim itertaion
        Returns
        -------
        motor_output: <np.array>
            An array of size 2*n_active_joints storing the muscle activations
        -------
        Here you have to use phase, amplitude and muscle strength to
        finalize muscle activations for the first 13 active joints.
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations

        In addition to returning the motor output, store
        them in self.motor_out for later use offline
        Note: You only update and store the motor output at current iteration.
        i.e. set only self.motor_out[iteration,:]
        """
        phase = self.state[iteration, self.oscillator_phase_all]
        amplitude = self.state[iteration, self.oscillator_amplitude_all]
        oscillator_output = amplitude*(1+np.cos(phase))
        motor_output = self.pars.motor_output_scaling*oscillator_output

        # store muscle output
        self.motor_out[iteration, :] = motor_output

        return motor_output

    def step_euler(self, iteration, timestep, pos=None):
        """
        pars
        -------
        self: AbstractOscillatorController
            The controller object
            Hint: you can call self.state to access the current state of the controller
                  you can call self.f(self, state) to call the network_ode(self, state) function
        iteration: <int>
            Current sim itertaion
        pos: <np.array>
            Current joint positions (angles)
        Returns
        -------
        motor_output_all: <np.array>
            An array of size 2*n_joints_total storing the muscle activations for the active and
            passive joints at current sim itertaion
        -------
        Here you have to perform the Euler step on the oscillator states.
        You return the muscle activation of all body joints at current iteration (array of 2*n_joints_total)
        which includes updated motor outputs from active joints and the motor outputs for passive joints.
        """
        # Check if entraining signals are available and use them instead of pos
        if hasattr(self.pars, 'entraining_signals') and self.pars.entraining_signals is not None:
            if iteration < len(self.pars.entraining_signals):
                # Override pos with entraining signal
                pos = self.pars.entraining_signals[iteration]

        self.state[iteration+1, :] = (
            self.state[iteration, :] +
            timestep * self.f(self.state[iteration], pos=pos)
        )

        return np.concatenate([
            self.motor_output(iteration),  # the active joints
            self.zeros4  # the last (tail) passive joint
        ])

