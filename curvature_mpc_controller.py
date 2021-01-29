#!/usr/bin/env python3

#
# authors: Michael Seegerer (michael.seegerer@tum.de)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a model predictive controller to perform low-level waypoint following. """
import glob
import os
import sys
import time

from enum import Enum
from collections import deque
import random
import numpy as np
import mpctools as mpc
from scipy.optimize import curve_fit

try:
    # sys.path.append(glob.glob('/opt/carla-simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #     sys.version_info.major,
    #     sys.version_info.minor,
    #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    import carla

    from agents.navigation.controller import VehiclePIDController
    from agents.tools.misc import distance_vehicle, draw_waypoints, get_speed
except ImportError:
    print("Carla libary is not installed !!")

from agents.navigation.model_predictive_controller import CurvMPCController, converting_mpc_u_to_control, wrap2pi
from agents.navigation.waypoint_utilities import *


def rotmat(angle_rad):
    """
    Computes a 2x2 rotation matrix
    :param angle_rad: Rotation angle in rad
    :return: 2x2 rotation matrix
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, s],
                     [-s, c]])


def polynomial3(x, p0, p1, p2, p3):
    """
    3th-polynomial function for modeling a street trajectory.
        function: y = p3 * x³ + p2 * x² + p1 * x + p0
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return p3 * x ** 3 + p2 * x ** 2 + p1 * x + p0


def polynomial3_prime(x, p1, p2, p3):
    """
    First derivation of 3th-polynomial function for modeling a street trajectory.
        function: y = 3 * p3 * x² + 2 * p2 * x + p01
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return 3 * p3 * x ** 2 + 2 * p2 * x + p1


def polynomial3_prime2(x, p2, p3):
    """
    Second derivation of 3th-polynomial function for modeling a street trajectory.
        function: y = 6 * p3 * x + 2 * p2 * x
    --------
    :param x:
    :param p0, p1, p2, p3:
    :return: y
    """
    return 6 * p3 * x + 2 * p2


def func_kappa(x, p_arg):
    """
    Curvature function of a waypoint reference line.
    :param x: longitude value
    :param p_arg: np.array of fitted polynomial factors --> [p0, p1, p2, p3]
    :return: curvature of reference line at point x.
    """
    result = polynomial3_prime2(x, p_arg[2], p_arg[3])
    denominator = 1 + polynomial3_prime(x, p_arg[1], p_arg[2], p_arg[3]) ** 2
    denominator = denominator ** 1.5

    return result / denominator



class VehicleCurvMPC(object):
    """
    Model Predictive Controller implements the basic behavior of following a trajectory of waypoints that is generated
     on-the-fly.

    When multiple paths are available (intersections) this controller makes a random choice.
    """

    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._last_control = np.array([0, 0])
        self._opt_dict = opt_dict
        self.curv_args = np.array([0, 0, 0, 0])
        self.curv_x0 = 0

        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=30)
        self._buffer_size = 26
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # define size of states, inputs, etc., for Casadi
        self.Nx =  9 # Number of states
        self.Nu = 2  # Number of Inputs
        self.Nt = 10  # Number of Steps

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        #self._dt = 1.0 / 30.0
        self._dt = 0.2
        self._target_speed = 80.0  # Km/h
        self._sampling_radius = calculate_step_distance(self._target_speed, 1/30,
                                                        factor=10)  # factor 11 --> prediction horizon 10 steps
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        self.data_log = dict()

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = calculate_step_distance(self._target_speed, opt_dict['sampling_radius'],
                                                                factor=5)  # factor 11 --> prediction horizon 10 steps

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        # fill waypoint trajectory queue
        self._waypoints_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=2000)

        # Set Vehicle controller
        state_dim = {'Nx': self.Nx, 'Nu': self.Nu, 'Nt': self.Nt}
        self._vehicle_controller = CurvMPCController(vehicle=self._vehicle, dt=self._dt, args_state_dimension=state_dim)

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def get_state(self):
        return self._vehicle_controller.state

    def run_step(self,timestep:int,  debug=True, log=False):
        """
        Execute one step of classic mpc controller which follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        start_time = time.time()

        # not enough waypoints in the horizon? => Sample new ones
        self._sampling_radius = calculate_step_distance(get_speed(self._vehicle), 0.2,
                                                        factor=2)  # factor 11 --> prediction horizon 10 steps
        if self._sampling_radius < 2:
            self._sampling_radius = 3

        # Getting future waypoints
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._next_wp_queue = compute_next_waypoints(self._current_waypoint, d=self._sampling_radius, k=20)
        # Getting waypoint history --> history somehow starts at last wp of future wp
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._previous_wp_queue = compute_previous_waypoints(self._current_waypoint, d=self._sampling_radius, k=5)

        # concentrate history, current waypoint, future
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._waypoint_buffer.extendleft(self._previous_wp_queue)
        self._waypoint_buffer.append((self._map.get_waypoint(self._vehicle.get_location()), RoadOption.LANEFOLLOW))
        self._waypoint_buffer.extend(self._next_wp_queue)

        self._waypoints_queue = self._next_wp_queue

        # If no more waypoints in queue, returning emergency braking control
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
            print("Applied emergency control !!!")

            return control


        # target waypoint for Frenet calculation
        self.wp1 = self._map.get_waypoint(self._vehicle.get_location())
        self.wp2, _ = self._next_wp_queue[0]

        # current vehicle waypoint
        self.kappa_log = dict()
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.curv_args, self.kappa_log = self.calculate_curvature_func_args(log=True)
        self.curv_x0 = func_kappa(0, self.curv_args)

        # input for MPC controller --> wp_current, wp_next, kappa
        target_wps = [self.wp1, self.wp2, self.curv_x0]

        # move using MPC controller
        if log:
            if timestep / 30 > 8 and timestep / 30 < 9:
                # Setting manual control
                manual_control = [self.data_log.get('u_acceleration'), 0.01]
                target_wps.append(manual_control)
                print(manual_control)

                # apply manual control
                control, state, u, x_log, u_log, _ = self._vehicle_controller.mpc_control(target_wps,
                                                                                          self._target_speed,
                                                                                          solve_nmpc=False, manual=True, log=log)
                self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
                                 'Eta': state[5],
                                 'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
                                 'pred_states': [x_log],
                                 'pred_control': [u_log], 'computation_time': time.time() - start_time,
                                 "kappa": self.curv_x0, "curvature_radius": 1 / self.curv_x0}

            elif timestep / 30 > 14 and timestep / 30 < 14.6:
                # Setting manual control
                manual_control = [self.data_log.get('u_acceleration'), -0.02]
                target_wps.append(manual_control)

                # apply manual control
                control, state, u, x_log, u_log, _ = self._vehicle_controller.mpc_control(target_wps,
                                                                                          self._target_speed,
                                                                                          solve_nmpc=False,
                                                                                          manual=True, log=log)
                self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3],
                                 'Xi': state[4],
                                 'Eta': state[5],
                                 'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
                                 'pred_states': [x_log],
                                 'pred_control': [u_log], 'computation_time': time.time() - start_time,
                                 "kappa": self.curv_x0, "curvature_radius": 1 / self.curv_x0}

            else:
                if timestep % 6 == 0:
                    control, state, u, u_log, x_log, _ = self._vehicle_controller.mpc_control(target_wps, self._target_speed, solve_nmpc=True, log=log)
                    # Updating logging information of the logger
                    self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
                                     'Eta': state[5],
                                     'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
                                     'pred_states': [x_log],
                                     'pred_control': [u_log], 'computation_time': time.time() - start_time, "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}
                else:
                    control, state, prediction, u = self._vehicle_controller.mpc_control(target_wps,self._target_speed, solve_nmpc=False, log=log)
                    # Updating logging information of the logger
                    self.data_log = {'X': state[0], 'Y': state[1], 'PSI': state[2], 'Velocity': state[3], 'Xi': state[4],
                                     'Eta': state[5],
                                     'Theta': state[6], 'u_acceleration': u[0], 'u_steering_angle': u[1],
                                     'pred_states': [prediction],
                                     'pred_control': self.data_log.get("pred_control"), 'computation_time': time.time() - start_time,
                                     "kappa": self.curv_x0, "curvature_radius": 1/self.curv_x0}

        else:
            if timestep % 6 == 0:
                control = self._vehicle_controller.mpc_control(target_wps, self._target_speed,solve_nmpc=True, log=False)
            else:
                control,_,  _, _ = self._vehicle_controller.mpc_control(target_wps, self._target_speed,
                                                                         solve_nmpc=False, log=log)


        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()

        if debug:
            last_wp, _ = self._waypoint_buffer[len(self._waypoint_buffer)-1]
            draw_waypoints(self._vehicle.get_world(), [
                self.wp1, self.wp2], self._vehicle.get_location().z + 1.0)
            draw_waypoints_debug(self._vehicle.get_world(), [
                last_wp], self._vehicle.get_location().z + 1.0, color=(0, 255, 0))


        return control, self.data_log, self.kappa_log if log else control

    def calculate_curvature_func_args(self, log=False):
        """
        Function to calculate the curvature function arguments.
        This is done by fitting all reference waypoint from the waypoint buffer to 3th-polynomial function.
        :return: [p0, p1, p2, p3] of the 3th polynomial.
        """
        kappa_log = dict()
        # Current waypoint as reference point to center all waypoints
        ref_point = np.array([self.wp1.transform.location.x, self.wp1.transform.location.y])

        # Rotation matrix to align current waypoint to x-axis
        angle_wp = get_wp_angle(self.wp1, self.wp2)
        R = rotmat(angle_wp)

        # Getting reduced waypoint matrix
        wp_mat = np.zeros([self._buffer_size, 2])
        wp_mat_0 = np.zeros([self._buffer_size, 2])  # saving original locations of wps for debug

        counter_mat = 0
        for i in range(self._buffer_size):
            wp = self._waypoint_buffer[i][0]
            loc = get_localization_from_waypoint(wp)
            point = np.array([loc.x, loc.y])  # init point
            point = point.T - ref_point.T  # center traj to origin
            point = mpc.mtimes(R, point)  # rotation of point ot allign with x-axis

            wp_mat_0[counter_mat] = np.array([loc.x, loc.y])
            wp_mat[counter_mat] = point
            counter_mat += 1

        # Getting optimal parameters for 3th polynomial by fitting the a curve
        # Doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        p_opt, _ = curve_fit(polynomial3, wp_mat[:, 0], wp_mat[:, 1])

        # Logging all necessary information
        if log:
            kappa_log = {
                'wp_mat': [wp_mat],
                'wp_mat_0': [wp_mat_0],
                'refernce_point': [ref_point],
                'rotations_mat': [R],
                'angle_wp': [angle_wp],
                'p_opt': [p_opt]
            }
            print(kappa_log)

        return p_opt, kappa_log


