"""
    Utility functionalities for handling waypoint for agents classes.
"""
from collections import deque
from enum import Enum
import random
import numpy as np
import math

try:
    import carla
    from agents.navigation.controller import VehiclePIDController
    from agents.tools.misc import distance_vehicle, draw_waypoints, vector
except ImportError:
    print("Carla library is not installed !!")


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beginning of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def calculate_step_distance(vehicle_speed, dt=1.0 / 30.0, factor=1.5):
    """
    Calculate Distance traveled between the timestamps of the prediction based on the vehicle speed.
    :param: vehicle_speed Velocity of the ego vehicle in [km/h]
    :param dt time between two predictions in seconds
    :param factor Multiplication factor for the traveld distance between two timestamps
    :return: sampling_radius based on prediction frequency and vehicle speed.
    """
    return vehicle_speed * dt * factor / 3.6


def compute_next_waypoints(current_wp, d=2, k=100, stay_on_lane=True):
    """
    Returning a trajectory queue of waypoint with the distance d to each other.

    :param stay_on_lane: True, if the vehicle should stay on the lane,
                            in case of multiple options for the next wp.
    :param current_wp: Current waypoint of the ego vehicle.
    :param k: how many waypoints to compute
    :param d: distance of two waypoints

    :return: ordered queue of waypoints, starting from the nearest.
    """

    waypoints_queue = deque(maxlen=k)

    for _ in range(k):
        next_waypoints = list(current_wp.next(d))

        if len(next_waypoints) == 1:
            # only one option available ==> lanefollowing
            next_waypoint = next_waypoints[0]
            road_option = RoadOption.LANEFOLLOW
        else:
            # random choice between the possible options
            road_options_list = _retrieve_options(
                next_waypoints, current_wp)

            if stay_on_lane:
                road_option = RoadOption.STRAIGHT
                next_waypoint = next_waypoints[0]
                # next_waypoint = next_waypoints[road_options_list.index(road_option)]
            else:
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

        waypoints_queue.append((next_waypoint, road_option))
        current_wp = next_waypoint

    return waypoints_queue


def compute_previous_waypoints(current_wp, d=2, k=100, stay_on_lane=True):
    """
    Returning a trajectory queue of previous waypoint with the distance d to each other.

    :param stay_on_lane: True, if the vehicle should stay on the lane,
                            in case of multiple options for the next wp.
    :param current_wp: Current waypoint of the ego vehicle.
    :param k: how many waypoints to compute
    :param d: distance of two waypoints

    :return: ordered queue of waypoints, starting from the nearest.
    """

    waypoints_queue = deque(maxlen=k)

    for _ in range(k):
        prev_waypoints = list(current_wp.previous(d))

        if len(prev_waypoints) == 1:
            # only one option available ==> lanefollowing
            prev_waypoint = prev_waypoints[0]
            road_option = RoadOption.LANEFOLLOW
        else:
            # random choice between the possible options
            road_options_list = _retrieve_options(
                prev_waypoints, current_wp)

            if stay_on_lane:
                road_option = RoadOption.STRAIGHT
                try:
                    prev_waypoint = prev_waypoints[road_options_list.index(
                        road_option)]
                except:
                    # Use first entry of possible previous waypoints, if road option STRAIGHT is not avaible.
                    prev_waypoint = prev_waypoints[0]
            else:
                road_option = random.choice(road_options_list)
                prev_waypoint = prev_waypoints[road_options_list.index(
                    road_option)]

        waypoints_queue.append((prev_waypoint, road_option))
        current_wp = prev_waypoint

    return waypoints_queue


def get_localization_from_waypoint(wp: carla.Waypoint):
    """
    Returning the localization object from the waypoint object.
    :param wp: Carla waypoint object
    :return: carla.Location object of the waypoint element
    """
    return wp.transform.location


def get_vehicle_velocity_vector(vehicle: carla.Vehicle, map_vehicle: carla.Map, velocity):
    """
    Function to return a velocity vector which points to the direction of the next waypoint.
    :param velocity: Desired vehicle velocity
    :param map_vehicle:  carla.Map
    :param vehicle: carla.Vehicle object
    :return: carla.Vector3D
    """

    # Getting current waypoint and next from vehicle
    current_wp = map_vehicle.get_waypoint(vehicle.get_location())
    next_wp = current_wp.next(1)[0]

    # Getting localization from the waypoints
    current_loc = get_localization_from_waypoint(current_wp)
    next_loc = get_localization_from_waypoint(next_wp)

    velocity_x = abs(next_loc.x - current_loc.x)
    velocity_y = abs(next_loc.y - current_loc.y)

    vector_vel0 = np.array([velocity_x, velocity_y, 0])
    vector_vel = (velocity / np.linalg.norm(vector_vel0)) * vector_vel0

    return carla.Vector3D(round(vector_vel[0], 3), round(vector_vel[1], 3), 0)


def euclidean_distance(loc1: carla.Location, loc2: carla.Location):
    """
    Calculating euclidean distance between two carla location points
    :param loc1: carla location 1
    :param loc2: carla location 2
    :return: distance
    """
    d = np.sqrt(np.square(loc1.x - loc2.x) + np.square(loc1.y - loc2.y))

    return d


def get_wp_vector(wp1: carla.Waypoint, wp2: carla.Waypoint):
    """
    Returns the unit vector from wp1 to wp2
    wp1, wp2:   carla.Waypoint objects
    """

    # Get location from wp if wp is a carla.Waypoint object
    location_1 = wp1.transform.location if isinstance(wp1, carla.Waypoint) else wp1
    location_2 = wp2.transform.location if isinstance(wp2, carla.Waypoint) else wp2

    x = location_2.x - location_1.x
    y = location_2.y - location_1.y

    return [x, y, 0]


def get_wp_angle(wp1: carla.Waypoint, wp2: carla.Waypoint):
    """
    Returning the orientation between two waypoints with respect to the x-axis.
    :param wp1: Carla Waypoint object 1 -->
    :param wp2: Carla Waypoint object 2
    :return: Angle [rad] between the both waypoints.
    """

    wp_vector = get_wp_vector(wp2, wp1)  # wp1 - wp2
    norm = np.linalg.norm(wp_vector) + np.finfo(float).eps

    return np.sign(-1 * wp_vector[1]) * np.arccos((-1 * np.dot(wp_vector, np.array([1, 0, 0])))/norm)


def get_distance2wp(vehicle_transform: carla.Transform, wp1: carla.Waypoint, wp2: carla.Waypoint):
    """
    Calculating a norm vector form the vehicle position to the reference line (from wp1 to wp2) and
    returning the length from this vector.

    :param vehicle_transform: carla.Transform object of the location of the ego-vehicle in world
    :param wp1: Carla Waypoint object 1
    :param wp2: Carla Waypoint object 2
    :return: Returning distance of the vehicle to the reference line (from wp1 to wp2)
    """
    wp_vector = get_wp_vector(wp2, wp1)
    xy_vector = get_wp_vector(wp2, vehicle_transform.location)

    norm = np.linalg.norm(wp_vector) + np.finfo(float).eps
    cross = np.cross(wp_vector, xy_vector)

    return np.linalg.norm(cross/norm) + np.finfo(float).eps


def get_angle2wp_line(vehicle_transform: carla.Transform, wp1: carla.Waypoint, wp2: carla.Waypoint):
    """
    Calculating a norm vector form the vehicle position to the reference line (from wp1 to wp2) and
    returning the length from this vector.

    :param vehicle_transform: carla.Transform object of the location of the ego-vehicle in world
    :param wp1: Carla Waypoint object 1
    :param wp2: Carla Waypoint object 2
    :return: Returning distance of the vehicle to the reference line (from wp1 to wp2)
    """
    wp_vector = get_wp_vector(wp2, wp1)
    xy_vector = get_wp_vector(wp2, vehicle_transform.location)

    sign = np.sign(xy_vector[1] * wp_vector[0] - xy_vector[0] * wp_vector[1])
    norm_wp = np.linalg.norm(wp_vector) + np.finfo(float).eps
    norm_xy = np.linalg.norm(xy_vector) + np.finfo(float).eps

    return sign * np.arccos(np.dot(xy_vector, wp_vector) / (norm_xy * norm_wp))


def draw_waypoints_debug(world, waypoints, z=0.5, color=None):
    """
    Draw a list of waypoints at a certain height given in z.

    :param color: RGB color of the waypoint marker
    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    if color is None:
        color = [255, 0, 0]
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, color=carla.Color(color[0], color[1], color[2]))


def draw_vehicle_bounding_box(world, location, heading, color=None):
    """
    Draw a list of waypoints at a certain height given in z.

    :param color: RGB color of the waypoint marker
    :param world: carla.world object
    :param location: carla.Location to place the center of the bounding box
    :param heading: heading of the bounding box element
    :return:
    """
    if color is None:
        color = [255, 165, 0]           # orange
    bbx = _create_bb(location)
    world.debug.draw_box(bbx, carla.Rotation(yaw=heading), 0.05, carla.Color(255,165,0,0),0)


def _create_bb(location: carla.Location, ):
  return carla.BoundingBox(location, carla.Vector3D(1, 1.9, 0.5))
