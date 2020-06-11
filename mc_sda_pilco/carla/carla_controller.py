#!/usr/bin/env python

""" A simple script to test the performance of a selected controller on a vehicle for Carla0.9.6 """
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import zmq
import carla
from random import randint
import image_converter
import time
import math

# Controllers for the vehicle
# from learning_agents.imitation.imitation_agent import ImitationAgent
from CAL_agent.CAL_controller import CAL
from measurements import Measurements

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

# High level direction commands for agents' run step input
REACH_GOAL = 0.0
GO_STRAIGHT = 5.0
TURN_RIGHT = 4.0
TURN_LEFT = 3.0
LANE_FOLLOW = 2.0


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / 20
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# message type const
RESTART = "RESTART"
NEXT_STATE = "NEXT_STATE"
STATE = "STATE"


def get_next_action_from_model(socket):
    print("receiving next action from SDA...")
    message = socket.recv_json()
    print("message received: " + str(message))

    return message["type"], message["data"]


def apply_action(action, world):
    cloudyness, precipitation, precipitation_deposits, wind_intensity, sun_azimuth_angle, sun_altitude_angle = action
    # obtain old weather
    w = world.get_weather()
    # apply changes
    new_weather = carla.WeatherParameters(
        cloudyness=max(0, min(100, w.cloudyness + cloudyness)),
        precipitation=max(0, min(100, w.precipitation + precipitation)),
        precipitation_deposits=max(0, min(100, w.precipitation_deposits + precipitation_deposits)),
        wind_intensity=max(0, min(100, w.wind_intensity + wind_intensity)),
        sun_azimuth_angle=max(0, min(360, w.sun_azimuth_angle + sun_azimuth_angle)),
        sun_altitude_angle=max(-90, min(90, w.sun_altitude_angle + sun_altitude_angle))
    )

    world.set_weather(new_weather)


def update_env(socket, world, vehicle, ped):
    print("sending the next state to SDA...")
    state = state_from_world(world, vehicle, ped)
    message = {"type": STATE, "data": state}
    socket.send_json(message)
    print("state sent")


def state_from_world(world, vehicle, ped):
    state = {"weather": [], "vehicle": [], "peds": []}

    # Populate weather state
    w = world.get_weather()
    state["weather"] = [w.cloudyness,
                        w.precipitation,
                        w.precipitation_deposits,
                        w.wind_intensity,
                        w.sun_azimuth_angle,
                        w.sun_altitude_angle]

    # Populate vehicle state
    v_location = vehicle.get_location()
    v_acceleration = vehicle.get_acceleration()
    v_velocity = vehicle.get_velocity()
    state["vehicle"] = [v_location.x, v_location.y, v_location.z,
                        v_velocity.x, v_velocity.y, v_velocity.z,
                        v_acceleration.x, v_acceleration.y, v_acceleration.z]

    # Populate pedestrian state
    p_location = ped.get_location()
    p_velocity = ped.get_velocity()
    state["peds"] = [p_location.x, p_location.y, p_location.z,
                     p_velocity.x, p_velocity.y, p_velocity.z]

    return state


def connect_mq(context, port=5555):
    print("binding MQ connection...")
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%d" % port)
    print("MQ connection established...")
    return socket


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def apply_agent_control(image, player, agent):
    """ Function to pack up the data into the necessary format for the agent to process """
    # print out the timestatmp of the image
    meas = Measurements()
    meas.update_measurements(image, player)
    # Do some processing of the image dat
    # Store processed data to pass to agent
    sensor_data = {}
    sensor_data['CameraRGB'] = image_converter.to_rgb_array(image)

    # Process next control step from the agent and execute it
    control = agent.run_step(meas, sensor_data, LANE_FOLLOW, None)

    player.apply_control(control)


def on_collision(event):
    collided_actor = event.other_actor
    impulse = event.normal_impulse
    intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
    print("### collision happened: " + collided_actor.type_id + " intensity: " + str(intensity))
    time.sleep(3)
    return


def init_env(world, actor_list):
    # Load the appropriate settings for the experiment
    bp_lib = world.get_blueprint_library()

    # Spawn the agent's vehicle into the scene
    vehicle_bp = bp_lib.find('vehicle.ford.mustang')
    vehicle_tf = carla.Transform(carla.Location(2, 100, 2), carla.Rotation(yaw=-90))
    vehicle = world.spawn_actor(vehicle_bp, vehicle_tf)
    actor_list.append(vehicle)
    # Advance to the next simulation step in the simulator to ensure car is spawned
    world.tick()

    # Spawn the collision sensor
    collision_sensor = world.spawn_actor(bp_lib.find('sensor.other.collision'),
                                         carla.Transform(), attach_to=vehicle)

    collision_sensor.listen(lambda event: on_collision(event))
    actor_list.append(collision_sensor)

    # Create RGB camera
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    # Config for CAL
    camera_bp.set_attribute('fov', '90')
    camera_rgb = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.44, z=1.2), carla.Rotation(pitch=0)),
                                   attach_to=vehicle)

    # Config for IML
    # camera_bp.set_attribute('fov', '100')
    # camera_rgb = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=2, z=1.4), carla.Rotation(pitch=-15)), attach_to=vehicle)
    actor_list.append(camera_rgb)

    # Finally, attach a controller to the vehicle
    # agent = ImitationAgent('Town01', True, vehicle)
    agent = CAL('Town01', vehicle)

    # Spawn a pedestrian in front of the car
    ped_bp = bp_lib.find('walker.pedestrian.0002')
    ped_tf = carla.Transform(carla.Location(2, 80, 2), carla.Rotation())
    ped_actor = world.spawn_actor(ped_bp, ped_tf)
    actor_list.append(ped_actor)

    world.tick()  # Advance on step to ensure
    world.wait_for_tick(seconds=2.0)  # Wait up to two seconds for step update

    return agent, vehicle, camera_rgb, ped_actor


def reset(world, vehicle, pedestrian):
    # reset vehicle
    vehicle_tf = carla.Transform(carla.Location(2, 100, 1), carla.Rotation(yaw=-90))
    vehicle.set_transform(vehicle_tf)

    # reset pedestrian
    ped_tf = carla.Transform(carla.Location(2, 80, 2), carla.Rotation())
    pedestrian.set_transform(ped_tf)

    # reset weather
    world.set_weather(carla.WeatherParameters.HardRainSunset)


def main():
    # Init MQ connection
    mq_port = 5555
    mq_context = zmq.Context()
    mq = connect_mq(mq_context, mq_port)

    actor_list = []
    # Init client connection
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    client.load_world('Town01')
    world = client.get_world()

    # Set up PYGAME window for easier visualization
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    try:
        # Listen for a RESTART message to start
        begin = False
        while not begin:
            action_type, action = get_next_action_from_model(mq)
            if action_type == RESTART:
                print("received first RESTART, initialising simulation...")
                begin = True
            else:
                print("Non RESTART message received and ignored, retry in 2 seconds...")
                time.sleep(2)

        # Initialise environment
        agent, vehicle, camera_rgb, ped = init_env(world, actor_list)
        print("simulation env initialised, actors created: %d" % len(actor_list))

        # Flag for the first tick
        first_tick = True

        # Create a synchronous mode context.
        snapshot = None
        image_rgb = None
        with CarlaSyncMode(world, camera_rgb, fps=20) as sync_mode:
            while True:
                if should_quit():
                    return
                # Handling of init case
                if (first_tick):
                    w = world.get_weather()
                    new_weather = carla.WeatherParameters(
                        cloudyness=0,
                        precipitation=0,
                        precipitation_deposits=60,
                        wind_intensity=0,
                        sun_azimuth_angle=90,
                        sun_altitude_angle=45
                    )
                    world.set_weather(new_weather)

                    for i in range(10):
                        clock.tick()
                        sync_mode.tick(timeout=10.0)

                    vehicle.set_velocity(carla.Vector3D(x=0, y=-15, z=0))

                    clock.tick()
                    snapshot, image_rgb = sync_mode.tick(timeout=10.0)
                    update_env(mq, world, vehicle, ped)
                    first_tick = False

                # Get next action from model
                action_type, action = get_next_action_from_model(mq)

                # Reset env if restart
                if action_type == RESTART:
                    reset(world, vehicle, ped)
                    # skip the next frames
                    for i in range(20):
                        clock.tick()
                        sync_mode.tick(timeout=10.0)

                    w = world.get_weather()
                    new_weather = carla.WeatherParameters(
                        cloudyness=0,
                        precipitation=0,
                        precipitation_deposits=60,
                        wind_intensity=0,
                        sun_azimuth_angle=90,
                        sun_altitude_angle=randint(15, 75)
                    )
                    world.set_weather(new_weather)

                    vehicle.set_velocity(carla.Vector3D(x=0, y=-15, z=0))

                # Otherwise, apply action then invoke vehicle controller
                else:
                    apply_action(action, world)
                    apply_agent_control(image_rgb, vehicle, agent)

                # Draw the display.
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                draw_image(display, image_rgb)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

                # Sync with simulator
                clock.tick()

                # Update env with new observation after action
                snapshot, image_rgb = sync_mode.tick(timeout=10.0)
                update_env(mq, world, vehicle, ped)

    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
