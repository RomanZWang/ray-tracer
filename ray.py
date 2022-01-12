from __future__ import division
from PIL import Image
import numpy as np
from pprint import pprint
import sys


def normalize(v):
    return v/np.linalg.norm(v)


def intersect_sphere(origin, direction, sphere_center, sphere_radius):
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    # solve quadratic equation ax^2 + bx + c

    def discriminant_checker(a, b, c, epsilon=0.00001):
        return abs(b**2 - 4*a*c) < epsilon

    origin_to_sphere = origin - sphere_center
    a = np.linalg.norm(direction)**2
    b = 2 * np.dot(direction, origin_to_sphere)
    c = np.linalg.norm(origin_to_sphere)**2 - sphere_radius**2

    if discriminant_checker(a, b, c):
        return np.inf
    else:
        sols = np.roots([a, b, c])
        real_sols = sols[~np.iscomplex(sols)]
        real_positive_sols = real_sols[real_sols > 0]

        if len(real_positive_sols) > 0:
            return min(real_positive_sols)
        else:
            return np.inf


def intersect_sphere_test_case():
    sphere_intersect_test_cases = []
    sphere_intersect_test_cases.append(np.random.rand(3))
    v = np.random.rand(3)
    sphere_intersect_test_cases.append(v/np.linalg.norm(v))
    sphere_intersect_test_cases.append(np.random.rand(3))
    sphere_intersect_test_cases.append(np.random.rand())
    return sphere_intersect_test_cases


def find_intersection_t(origin, direction, object):
    if object['type'] == "sphere":
        sphere = object
        t = intersect_sphere(
            origin, direction, sphere['coordinates'], sphere['radius'])
        return t
    else:
        raise ValueError("Need to implement different intersection handler")


def get_sphere_normal(center, intersection):
    return normalize(center - intersection)


def get_normal(object, intersection):
    if object['type'] == 'sphere':
        return get_sphere_normal(object['coordinates'], intersection)
    else:
        raise ValueError("Object type not implemented")


def read_input(file_path, lights, scene, camera, file_info):

    def add_sphere(x, y, z, r, sigma=0):
        scene.append({
            'coordinates': np.array([x, y, z]),
            'radius': r,
            'color': color,
            'type': "sphere",
            'sigma': sigma
        })

    def add_sun(x, y, z, tau=99999999999):
        lights.append(
            {
                'origin': tau * np.array([x, y, z]),
                'color': color,
                'type': "sun",
            }
        )

    def add_bulb(x, y, z):
        lights.append(
            {
                'origin': np.array([x, y, z]),
                'color': color,
                'type': "bulb",
            }
        )

    def change_eye(camera, x, y, z):
        camera["eye"] = np.array([x, y, z])

    def set_file_info(file_info, width, height, file_name):
        file_info['width'] = width
        file_info['height'] = height
        file_info['file_name'] = file_name

    color = np.array([1, 1, 1])

    with open(file_path, 'r') as textfile:
        for line in textfile:
            preprocessed_line = line.replace("\n", "")
            splitLine = preprocessed_line.split(" ")

            number_of_arguments = len(splitLine)-1
            command = splitLine[0]

            if command == "sphere":
                arguments = [float(num_string) for num_string in splitLine[1:]]
                if len(arguments) != 4 and len(arguments) != 5:
                    raise ValueError(
                        "Incorrect args specified: %d" % len(arguments))
                add_sphere(*arguments)
            elif command == "color":
                arguments = [float(num_string) for num_string in splitLine[1:]]
                print(arguments)
                if len(arguments) != 3:
                    raise ValueError(
                        "Incorrect args specified: %d" % len(arguments))
                color = np.array(arguments)
            elif command == "sun":
                arguments = [float(num_string) for num_string in splitLine[1:]]
                if len(arguments) != 3:
                    raise ValueError(
                        "Incorrect args specified: %d" % len(arguments))
                add_sun(*arguments)
            elif command == "camera":
                arguments = [float(num_string) for num_string in splitLine[1:]]
                if len(arguments) != 3:
                    raise ValueError(
                        "Incorrect args specified: %d" % len(arguments))
                change_eye(camera, *arguments)
            elif command == "fisheye":
                arguments = [float(num_string) for num_string in splitLine[1:]]
                if len(arguments) != 0:
                    raise ValueError(
                        "Incorrect args specified: %d" % len(arguments))
                camera['focus_parameter'] = -4
            elif command == "panorama":
                arguments = [float(num_string) for num_string in splitLine[1:]]
                if len(arguments) != 0:
                    raise ValueError(
                        "Incorrect args specified: %d" % len(arguments))
                camera['focus_parameter'] = -0.125
            elif command == "bulb":
                arguments = [float(num_string) for num_string in splitLine[1:]]
                if len(arguments) != 3:
                    raise ValueError(
                        "Incorrect args specified: %d" % len(arguments))
                add_bulb(*arguments)
            elif command == "png":
                arguments = [int(num_string) for num_string in splitLine[1:3]]
                arguments.append(str(splitLine[3]))
                if len(arguments) != 3:
                    raise ValueError(
                        "Incorrect args specified: %d" % len(arguments))
                print("ARGUMENTS FOR PNG")
                print(arguments)
                set_file_info(file_info, *arguments)


def render(file_path, camera, file_info, debug_output='output.txt', shadows_disabled=True):
    with open(debug_output, 'wt') as out:
        image_width = file_info['width']
        image_height = file_info['height']
        file_name = file_info['file_name']
        for current_x in range(image_width):

            viewport_x = (2*current_x - image_width) / \
                max(image_height, image_width)

            for current_y in range(image_height):
                print("(x, y) pixel being rendered: ", current_x, current_y)

                viewport_y = (image_height - 2*current_y) / \
                    max(image_height, image_width)
                # print(viewport_x, ", ", viewport_y)

                origin = camera['eye']

                viewport_pixel_coordinates = np.array(
                    [viewport_x, viewport_y, camera['focus_parameter']])
                ray_origin_to_viewport_pixel = normalize(
                    viewport_pixel_coordinates - origin)

                # Intersecter
                # (origin, destination, object coords) => (object, intersection)
                scene_intersection_ts = np.array([find_intersection_t(
                    origin, ray_origin_to_viewport_pixel, current_object) for current_object in scene])

                # print("cp1.3")
                if len(scene_intersection_ts) > 0:
                    if any(scene_intersection_ts != np.inf):
                        intersected_object_index = np.argmin(
                            scene_intersection_ts)
                        intersected_object = scene[intersected_object_index]
                        intersected_object_t = scene_intersection_ts[intersected_object_index]

                        intersection_point = origin + intersected_object_t * ray_origin_to_viewport_pixel

                        current_normal = get_normal(
                            intersected_object, intersection_point)

                        perturb_sigma = intersected_object['sigma']

                        perturbation_vector = np.random.normal(
                            loc=0.0, scale=perturb_sigma, size=3) if perturb_sigma != 0 else np.array([0, 0, 0])

                        color = ambient_light

                        intersection_to_origin_vector = normalize(
                            origin - intersection_point)
                        for light in lights:
                            intersection_to_light_vector = normalize(
                                light['origin'] - intersection_point)
                            light_to_intersection_vector = normalize(
                                intersection_point - light['origin'])

                            # SHADOWS
                            light_intersection_ts = np.array([find_intersection_t(
                                origin, light_to_intersection_vector, current_object) for current_object in scene])
                            light_intersected_object_index = np.argmin(
                                light_intersection_ts)

                            if light_intersected_object_index == intersected_object_index or shadows_disabled:

                                if light['type'] == "sun":
                                    color += diffuse_light * \
                                        max(np.dot(current_normal + perturbation_vector,
                                            intersection_to_light_vector), 0) * light['color']
                                elif light['type'] == "bulb":
                                    bulb_distance = np.linalg.norm(
                                        intersection_to_light_vector)**2
                                    color += diffuse_light * \
                                        max(np.dot(current_normal + perturbation_vector,
                                            intersection_to_light_vector), 0) * light['color']/bulb_distance

                        img[image_height - current_y - 1,
                            current_x, :] = np.clip(color, 0, 1)

                        diagonistic_data = {
                            'pixel_x': current_x,
                            'pixel_y': current_y,
                            'viewport_x': viewport_x,
                            'viewport_y': viewport_y,
                            'intersection_candidates': scene_intersection_ts,
                            'intersected_obj_index': intersected_object_index,
                            'intersected_obj': intersected_object,
                            'intersected_obj_time': intersected_object_t,
                            'intersection_point': intersection_point,
                            'normal': current_normal,
                            'color': color
                        }

                    else:
                        diagonistic_data = {
                            'pixel_x': current_x,
                            'pixel_y': current_y,
                            'viewport_x': viewport_x,
                            'viewport_y': viewport_y,
                            'intersection_candidates': None,
                            'intersected_obj_index': None,
                            'intersected_obj': None,
                            'intersected_obj_time': None,
                            'intersection_point': None,
                            'normal': None,
                            'color': None
                        }

                    pprint(diagonistic_data, stream=out, depth=6)
                    diagonistic_data_keeper.append(diagonistic_data)
                else:
                    # Need to implement
                    pass
    im = Image.fromarray((255 * img).astype(np.uint8), "RGB")
    im.save(file_name)
    print(camera)


if __name__ == "__main__":

    file_name = sys.argv[1]

    camera = {
        'eye': np.array([0, 0, 0]),
        'forward': np.array([0, 0, -1]),
        'up': np.array([0, 1, 0]),
        'focus_parameter': -1
    }

    file_info = {  # DEFAULT FILE INFO AND SETTINGS
        'width': 64,
        'height': 48,
        'file_name': "fig.png",
    }

    ambient_light = .05
    diffuse_light = 1.

    diagonistic_data_keeper = []

    scene = []
    lights = []

    read_input(file_name, lights, scene, camera, file_info)

    img = np.zeros((file_info['height'], file_info['width'], 3))

    render('fig.png', camera, file_info)
