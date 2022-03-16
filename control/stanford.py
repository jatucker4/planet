import cv2
import glob
import numpy as np
import os
import pickle
import random
import gym

from planet.humanav.humanav_renderer import HumANavRenderer
from planet.humanav.renderer_params import create_params as create_base_params
from planet.humanav.renderer_params import get_surreal_texture_dir


def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)


def create_params():
    p = create_base_params()

	# Set any custom parameters
    p.building_name = 'area5a' #'area3'

    p.camera_params.width = 1024
    p.camera_params.height = 1024
    p.camera_params.fov_vertical = 75.
    p.camera_params.fov_horizontal = 75.

    # The camera is assumed to be mounted on a robot at fixed height
    # and fixed pitch. See humanav/renderer_params.py for more information

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.camera_elevation_degree = -10

    p.camera_params.modalities = ['rgb', 'disparity']
    return p


def plot_rgb(rgb_image_1mk3, filename):
    import cv2
    # fig = plt.figure(figsize=(30, 10))

    src = rgb_image_1mk3[0].astype(np.uint8)
    src = src[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb!
    # percent by which the image is resized
    scale_percent = (32. / src.shape[0]) * 100

    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)

    # Plot the RGB Image
    # plt.imshow(output)
    # plt.imshow(rgb_image_1mk3[0].astype(np.uint8))
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_title('RGB')

    cv2.imwrite(filename, output)
    # cv2.imwrite('original.png', src)

    # fig.savefig(filename, bbox_inches='tight', pad_inches=0)

def render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=False):
    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2]/dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=False)

    depth_image_1mk1, _, _ = r._get_depth_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05, map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

    return rgb_image_1mk3, depth_image_1mk1


def generate_observation(camera_pos_13, path):
    p = create_params()

    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=False)

    camera_pos_str = '_' + str(camera_pos_13[0][0]) + '_' + str(camera_pos_13[0][1]) + '_' + str(camera_pos_13[0][2])
    filename_rgb = 'rgb' + camera_pos_str + '.png'

    # Plot the rendered images
    plot_rgb(rgb_image_1mk3, path + filename_rgb)

    return path + filename_rgb, traversible, dx_m


class Stanford_Environment_Params():
    def __init__(self):
        self.epi_reward = 100 #1000 #100
        self.step_reward = -1 #-10 #-1
        self.dim_action = 1
        self.velocity = 0.2 #0.1 #0.5
        self.dim_state = 2
        self.img_size = 32
        self.dim_obs = 4
        self.obs_std_light = 0.01
        self.obs_std_dark = 0.1
        self.step_range = 1 #0.05
        self.max_steps = 200
        self.noise_amount = 0.4 #1.0 #0.4 #0.15
        self.fig_format = '.png'
        self.img = 'data/img/'
        self.ckpt = 'data/ckpt/'

        self.training_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/training_hallway/rgbs/*"
        self.testing_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/testing_hallway/rgbs/*"
        # self.training_data_path = "/home/sampada/training_hallway/rgbs/*"
        # self.testing_data_path = "/home/sampada/testing_hallway/rgbs/*"
        self.normalization = True

sep = Stanford_Environment_Params()


class AbstractEnvironment(object):
    def __init__(self):
        if type(self) is Base:
            raise Exception('Base is an abstract class and cannot be instantiated directly')

    def get_observation(self):
        # Generates observation from the current environment state
        return

    def step(self):
        # Takes a step from the current environment state with a supplied action and updates the environment state
        return

    # # New functions to be added
    # def transition_state(self, s, a):
    #     # Given a state and an action, return the next step state and reward. All inputs and outputs are in numpy array format.
    #     return

    # def transition_tensor(self, s, a):
    #     # Given a state tensor and an action, return the next step state tensor and rewards. All inputs and outputs are in numpy array format.
    #     return

    def is_terminal(self, s):
        # Check if a given state is a terminal state
        return

    def action_sample(self):
        # Gives back a uniformly sampled random action
        return

    def reward(self, s):
        # Gives back reward for corresponding state
        return

class StanfordEnvironment(AbstractEnvironment):
    def __init__(self, disc_thetas=False):
        self.done = False
        self._step = None
        self.true_env_corner = [24.0, 23.0]
        self.xrange = [0, 8.5]
        self.yrange = [0, 1.5]
        self.thetas = [0.0, 2 * np.pi]
        self.disc_thetas = disc_thetas
        self.discrete_thetas = np.array(
            [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4])
        self.discrete_thetas = self.discrete_thetas.reshape((len(self.discrete_thetas), 1))
        self.trap_x = [[1.5, 2], [6.5, 7]]
        self.trap_y = [0, 0.25]
        self.target_x = [4, 4.5]
        self.target_y = [0, 0.25]
        self.init_strip_x = self.xrange
        self.init_strip_y = [0.25, 0.5]
        self.state, self.orientation = self.initial_state()
        self.dark_line = (self.yrange[0] + self.yrange[1]) / 2
        self.dark_line_true = self.dark_line + self.true_env_corner[1]

        # Get the traversible
        try:
            traversible = pickle.load(open("traversible.p", "rb"))
            dx_m = 0.05
        except Exception:
            path = os.getcwd() + '/temp/'
            os.mkdir(path)
            _, _, traversible, dx_m = self.get_observation(normalize=False)
            pickle.dump(traversible, open("traversible.p", "wb"))

        # if traversible_dump:
        #     path = os.getcwd() + '/temp/'
        #     os.mkdir(path)
        #     _, _, traversible, dx_m = self.get_observation(path=path)
        #     pickle.dump(traversible, open("traversible.p", "wb"))
        # else:
        #     traversible = pickle.load(open("traversible.p", "rb"))
        #     dx_m = 0.05

        self.traversible = traversible
        self.dx = dx_m
        self.map_origin = [0, 0]

        # For making training batches
        self.training_data_path = sep.training_data_path
        self.training_data_files = glob.glob(self.training_data_path)
        self.testing_data_path = sep.testing_data_path
        self.testing_data_files = glob.glob(self.testing_data_path)
        self.normalization = sep.normalization

        # TODO: Unclean code!
        # For training the generator with noisy images
        # Pre-generate the corrupted indices in the image
        # Noise in the image plane
        self.generator_is_training = False
        self.diff_pattern = False
        s_vs_p = 0.5
        image_plane_size = 32 * 32
        num_salt = np.ceil(sep.noise_amount * image_plane_size * s_vs_p)
        num_pepper = np.ceil(sep.noise_amount * image_plane_size * (1. - s_vs_p))
        if self.diff_pattern:  # Same noise pattern in all dark images or a different noise pattern per image?
            # Pre-generate the corrupted indices per image in the training data
            self.noise_list = []
            for i in range(len(self.training_data_files)):
                self.noise_list.append(np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False))
        else:
            self.noise_indices = np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False)

    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

    def observation_space(self):
        # low = np.zeros([64, 64, 3], dtype=np.float32)
        low = np.ones([32, 32, 3], dtype=np.float32)*(-10)
        high = np.ones([32, 32, 3], dtype=np.float32)*10
        spaces = {'image': gym.spaces.Box(low, high)}
        return gym.spaces.Dict(spaces)

    def reset(self):
        self.done = False
        self._step = 0
        self.state, self.orientation = self.initial_state()
        _, img_path, _, _ = self.get_observation(normalize=False)
        obs = self.read_observation(img_path)
        return obs
        # self.state = np.random.rand(sep.dim_state)
        # self.orientation = np.random.rand()
        # self.state[0] = self.state[0] * (self.init_strip_x[1] - self.init_strip_x[0]) + self.init_strip_x[0]
        # self.state[1] = self.state[1] * (self.init_strip_y[1] - self.init_strip_y[0]) + self.init_strip_y[0]
        # self.orientation = self.orientation * (self.thetas[1] - self.thetas[0]) + self.thetas[0]

    def initial_state(self):
        if self.disc_thetas:
            orientation = self.discrete_thetas[np.random.randint(len(self.discrete_thetas))][0]
        else:
            orientation = np.random.rand()
            orientation = orientation * (self.thetas[1] - self.thetas[0]) + self.thetas[0]

        valid_state = False
        while not valid_state:
            state = np.random.rand(sep.dim_state)
            temp = state[1]
            state[0] = state[0] * (self.init_strip_x[1] - self.init_strip_x[0]) + self.init_strip_x[0]
            state[1] = temp * (self.trap_y[1] - self.trap_y[0]) + self.trap_y[0]  # Only consider x for in_trap
            if not self.in_trap(state) and not self.in_goal(state):
                valid_state = True
                state[1] = temp * (self.init_strip_y[1] - self.init_strip_y[0]) + self.init_strip_y[0]

        return state, orientation

    def noise_image(self, image, state, noise_amount=sep.noise_amount):
        salt = 255
        pepper = 0

        out = image

        if state[1] <= self.dark_line:  # Dark observation - add salt & pepper noise
            # if True:
            s_vs_p = 0.5
            amount = noise_amount
            out = np.copy(image)
            num_salt = np.ceil(amount * image.size * s_vs_p)
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            noise_indices = np.random.choice(image.size, int(num_salt + num_pepper), replace=False)
            salt_indices = noise_indices[:int(num_salt)]
            pepper_indices = noise_indices[int(num_salt):]
            salt_coords = np.unravel_index(salt_indices, image.shape)
            pepper_coords = np.unravel_index(pepper_indices, image.shape)
            out[salt_coords] = salt
            out[pepper_coords] = pepper

        # cv2.imwrite("out_debug.png", out)

        return out

    def noise_image_plane(self, image, state, noise_amount=sep.noise_amount, img_index=None):
        # Corrupts the R, G, and B channels of noise_amount * (32 x 32) pixels

        salt = 255
        pepper = 0

        out = image

        image_plane_size = image.shape[0] * image.shape[1]
        image_plane_shape = (image.shape[0], image.shape[1])
        if state[1] <= self.dark_line:  # Dark observation - add salt & pepper noise
            s_vs_p = 0.5
            amount = noise_amount
            out = np.copy(image)
            num_salt = np.ceil(amount * image_plane_size * s_vs_p)
            num_pepper = np.ceil(amount * image_plane_size * (1. - s_vs_p))
            if self.generator_is_training:
                if self.diff_pattern:
                    noise_indices = self.noise_list[img_index]
                else:
                    noise_indices = self.noise_indices
            else:
                noise_indices = np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False)
            salt_indices = noise_indices[:int(num_salt)]
            pepper_indices = noise_indices[int(num_salt):]
            salt_coords = np.unravel_index(salt_indices, image_plane_shape)
            pepper_coords = np.unravel_index(pepper_indices, image_plane_shape)
            for i in range(len(salt_coords[0])):  # salt_coords[0] is row indices, salt_coords[1] is col indices
                row = salt_coords[0][i]
                col = salt_coords[1][i]
                for j in range(3):
                    out[row, col, j] = salt
            for i in range(len(pepper_coords[0])):  # pepper_coords[0] is row indices, pepper_coords[1] is col indices
                row = pepper_coords[0][i]
                col = pepper_coords[1][i]
                for j in range(3):
                    out[row, col, j] = pepper

        # cv2.imwrite("out_debug1.png", out)

        return out

    def get_observation(self, state=None, normalize=True, normalization_data=None):
        if state == None:
            state_temp = self.state
            state = self.state + self.true_env_corner
            state_arr = np.array([[state[0], state[1], self.orientation]])
        else:
            state_temp = state
            state = state + self.true_env_corner
            state_arr = np.array([state])

        path = os.getcwd() + '/images/'
        # os.mkdir(path)
        check_path(path)

        img_path, traversible, dx_m = generate_observation(state_arr, path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        out = self.noise_image_plane(image, state_temp)
        out = out[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb! --- now image is in RGB
        out = np.ascontiguousarray(out)

        if normalize:
            rmean, gmean, bmean, rstd, gstd, bstd = normalization_data
            img_rslice = (out[:, :, 0] - rmean) / rstd
            img_gslice = (out[:, :, 1] - gmean) / gstd
            img_bslice = (out[:, :, 2] - bmean) / bstd

            out = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

            # out = (out - out.mean())/out.std()  # "Normalization" -- TODO

        os.remove(img_path)
        os.rmdir(path)

        return out, img_path, traversible, dx_m

    # Currently not used
    def read_observation(self, img_path, normalize):
        obs = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # obs = cv2.imread(img_path, cv2.IMREAD_COLOR)
        obs = obs[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb! --- now obs is in RGB
        # if normalize:
        #     obs = (obs - obs.mean())/obs.std()  # "Normalization" -- TODO

        # obs = obs * 100 + (255./2)
        # obs = obs[:, :, ::-1]
        # cv2.imwrite(img_path[:-4] + str, obs)

        os.remove(img_path)

        return obs

    def point_to_map(self, pos_2, cast_to_int=True):
        """
        Convert pos_2 in real world coordinates
        to a point on the map.
        """
        map_pos_2 = pos_2 / self.dx - self.map_origin
        if cast_to_int:
            map_pos_2 = map_pos_2.astype(np.int32)
        return map_pos_2

    def detect_collision(self, state):
        # Returns true if you collided
        # Map value 0 means there's an obstacle there

        # Don't hit rectangle boundaries
        if state[0] < self.xrange[0] or state[0] > self.xrange[1]:
            return True
        if state[1] < self.yrange[0] or state[1] > self.yrange[1]:
            return True

        # Check if state y-value is the same as trap/goal but it's not in the trap or goal - that's a wall
        if self.in_trap([self.trap_x[0][0], state[1]]) and \
                not self.in_trap(state) and not self.in_goal(state):
            return True

        map_state = self.point_to_map(np.array(state[:sep.dim_state] + self.true_env_corner))
        map_value = self.traversible[map_state[1], map_state[0]]
        collided = (map_value == 0)
        return collided

    def in_trap(self, state):
        # Returns true if in trap
        first_trap = self.trap_x[0]
        first_trap_x = (state[0] >= first_trap[0] and state[0] <= first_trap[1])
        second_trap = self.trap_x[1]
        second_trap_x = (state[0] >= second_trap[0] and state[0] <= second_trap[1])
        trap_x = first_trap_x or second_trap_x

        trap = trap_x and (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1])

        return trap and not self.in_goal(state)

    def in_goal(self, state):
        # Returns true if in goal
        goal = (state[0] >= self.target_x[0] and state[0] <= self.target_x[1]) and \
               (state[1] >= self.target_y[0] and state[1] <= self.target_y[1])
        return goal

    def step(self, action, action_is_vector=False):
        self.done = False
        curr_state = self.state
        
        ## Get the observation at the current state to provide PlaNet the expected output
        _, img_path, _, _ = self.get_observation(normalize=False)
        obs = self.read_observation(img_path)


        if action_is_vector:
            new_theta = np.arctan2(action[1], action[0])
            if new_theta < 0:  # Arctan stuff
                new_theta += 2 * np.pi
            next_state = curr_state + action
        else:
            new_theta = action[0] * np.pi + np.pi
            vector = np.array(
                [np.cos(new_theta), np.sin(new_theta)]) * sep.velocity  # Go in the direction the new theta is
            next_state = curr_state + vector

        cond_hit = self.detect_collision(next_state)

        if self.in_goal(next_state):
            self.state = next_state
            self.orientation = new_theta
            self.done = True
        elif cond_hit == False:
            self.state = next_state
            self.orientation = new_theta
        reward = sep.epi_reward * self.done

        cond_false = self.in_trap(next_state)
        reward -= sep.epi_reward * cond_false
        info = {}
        return obs, reward, self.done, info

    ########## Methods for tree search ###########

    def action_sample(self):
        # Gives back a uniformly sampled random action
        rnd = int(random.random() * 9)

        # No blank move
        while rnd == 4:
            rnd = int(random.random() * 9)

        action = sep.velocity * np.array([(rnd % 3) - 1, (rnd // 3) - 1])

        # # just generate completely random
        # action_x = STEP_RANGE * (2 * random.random() - 1)
        # action_y = STEP_RANGE * (2 * random.random() - 1)
        # action = np.array([action_x, action_y])

        return action

    def is_terminal(self, s):
        # Check if a given state tensor is a terminal state
        terminal = [self.in_goal(state) for state in s]
        return all(terminal)

        # s = s[:, :2]
        # targets = np.tile(self.target, (s.shape[0], 1))

        # true_dist = l2_distance_np(s, targets)

        # return all(true_dist <= sep.end_range)

    def transition(self, s, w, a):
        '''
        transition each state in state tensor s with actions in action/action tensor a
        s: [num_par, dim_state]
        a: [dim_state]  (It's a vector)
        w: [num_par]
        Needs to return a next_state that includes orientation, even though the
        current orientation is not given
        TODO: Should probably fix that
        '''

        if w is not None:
            weights = np.copy(w)
            next_weights = np.copy(w)
        else:
            # Dummy weight
            weights = np.ones(np.shape(s)[0])
            next_weights = np.ones(np.shape(s)[0])
        sp = s[:, :sep.dim_state] + a
        action_angle = np.arctan2(a[1], a[0])
        if action_angle < 0:  # Arctan stuff
            action_angle += 2 * np.pi
        orientations_next = np.tile(action_angle, (len(sp), 1))  # [num_par, 1]
        reward = 0.0

        cond_hit = np.array([self.detect_collision(state) for state in sp])  # [num_par]
        # Don't transition the states that are going to collide (but change their orientations)
        next_state = np.copy(sp)
        next_state[cond_hit, :2] = s[cond_hit, :2]
        next_state = np.concatenate((next_state, orientations_next), -1)  # [num_par, dim_state + 1]

        goal_achieved = np.array([self.in_goal(state) for state in sp])  # [num_par]
        # If goal reached
        next_weights[goal_achieved] = 0.0
        reward += np.sum(weights[goal_achieved]) * sep.epi_reward

        trap = np.array([self.in_trap(state) for state in sp])  # [num_par]
        # If trap reached
        reward -= np.sum(weights[trap]) * sep.epi_reward

        # Penalize taking a step (collision or not doesn't matter)
        normal_step = ~(goal_achieved | trap)
        reward += np.sum(weights[normal_step]) * sep.step_reward

        # Is the transition terminal?
        is_terminal = all(goal_achieved)

        if is_terminal:
            # Dummy weight
            next_weights = np.array([1 / len(next_weights)] * len(next_weights))
        else:
            # Reweight
            next_weights = next_weights / np.sum(next_weights)

        return next_state, next_weights, reward, is_terminal

    def distance_to_goal(self, state):
        '''
        Returns vector that gets the state x, y to the goal (rectangular) and
        the length of that vector
        '''
        x = state[0]
        y = state[1]

        if self.in_goal(state):
            return np.array([0, 0]), 0

        # Shortest distance to goal is to horizontal edges of rectangle
        if x >= self.target_x[0] and x <= self.target_x[1]:
            y_dist_argmin = np.argmin([abs(self.target_y[1] - y), abs(self.target_y[0] - y)])
            vec_dist = [self.target_y[1] - y, self.target_y[0] - y]
            vec = np.array([0, vec_dist[y_dist_argmin]])

        # Shortest distance to goal is to vertical edges of rectangle
        elif y >= self.target_y[0] and y <= self.target_y[1]:
            x_dist_argmin = np.argmin([abs(self.target_x[1] - x), abs(self.target_x[0] - x)])
            vec_dist = [self.target_x[1] - x, self.target_x[0] - x]
            vec = np.array([vec_dist[x_dist_argmin], 0])

        # Shortest distance to goal is to one of the goal corners
        else:
            corners = np.array([[self.target_x[0], self.target_y[0]],
                                [self.target_x[0], self.target_y[1]],
                                [self.target_x[1], self.target_y[0]],
                                [self.target_x[1], self.target_y[1]]])
            corner_argmin = np.argmin([np.linalg.norm(np.array([x, y]) - corner) for corner in corners])
            corner_argmin = corners[corner_argmin]
            vec = corner_argmin - np.array([x, y])

        return vec, np.linalg.norm(vec)


    ########## For (pre)training - making batches ##########

    def shuffle_dataset(self):
        data_files_indices = list(range(len(self.training_data_files)))
        np.random.shuffle(data_files_indices)

        return data_files_indices

    def preprocess_data(self):
        # For normalizing the images - per channel mean and std
        print("Preprocessing the data")

        try:
            normalization_data = pickle.load(open("data_normalization.p", "rb"))
            rmean, gmean, bmean, rstd, gstd, bstd = normalization_data
            print("Done preprocessing")
            return rmean, gmean, bmean, rstd, gstd, bstd
        except Exception:
            rmean = 0
            gmean = 0
            bmean = 0
            rstd = 0
            gstd = 0
            bstd = 0
            for i in range(len(self.training_data_files)):
                img_path = self.training_data_files[i]
                src = cv2.imread(img_path, cv2.IMREAD_COLOR)
                src = src[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB

                rmean += src[:, :, 0].mean() / len(self.training_data_files)
                gmean += src[:, :, 1].mean() / len(self.training_data_files)
                bmean += src[:, :, 2].mean() / len(self.training_data_files)

                rstd += src[:, :, 0].std() / len(self.training_data_files)  ## TODO: FIX?
                gstd += src[:, :, 1].std() / len(self.training_data_files)
                bstd += src[:, :, 2].std() / len(self.training_data_files)

            normalization_data = [rmean, gmean, bmean, rstd, gstd, bstd]
            pickle.dump(normalization_data, open("data_normalization.p", "wb"))

            print("Done preprocessing")

            return rmean, gmean, bmean, rstd, gstd, bstd

    def get_training_batch(self, batch_size, data_files_indices, epoch_step,
                           normalization_data, num_particles,
                           noise_amount=sep.noise_amount,
                           percent_blur=0, blur_kernel=3):

        rmean, gmean, bmean, rstd, gstd, bstd = normalization_data

        states = []
        orientations = []
        images = []
        remove = 4
        rounding = 3

        if (epoch_step + 1) * batch_size > len(
                data_files_indices):  # If amount of training data not divisible by batch size
            indices = data_files_indices[epoch_step * batch_size:]
        else:
            indices = data_files_indices[epoch_step * batch_size:(epoch_step + 1) * batch_size]

        num_blurred_images = int(percent_blur * len(indices))
        blur = np.zeros(len(indices))
        blur_indices = np.random.choice(len(indices), num_blurred_images, replace=False)
        blur[blur_indices] = 1.

        for i in range(len(indices)):
            index = indices[i]
            img_path = self.training_data_files[index]

            splits = img_path[:-remove].split('_')
            state = np.array([np.round(float(elem), rounding) for elem in splits[-(sep.dim_state + 1):]])
            state[:sep.dim_state] = state[:sep.dim_state] - self.true_env_corner
            states.append(state[:sep.dim_state])
            orientations.append(state[sep.dim_state])

            src = cv2.imread(img_path, cv2.IMREAD_COLOR)  # src is now in BGR

            # For training the generator: pass in the index of the image in the dataset
            src = self.noise_image_plane(src, state, noise_amount, index)

            if blur[i] == 1:
                blurred = cv2.GaussianBlur(src, (blur_kernel, blur_kernel), cv2.BORDER_DEFAULT)
                src = blurred[:, :, ::-1]
            else:
                src = src[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB

            if self.normalization:
                img_rslice = (src[:, :, 0] - rmean) / rstd
                img_gslice = (src[:, :, 1] - gmean) / gstd
                img_bslice = (src[:, :, 2] - bmean) / bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)
            else:
                src = (src - src.mean()) / src.std()
                images.append(src)

            if state[1] < self.dark_line:
                obs_std = sep.obs_std_dark
            else:
                obs_std = sep.obs_std_light

            par_vec_x = np.random.normal(state[0], obs_std, num_particles)
            par_vec_y = np.random.normal(state[1], obs_std, num_particles)

            middle_var = np.stack((par_vec_x, par_vec_y), 1)

            if i == 0:
                par_batch = middle_var
            else:
                par_batch = np.concatenate((par_batch, middle_var), 0)

        return np.array(states), np.array(orientations), np.array(images), par_batch

    def get_testing_batch(self, batch_size, normalization_data):
        rmean, gmean, bmean, rstd, gstd, bstd = normalization_data

        states = []
        orientations = []
        images = []
        blurred_images = []
        remove = 4
        rounding = 3

        indices = np.random.choice(range(len(self.testing_data_files)), batch_size, replace=False)

        for index in indices:

            img_path = self.testing_data_files[index]

            splits = img_path[:-remove].split('_')
            state = np.array([np.round(float(elem), rounding) for elem in splits[-(sep.dim_state + 1):]])
            state[:sep.dim_state] = state[:sep.dim_state] - self.true_env_corner
            states.append(state[:sep.dim_state])
            orientations.append(state[sep.dim_state])

            src = cv2.imread(img_path, cv2.IMREAD_COLOR)
            src = self.noise_image_plane(src, state)

            blurred = cv2.GaussianBlur(src, (5, 5), cv2.BORDER_DEFAULT)
            src = src[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB
            blurred = blurred[:, :, ::-1]

            if self.normalization:
                img_rslice = (src[:, :, 0] - rmean) / rstd
                img_gslice = (src[:, :, 1] - gmean) / gstd
                img_bslice = (src[:, :, 2] - bmean) / bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                images.append(img)

                img_rslice = (blurred[:, :, 0] - rmean) / rstd
                img_gslice = (blurred[:, :, 1] - gmean) / gstd
                img_bslice = (blurred[:, :, 2] - bmean) / bstd

                img = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

                blurred_images.append(img)
            else:
                src = (src - src.mean()) / src.std()
                images.append(src)

                blurred = (blurred - blurred.mean()) / blurred.std()
                blurred_images.append(blurred)

        return np.array(states), np.array(orientations), np.array(images), np.array(blurred_images)
