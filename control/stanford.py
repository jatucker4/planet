import cv2
import glob
import numpy as np
import os
import pickle
import random
import gym
from collections import OrderedDict
from planet.control.abstract import AbstractEnvironment
from planet.examples.examples import *
#from planet.humanav.humanav_renderer import HumANavRenderer
#from planet.humanav.renderer_params import create_params as create_base_params
#from planet.humanav.renderer_params import get_surreal_texture_dir

def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)


# def create_params():
#     p = create_base_params()

# 	# Set any custom parameters
#     p.building_name = 'area5a' #'area3'

#     p.camera_params.width = 1024
#     p.camera_params.height = 1024
#     p.camera_params.fov_vertical = 75.
#     p.camera_params.fov_horizontal = 75.

#     # The camera is assumed to be mounted on a robot at fixed height
#     # and fixed pitch. See humanav/renderer_params.py for more information

#     # Tilt the camera 10 degree down from the horizontal axis
#     p.robot_params.camera_elevation_degree = -10

#     p.camera_params.modalities = ['rgb', 'disparity']
#     return p


# def plot_rgb(rgb_image_1mk3, filename):
#     import cv2
#     # fig = plt.figure(figsize=(30, 10))

#     src = rgb_image_1mk3[0].astype(np.uint8)
#     src = src[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb!
#     # percent by which the image is resized
#     scale_percent = (32. / src.shape[0]) * 100

#     width = int(src.shape[1] * scale_percent / 100)
#     height = int(src.shape[0] * scale_percent / 100)

#     # dsize
#     dsize = (width, height)

#     # resize image
#     output = cv2.resize(src, dsize)

#     # Plot the RGB Image
#     # plt.imshow(output)
#     # plt.imshow(rgb_image_1mk3[0].astype(np.uint8))
#     # ax.set_xticks([])
#     # ax.set_yticks([])
#     # ax.set_title('RGB')

#     cv2.imwrite(filename, output)
#     # cv2.imwrite('original.png', src)

#     # fig.savefig(filename, bbox_inches='tight', pad_inches=0)

# def render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=False):
#     # Convert from real world units to grid world units
#     camera_grid_world_pos_12 = camera_pos_13[:, :2]/dx_m

#     # Render RGB and Depth Images. The shape of the resulting
#     # image is (1 (batch), m (width), k (height), c (number channels))
#     rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=False)

#     depth_image_1mk1, _, _ = r._get_depth_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05, map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

#     return rgb_image_1mk3, depth_image_1mk1


# def generate_observation(camera_pos_13, path):
#     p = create_params()

#     r = HumANavRenderer.get_renderer(p)
#     dx_cm, traversible = r.get_config()

#     # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
#     dx_m = dx_cm/100.

#     rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=False)

#     camera_pos_str = '_' + str(camera_pos_13[0][0]) + '_' + str(camera_pos_13[0][1]) + '_' + str(camera_pos_13[0][2])
#     filename_rgb = 'rgb' + camera_pos_str + '.png'

#     # Plot the rendered images
#     plot_rgb(rgb_image_1mk3, path + filename_rgb)

#     return path + filename_rgb, traversible, dx_m


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
        self.occlusion_amount = 15
        self.salt_vs_pepper = 0.5
        self.fig_format = '.png'
        self.img = 'data/img/'
        self.ckpt = 'data/ckpt/'

        #self.training_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/training_hallway/rgbs/*"
        #self.testing_data_path = "/home/ext_drive/sampada_deglurkar/vae_stanford/testing_hallway/rgbs/*"
        # self.training_data_path = "/home/sampada/training_hallway/rgbs/*"
        # self.testing_data_path = "/home/sampada/testing_hallway/rgbs/*"
        self.training_data_path = "/home/sampada/vae_stanford/training_hallway/rgbs/*"
        self.testing_data_path = "/home/sampada/vae_stanford/testing_hallway/rgbs/*"
        self.normalization = True

sep = Stanford_Environment_Params()

class StanfordEnvironment(AbstractEnvironment):
    def __init__(self, disc_thetas=False):
        self.done = False
        self._step = None
        self.true_env_corner = [24.0, 23.0] 
        self.xrange = [0, 8.5]
        self.yrange = [0, 1.5]
        self.thetas = [0.0, 2*np.pi]
        self.disc_thetas = disc_thetas
        self.discrete_thetas = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        self.discrete_thetas = self.discrete_thetas.reshape((len(self.discrete_thetas), 1))
        self.trap_x = [[1.5, 2], [6.5, 7]] 
        self.trap_y = [0, 0.25]
        self.target_x = [4, 4.5] 
        self.target_y = [0, 0.25]

        # During test time - have an additional trap region (optional)
        self.test_trap = False
        self.test_trap_is_random = False
        self.test_trap_x = [[3, 3.5], [5, 5.5]] #[[1.5, 2], [6.5, 7]] #[3.5, 5]
        self.test_trap_y = [[0.5, 1], [0.5, 1]] #[0.5, 1] #[0.75, 1.25]

        self.init_strip_x = self.xrange 
        self.init_strip_y = [0.25, 0.5]
        self.state, self.orientation = self.initial_state()
        self.dark_line = (self.yrange[0] + self.yrange[1])/2
        self.dark_line_true = self.dark_line + self.true_env_corner[1]

        # Get the traversible
        try:
            traversible = pickle.load(open("/home/jtucker/planet/planet/planet/traversible.p", "rb"))
            dx_m = 0.05
        except Exception:
            path = os.getcwd() + '/temp/'
            os.mkdir(path)
            # _, _, traversible, dx_m = self.get_observation(path=path)
            pickle.dump(traversible, open("traversible.p", "wb"))


        self.traversible = traversible
        self.dx = dx_m
        self.map_origin = [0, 0]

        # For making training batches
        self.training_data_path = sep.training_data_path
        self.training_data_files = glob.glob(self.training_data_path)
        self.testing_data_path = sep.testing_data_path
        self.testing_data_files = glob.glob(self.testing_data_path)
        self.normalization = sep.normalization


    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones([64, 64, 3], dtype=np.float32)*-10
        high = np.ones([64, 64, 3], dtype=np.float32)*10
        spaces = {'image': gym.spaces.Box(low, high)}
        return gym.spaces.Dict(spaces)
    
    def set_test_trap(self, test_trap_is_random=False):
        self.test_trap = True
        self.test_trap_is_random = test_trap_is_random

    def reset(self):
        self.done = False
        self._step = 0
        self.state, self.orientation = self.initial_state()

        # Randomizing the test traps: 
        # Each trap is size 0.5 by 0.5
        # Trap 1 will be randomly placed between x = 0 and 4 (goal x)
        # Trap 2 will be randomly placed between x = 4.5 and 8 (end of goal x to end of xrange - 0.5)
        # Each trap will be randomly placed between y = 0.5 and 0.75 (dark line)

        if self.test_trap and self.test_trap_is_random:
            trap_size = 0.5
            trap1_x = np.random.rand() * (self.target_x[0] - self.xrange[0]) + self.xrange[0]
            trap2_x = np.random.rand() * (self.xrange[1]-trap_size - self.target_x[1]) + self.target_x[1]
            self.test_trap_x = [[trap1_x, trap1_x+trap_size], [trap2_x, trap2_x+trap_size]] 

            trap1_y = np.random.rand() * (self.dark_line - self.init_strip_y[1]) + self.init_strip_y[1]
            trap2_y = np.random.rand() * (self.dark_line - self.init_strip_y[1]) + self.init_strip_y[1]
            self.test_trap_y = [[trap1_y, trap1_y+trap_size], [trap2_y, trap2_y+trap_size]]

        #normalization_data = self.preprocess_data()
        #obs_nav, _, _, _ = self.get_observation()
        obs = OrderedDict()        
        obs['image'] = obs_nav
        return obs

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

        if state[1] <= self.dark_line: # Dark observation - add salt & pepper noise
        #if True:
            s_vs_p = sep.salt_vs_pepper
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
        
        #cv2.imwrite("out_debug.png", out)

        return out

    def noise_image_plane(self, image, state, noise_amount=sep.noise_amount, noise_indices=None):
        # Corrupts the R, G, and B channels of noise_amount * (32 x 32) pixels

        salt = 255
        pepper = 0

        out = image

        image_plane_size = image.shape[0] * image.shape[1]
        image_plane_shape = image.shape[:2]
        if state[1] <= self.dark_line: # Dark observation - add salt & pepper noise
            s_vs_p = sep.salt_vs_pepper
            amount = noise_amount  
            out = np.copy(image)
            num_salt = np.ceil(amount * image_plane_size * s_vs_p)
            num_pepper = np.ceil(amount * image_plane_size * (1. - s_vs_p))
            # We may provide pre-generated noise indices when the generator is training
            if noise_indices is None:
                noise_indices = np.random.choice(image_plane_size, int(num_salt + num_pepper), replace=False) 
            
            salt_indices = noise_indices[:int(num_salt)]
            pepper_indices = noise_indices[int(num_salt):]
            #salt_coords = np.unravel_index(salt_indices, image_plane_shape)
            #pepper_coords = np.unravel_index(pepper_indices, image_plane_shape)
            salt_coords = (np.array([int(elem) for elem in salt_indices/image_plane_shape[1]]), 
                            salt_indices%image_plane_shape[1])
            pepper_coords = (np.array([int(elem) for elem in pepper_indices/image_plane_shape[1]]), 
                            pepper_indices%image_plane_shape[1])
            
            if num_salt != 0:
                out[salt_coords[0], salt_coords[1], :] = salt
            # for i in range(len(salt_coords[0])):  # salt_coords[0] is row indices, salt_coords[1] is col indices
            #     row = salt_coords[0][i]
            #     col = salt_coords[1][i]
            #     for j in range(3):
            #         out[row, col, j] = salt
            if num_pepper != 0:
                out[pepper_coords[0], pepper_coords[1], :] = pepper
            # for i in range(len(pepper_coords[0])):  # pepper_coords[0] is row indices, pepper_coords[1] is col indices
            #     row = pepper_coords[0][i]
            #     col = pepper_coords[1][i]
            #     for j in range(3):
            #         out[row, col, j] = pepper
        
        #cv2.imwrite("out_debug1.png", out)

        return out
    
    def noise_image_occlusion(self, image, state, occlusion_amount=sep.occlusion_amount):
         # Turns a randomly chosen occlusion_amount x occlusion_amount square in the image to black 

         out = image

         if state[1] <= self.dark_line: # Dark observation - add occlusion
             amount = occlusion_amount

             start_index_x = np.random.randint(image.shape[1] - occlusion_amount)  # column index
             start_index_y = np.random.randint(image.shape[0] - occlusion_amount)  # row index
             start_point = (start_index_x, start_index_y)
             end_point = (start_index_x + amount, start_index_y + amount)
             color = (0, 0, 0)
             thickness = -1

             out = np.copy(image)
             out = cv2.rectangle(out, start_point, end_point, color, thickness)

         #cv2.imwrite("out_debug2.png", out)

         return out
    
    def get_observation(self, state=None, normalize=True, normalization_data=None, occlusion=False):
        if state == None:
            state_temp = self.state
            state = self.state + self.true_env_corner
            state_arr = np.array([[state[0], state[1], self.orientation]])
        else:
            state_temp = state
            state = state + self.true_env_corner
            state_arr = np.array([state])

        path = os.getcwd() + '/images/' 
        #os.mkdir(path)
        check_path(path)

        img_path, traversible, dx_m = generate_observation(state_arr, path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if occlusion:
            out = self.noise_image_occlusion(image, state_temp)
        else:
            out = self.noise_image_plane(image, state_temp)

        out = out[:, :, ::-1]  ## CV2 works in BGR space instead of RGB!! So dumb! --- now image is in RGB
        out = np.ascontiguousarray(out)

        if normalize:
            #rmean, gmean, bmean, rstd, gstd, bstd = normalization_data
            #img_rslice = (out[:, :, 0] - rmean)/rstd
            #img_gslice = (out[:, :, 1] - gmean)/gstd
            #img_bslice = (out[:, :, 2] - bmean)/bstd

            #out = np.stack([img_rslice, img_gslice, img_bslice], axis=-1)

            out = (out - out.mean())/out.std()  # "Normalization" -- TODO

        os.remove(img_path)
        os.rmdir(path)

        return out, img_path, traversible, dx_m

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
        # if self.in_trap([self.trap_x[0][0], state[1]]) and \
        #     not self.in_trap(state) and not self.in_goal(state):
        #     return True
        if (state[1] >= self.trap_y[0] and state[1] <= self.trap_y[1]) and \
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

        # Traps that may appear during test time -- optional
        if self.test_trap:
            first_test_trap = self.test_trap_x[0]
            first_test_trap_x = (state[0] >= first_test_trap[0] and state[0] <= first_test_trap[1])
            second_test_trap = self.test_trap_x[1]
            second_test_trap_x = (state[0] >= second_test_trap[0] and state[0] <= second_test_trap[1])
            #test_trap_x = first_test_trap_x or second_test_trap_x

            first_test_trap = self.test_trap_y[0]
            first_test_trap_y = (state[1] >= first_test_trap[0] and state[1] <= first_test_trap[1])
            second_test_trap = self.test_trap_y[1]
            second_test_trap_y = (state[1] >= second_test_trap[0] and state[1] <= second_test_trap[1])
            
            test_trap = (first_test_trap_x and first_test_trap_y) or \
                        (second_test_trap_x and second_test_trap_y)

            # test_trap = test_trap_x and (state[1] >= self.test_trap_y[0] and state[1] <= self.test_trap_y[1])
            
            return (trap or test_trap) and not self.in_goal(state)
        
        return trap and not self.in_goal(state)
    
    def in_goal(self, state):
        # Returns true if in goal
        goal = (state[0] >= self.target_x[0] and state[0] <= self.target_x[1]) and \
                (state[1] >= self.target_y[0] and state[1] <= self.target_y[1])
        return goal

    def step(self, action, action_is_vector=False):
        self.done = False
        curr_state = self.state

        # Get the observation at the current state to provide PlaNet the expected output
        #normalization_data = self.preprocess_data()
        #obs_nav, _, _, _ = self.get_observation()
        obs = OrderedDict()        
        obs['image'] = obs_nav
        
        if action_is_vector:
            new_theta = np.arctan2(action[1], action[0])
            if new_theta < 0:  # Arctan stuff
                new_theta += 2*np.pi
            next_state = curr_state + action
        else:
            new_theta = action[0] * np.pi + np.pi
            vector = np.array([np.cos(new_theta), np.sin(new_theta)]) * sep.velocity  # Go in the direction the new theta is
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
                src = src[:,:,::-1]   ## CV2 works in BGR space instead of RGB!! So dumb! --- now src is in RGB
                
                rmean += src[:, :, 0].mean()/len(self.training_data_files)
                gmean += src[:, :, 1].mean()/len(self.training_data_files)
                bmean += src[:, :, 2].mean()/len(self.training_data_files)
                
                rstd += src[:, :, 0].std()/len(self.training_data_files)  ## TODO: FIX?
                gstd += src[:, :, 1].std()/len(self.training_data_files)
                bstd += src[:, :, 2].std()/len(self.training_data_files)
            
            normalization_data = [rmean, gmean, bmean, rstd, gstd, bstd]
            pickle.dump(normalization_data, open("data_normalization.p", "wb"))

            print("Done preprocessing")

            return rmean, gmean, bmean, rstd, gstd, bstd

    
