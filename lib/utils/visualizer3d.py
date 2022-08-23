import sys
if __name__ == '__main__':
    sys.path.append('./')
import pyvista
import time
import math
import numpy as np
import os
import os.path as osp
import shutil
import platform
import tempfile
import pyrender
import cv2 as cv
from lib.utils.vis import images_to_video, make_checker_board_texture, nparray_to_vtk_matrix


class Visualizer3D:

    def __init__(self, init_T=6, enable_shadow=False, anti_aliasing=True, use_floor=True,
                 add_cube=False, distance=5, elevation=20, azimuth=0, verbose=True):
        if platform.system() == 'Linux':
            print('Attention: Your OS may not support anti-aliasing, but you can try to turn in on in lib.utils.visualizer3d.__init__() to improve rendering quality.')
            anti_aliasing = False
        self.enable_shadow = enable_shadow
        self.anti_aliasing = anti_aliasing
        self.use_floor = use_floor
        self.add_cube = add_cube
        self.verbose = verbose
        self.pl = None
        self.interactive = True
        self.hide_env = False
        # animation control
        self.fr = 0
        self.num_fr = 1
        self.fps_arr = [10, 20, 30, 40, 50, 60]
        self.T_arr = [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60]
        self.T = init_T
        self.paused = False
        self.reverse = False
        self.repeat = False
        # camera
        self.distance = distance
        self.elevation = elevation
        self.azimuth = azimuth
        # background
        self.background_img = None
    
    def init_camera(self):
        # self.pl.camera_position = 'yz'
        self.pl.camera.focal_point = (0, 0, 0)
        self.pl.camera.position = (self.distance, 0, 0)
        self.pl.camera.elevation = self.elevation
        self.pl.camera.azimuth = self.azimuth
        # self.pl.camera.zoom(1.0)

    def set_camera_instrinsics(self, fx=None, fy=None, cx=None, cy=None, z_near=0.1, zfar=1000):
        wsize =  np.array(self.pl.window_size)
        if platform.system() == 'Darwin':
            wsize //= 2
            if not (fx is None and fy is None):
                fx *= 0.5
                fy *= 0.5

        if fx is None and fy is None:
            fx = fy = wsize.max()
        if cx is None and cy is None:
            cx, cy = 0.5 * wsize

        intrinsic_cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, z_near, zfar)
        proj_transform = intrinsic_cam.get_projection_matrix(*self.pl.window_size)
        self.pl.camera.SetExplicitProjectionTransformMatrix(nparray_to_vtk_matrix(proj_transform))
        self.pl.camera.SetUseExplicitProjectionTransformMatrix(1)

    def init_scene(self, init_args):
        if not self.hide_env:
            # self.pl.set_background('#DBDAD9')
            self.pl.set_background('#FCC2EB', top='#C9DFFF')    # Classic Rose -> Lavender Blue
        # shadow
        if self.enable_shadow:
            self.pl.enable_shadows()
        if self.anti_aliasing:
            self.pl.enable_anti_aliasing()
        # floor
        if self.use_floor:
            wlh = (20.0, 20.0, 0.05)
            center = np.array([0, 0, -wlh[2] * 0.5])
            self.floor_mesh = pyvista.Cube(center, *wlh)
            self.floor_mesh.t_coords *= 2 / self.floor_mesh.t_coords.max()
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#81C6EB', '#D4F1F7'))
            self.pl.add_mesh(self.floor_mesh, texture=tex, ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
        else:
            self.floor_mesh = None
        # cube
        if self.add_cube:
            self.cube_mesh = pyvista.Box()
            self.cube_mesh.points *= 0.1
            self.cube_mesh.translate((0.0, 0.0, 0.1))
            self.pl.add_mesh(self.cube_mesh, color='orange', ambient=0.2, diffuse=0.8, specular=0.8, specular_power=10, smooth_shading=True)
        
    def update_camera(self, interactive):
        pass

    def update_scene(self):
        pass

    def setup_key_callback(self):

        def close():
            exit(0)

        def slowdown():
            if self.frame_mode == 'fps':
                self.fps = self.fps_arr[(self.fps_arr.index(self.fps) - 1) % len(self.fps_arr)]
            else:
                self.T = self.T_arr[(self.T_arr.index(self.T) + 1) % len(self.T_arr)]

        def speedup():
            if self.frame_mode == 'fps':
                self.fps = self.fps_arr[(self.fps_arr.index(self.fps) + 1) % len(self.fps_arr)]
            else:
                self.T = self.T_arr[(self.T_arr.index(self.T) - 1) % len(self.T_arr)]

        def reverse():
            self.reverse = not self.reverse

        def repeat():
            self.repeat = not self.repeat

        def pause():
            self.paused = not self.paused

        def next_frame():
            if self.fr < self.num_fr - 1:
                self.fr += 1
            self.update_scene()
        
        def prev_frame():
            if self.fr > 0:
                self.fr -= 1
            self.update_scene()

        def go_to_start():
            self.fr = 0
            self.update_scene()

        def go_to_end():
            self.fr = self.num_fr - 1
            self.update_scene()

        self.pl.add_key_event('q', close)
        self.pl.add_key_event('s', slowdown)
        self.pl.add_key_event('d', speedup)
        self.pl.add_key_event('a', reverse)
        self.pl.add_key_event('g', repeat)
        self.pl.add_key_event('Up', go_to_start)
        self.pl.add_key_event('Down', go_to_end)
        self.pl.add_key_event('space', pause)
        self.pl.add_key_event('Left', prev_frame)
        self.pl.add_key_event('Right', next_frame)

    def render(self, interactive):
        self.update_camera(interactive)
        self.pl.update()
        
    def tframe_animation_loop(self):
        t = 0
        while True:
            self.render(interactive=True)
            if t >= math.floor(self.T):
                if not self.reverse:
                    if self.fr < self.num_fr - 1:
                        self.fr += 1
                    elif self.repeat:
                        self.fr = 0
                elif self.reverse and self.fr > 0:
                    self.fr -= 1
                self.update_scene()
                t = 0
            if not self.paused:
                t += 1

    def fps_animation_loop(self):
        last_render_time = time.time()
        self.update_scene()
        while True:
            while True:
                self.render(interactive=True)
                if time.time() - last_render_time >= (1 / self.fps - 0.002):
                    break
            if not self.paused:
                if not self.reverse:
                    if self.fr < self.num_fr - 1:
                        self.fr += 1
                        self.update_scene()
                    elif self.repeat:
                        self.fr = 0
                        self.update_scene()
                elif self.reverse and self.fr > 0:
                    self.fr -= 1
                    self.update_scene()
            # print('fps', 1 / (time.time() - last_render_time))
            last_render_time = time.time()

    def show_animation(self, window_size=(800, 800), init_args=None, enable_shadow=None, frame_mode='fps', fps=30, repeat=False, show_axes=True):
        self.interactive = True
        self.frame_mode = frame_mode
        self.fps = fps
        self.repeat = repeat
        if enable_shadow is not None:
            self.enable_shadow = enable_shadow
        self.pl = pyvista.Plotter(window_size=window_size)
        self.init_camera()
        self.init_scene(init_args)
        self.update_scene()
        self.setup_key_callback()
        if show_axes:
            self.pl.show_axes()
        self.pl.show(interactive_update=True)
        if self.frame_mode == 'fps':
            self.fps_animation_loop()
        else:
            self.tframe_animation_loop()

    def save_frame(self, fr, img_path):
        self.fr = fr
        self.update_scene()
        self.render(interactive=False)
        if self.background_img is not None:
            img = self.pl.screenshot(transparent_background=True, return_img=True)
            alpha = img[..., [3]] / 255.0
            alpha[alpha > 0.0] = 1.0
            fg_img = cv.cvtColor(img[..., :3], cv.COLOR_RGB2BGR)
            bg_img = cv.imread(self.background_img)
            c_img = fg_img * alpha + bg_img * (1 - alpha)
            cv.imwrite(img_path, c_img)
        else:
            self.pl.screenshot(img_path)


    def save_animation_as_video(self, video_path, init_args=None, window_size=(800, 800), enable_shadow=None, fps=30, crf=25, frame_dir=None, cleanup=True):
        self.interactive = False
        if platform.system() == 'Linux':
            pyvista.start_xvfb()
        if enable_shadow is not None:
            self.enable_shadow = enable_shadow
        self.pl = pyvista.Plotter(window_size=window_size, off_screen=True)
        self.init_camera()
        self.init_scene(init_args)
        self.pl.show(interactive_update=True)
        if frame_dir is None:
            frame_dir = tempfile.mkdtemp(prefix="visualizer3d-")
        else:
            if osp.exists(frame_dir):
                shutil.rmtree(frame_dir)
            os.makedirs(frame_dir)
        os.makedirs(osp.dirname(video_path), exist_ok=True)
        for fr in range(self.num_fr):
            self.save_frame(fr, f'{frame_dir}/{fr:06d}.jpg')
        images_to_video(frame_dir, video_path, fps=fps, crf=crf, verbose=self.verbose)
        if cleanup:
            shutil.rmtree(frame_dir)



if __name__ == '__main__':
    visualizer = Visualizer3D(add_cube=True, enable_shadow=True)
    visualizer.show_animation(show_axes=True)