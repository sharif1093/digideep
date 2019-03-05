"""This module hacks the ``dm_control.viewer`` to add capability of viewing environments in a step-wise manner,
mostly like OpenAI Gym. It enables updating the ``physics`` from outside of the viewer by the ``env.step(action)``
command.


Caution:
    This class might break over the future updates of the ``dm_control`` package.

Caution:
    Since they are related to the "runtime" and we have already made "runtime=None", we loose the following properties of the viewer:
    * Pause/Resume
    * Speed Up/ Slow Down: FPS will be fixe:
    * Where does this 60Hz come from though?
    * Applying perturbations on the bodies.

Caution:
    Since the hack changes the ``dm_control``'s viewer globally, this module should not be used if you rely on lost
    capabilities of the viewer.
"""

##########################################################################
# TODO: Implementing an absolutedly decoupled viewer
#       which will have the above lost properties, and
#       yet is completely decoupled.
##########################################################################
# START HACKING
###############
## 1. Correct integration of nested action
from dm_control.viewer import runtime
def _get_default_action(action_spec):
    return None
runtime._get_default_action = _get_default_action


## 2. Adding a "step" function to the viewer.gui.glfw_gui
import glfw
from dm_control.viewer import gui
def view_step(self, tick_func):
    pixels = tick_func()
    # print("Shape of pixels:", pixels.shape)
    with self._context.make_current() as ctx:
        ctx.call(self._update_gui_on_render_thread, self._context.window, pixels)
    self._mouse.process_events()
    self._keyboard.process_events()
    if glfw.window_should_close(self._context.window):
        raise RuntimeError("User exited the simulation!")

gui.glfw_gui.GlfwWindow.step = view_step


## 3. Fixing the launch function in application to avoid infinite loop.
from dm_control.viewer import application
def application_launch(self, environment_loader, policy=None):
    if environment_loader is None:
      raise ValueError('"environment_loader" argument is required.')
    if callable(environment_loader):
      self._environment_loader = environment_loader
    else:
      self._environment_loader = lambda: environment_loader
    self._policy = policy
    self._load_environment(zoom_to_scene=True)
    def tick():
      self._viewport.set_size(*self._window.shape)
      self._tick()
      return self._renderer.pixels
    self.tick_func = tick

    # Make _runtim None to prevent perturbations done in the _tick function.
    self._runtime = None

def application_step(self):
    self._window.step(tick_func=self.tick_func)

application.runtime = runtime
application.gui = gui

Application = application.Application
Application.launch = application_launch
Application.step = application_step
################
# END OF HACKING
##########################################################################


class Viewer(object):
    """A class which uses the hacked viewer of the ``dm_control`` to provide a native viewer
    for dm_control environments with 3D manipulation capabilities.
    """

    def __init__(self, dmcenv, width, height, display=None):
        self.initialized = False
        self.dmcenv = dmcenv
        self.width = width
        self.height = height

        self.app = Application(title="Simulation", width=self.width, height=self.height)

    def __call__(self, **kwargs): # pixel as an argument || what to do with depth rendering??
        if not self.initialized:
            self.app.launch(environment_loader=self.dmcenv)
            self.initialized = True
        self.app.step()
    
    def close(self):
        pass
