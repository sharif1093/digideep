"""
The MIT License

Copyright (c) 2017 OpenAI (http://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import os
from digideep.utility.logging import logger
from digideep.environment.common.vec_env import VecEnvWrapper
from gym.wrappers.monitoring import video_recorder


class VecVideoRecorder(VecEnvWrapper):
    """
    Wrap VecEnv to record rendered image as mp4 video.
    """

    def __init__(self, venv, directory, record_video_trigger, video_length=200):
        """
        # Arguments
            venv: VecEnv to wrap
            directory: Where to save videos
            record_video_trigger:
                Function that defines when to start recording.
                The function takes the current number of step,
                and returns whether we should start recording or not.
            video_length: Length of recorded video
        """

        VecEnvWrapper.__init__(self, venv)
        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory): os.mkdir(self.directory)

        self.file_prefix = "vecenv"
        self.file_infix = '{}'.format(os.getpid())
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0

    def reset(self):
        obs = self.venv.reset()

        self.start_video_recorder()

        return obs

    def start_video_recorder(self):
        self.close_video_recorder()

        base_path = os.path.join(self.directory, '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.step_id))
        self.video_recorder = video_recorder.VideoRecorder(
                env=self.venv,
                base_path=base_path,
                metadata={'step_id': self.step_id}
                )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        return self.record_video_trigger(self.step_id)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                logger.info("Saving video to ", self.video_recorder.path)
                self.close_video_recorder()
        elif self._video_enabled():
                self.start_video_recorder()

        return obs, rews, dones, infos

    def close_video_recorder(self):
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 0

    def close(self):
        VecEnvWrapper.close(self)
        self.close_video_recorder()

    def __del__(self):
        self.close()
