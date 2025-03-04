import asyncio
import os
from time import time
from random import randint, random
from PIL import Image
import io

from models.session import Session
from services.message_sender import MessageSender
from models.messages import *
from utils.frame_loader import FrameLoader


class VideoStreamer:
    
    def __init__(
        self,
        session: Session,
        message_sender: MessageSender,
        base_path: str="videos",
    ):
        self._settings = None
        self._loop_task = None
        self.session = session
        self.message_sender = message_sender
        self._tasks = []
        
        self._video_dirs = [
            file_path
            for file_path in [os.path.join(base_path, d) for d in os.listdir(base_path)] 
            if os.path.isdir(file_path)
        ]
        print(self._video_dirs)


    def begin_streaming(self):
        if not self.session.is_connected:
            print("Cannot stream video before connection")
            return
        if self._tasks:
            print("Already streaming video")
            return

        self.session.add_listener(self._on_session_changed)
        self._update_streams()


    def end_streaming(self):
        self.session.remove_listener(self._on_session_changed)

        for task in self._tasks:
            task.cancel()

        self._tasks.clear()


    def _on_session_changed(self):
        if self._settings != self.session.stream_settings:
            self._update_streams()
    

    def _update_streams(self):
        settings = self.session.stream_settings
        self._settings = settings

        n_tasks = len(self._tasks)
                
        if n_tasks < settings.n_streams:
            self._tasks.extend([
                asyncio.create_task(self._stream(i))
                for i in range(n_tasks, settings.n_streams)
            ])
        elif n_tasks > settings.n_streams:
            for _ in range(settings.n_streams, n_tasks):
                self._tasks.pop().cancel()


    async def _stream(self, index):
        print(f"Stream {index} started")

        while True:
            video_index = randint(-1, len(self._video_dirs) - 1)
            if video_index == -1:
                try:
                    await self._send_black_frames(index)
                    continue
                except:
                    print(f"Stream {index} finished")
                    return

            video_dir = self._video_dirs[video_index]

            with FrameLoader() as frame_loader:
                frame_index = 0

                while True:
                    settings = self._settings
                    original_fps = 30
                    fps = min(original_fps, settings.fps)
                    required_waiting_time = 1.0 / fps
                    resolution = self._settings.resolution

                    new_frame_index = frame_loader.prefetch_with_params(
                        frame_dir=os.path.join(video_dir, f"{resolution}x{resolution}"),
                        settings=settings,
                        current_index=frame_index,
                        original_fps=original_fps,
                    )
                    if new_frame_index == 0:
                        break

                    try:
                        timestamp = time()
                        frame = await frame_loader.request_frame(frame_index)
                        time_elapsed = time() - timestamp
                        frame_index = new_frame_index

                        self.message_sender.send(OutMessageFrame(
                            stream=index,
                            frame=frame,
                            settings=settings,
                        ))

                        if required_waiting_time > time_elapsed:
                            await asyncio.sleep(required_waiting_time - time_elapsed)
                    except asyncio.CancelledError:
                        print(f"Stream {index} finished")
                        return


    async def _send_black_frames(self, index):
        black_frame_time = 10 + (20 * random())
        elapsed_time = 0.0

        while elapsed_time < black_frame_time:
            settings = self._settings
            original_fps = 30
            fps = min(original_fps, settings.fps)
            required_waiting_time = 1.0 / fps
            resolution = self._settings.resolution

            with Image.new("RGB", (resolution, resolution), color="black") as black_frame:
                buffer = io.BytesIO()
                black_frame.save(buffer, "JPEG")

            self.message_sender.send(OutMessageFrame(
                stream=index,
                frame=buffer.getvalue(),
                settings=settings,
            ))
            buffer.close()

            await asyncio.sleep(required_waiting_time)
            elapsed_time += required_waiting_time