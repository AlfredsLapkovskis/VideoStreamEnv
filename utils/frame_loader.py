import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from models.stream_settings import StreamSettings


class FrameLoader:

    _EXECUTOR = None
    _loader_counter = 0

    def __init__(self):
        self._task_counter = -1
        self._frames = dict()
        self._pending_requests = dict()
        self._frame_dir = None
        self._settings = None
        self._closed = False

        if FrameLoader._loader_counter <= 0:
            FrameLoader._loader_counter = 1

            FrameLoader._EXECUTOR = ThreadPoolExecutor(
                max_workers=os.cpu_count(),
                thread_name_prefix="FrameLoader"
            )
        else:
            FrameLoader._loader_counter += 1
    

    def __enter__(self):
        return self
    

    def __exit__(self, type, value, traceback):
        self.close()


    def close(self):
        if self._closed:
            return
        self._closed = True
        FrameLoader._loader_counter -= 1
        if FrameLoader._loader_counter <= 0:
            FrameLoader._EXECUTOR.shutdown(cancel_futures=True)


    def prefetch_with_params(
        self, 
        frame_dir: str, 
        settings: StreamSettings, 
        current_index: int, 
        buffer_size: int=None,
        original_fps: int=None,
    ):
        assert(frame_dir)
        assert(isinstance(settings, StreamSettings))

        if not buffer_size:
            buffer_size = settings.fps
        if not original_fps:
            original_fps = settings.fps

        files = sorted(os.listdir(frame_dir))
        frame_skip = max(int(original_fps / settings.fps), 1)

        indices = range(len(files))
        indices = indices[current_index:(current_index + buffer_size) * frame_skip:frame_skip]

        self._clear_frames(frame_dir, settings, indices)

        ready_indices = set([i for i in self._frames.keys()])
        indices = [i for i in indices if i not in ready_indices]

        self._task_counter += 1

        loop = asyncio.get_running_loop()
        loop.run_in_executor(FrameLoader._EXECUTOR, self._prefetch_batch, 
            self._task_counter,
            loop,
            frame_dir,
            files,
            indices,
        )

        current_index += frame_skip
        if current_index >= len(files):
            current_index = 0
        
        return current_index


    async def request_frame(self, index):
        if index in self._frames:
            return self._frames[index]
        else:
            request = asyncio.Future()
            self._pending_requests[index] = request
            return await request
        

    def _prefetch_batch(self, task_index, loop: asyncio.BaseEventLoop, frame_dir: str, files: list[str], indices: range):
        for index in indices:
            if self._closed:
                break

            file_path = os.path.join(frame_dir, files[index])
            with open(file_path, "rb") as f:
                frame = f.read()
            if task_index != self._task_counter:
                break
            loop.call_soon_threadsafe(
                self._on_frame,
                frame,
                index,
            )
            

    def _clear_frames(self, frame_dir, settings, current_indices):
        if frame_dir != self._frame_dir or settings != self._settings:
            self._frame_dir = frame_dir
            self._settings = settings
            self._frames.clear()
            self._pending_requests.clear()
        else:
            index_set = set(current_indices)
            self._frames = {k: v for k, v in self._frames.items() if k in index_set}
            for key in list(self._pending_requests.keys()):
                if key not in index_set:
                    self._pending_requests[key].cancel()
                    del self._pending_requests[key]


    def _on_frame(self, frame, index):
        if not self._closed:
            self._frames[index] = frame

        if index in self._pending_requests:
            if self._closed:
                self._pending_requests[index].set_exception(asyncio.CancelledError())
            else:
                self._pending_requests[index].set_result(frame)
                
            self._pending_requests.pop(index)
