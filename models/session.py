from uuid import uuid4
import os
from datetime import datetime
import json

from .stream_settings import StreamSettings
from .service_level_objectives import ServiceLevelObjectives


class Session:
    
    def __init__(self):
        self._id = uuid4()

        self.is_connected = False
        self._stream_settings = None
        self._slos = None
        self._listeners = set()
        self._historical_stream_settings = dict()
        self._datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")

        config = {}
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.loads(f.read()) 

        self._resource_dir = os.path.join(config.get("resource_dir", "resources"))
        os.makedirs(self._resource_dir, exist_ok=True)

        self._logs_dir = os.path.join(config.get("logs_dir", "logs"))
        os.makedirs(self._logs_dir, exist_ok=True)

    @property
    def id(self):
        return self._id
    

    @property
    def stream_settings(self):
        return self._stream_settings
    

    @stream_settings.setter
    def stream_settings(self, value):
        if value != self._stream_settings:
            if value is not None:
                self._historical_stream_settings[value.id] = value

            self._stream_settings = value
            self._notify_listeners()

        
    @property
    def slos(self):
        return self._slos
    

    @slos.setter
    def slos(self, value):
        if value != self._slos:
            self._slos = value
            self._notify_listeners()


    @property
    def resource_dir(self):
        return self._resource_dir
    

    @property
    def logs_dir(self):
        return self._logs_dir
    

    @property
    def datetime_string(self):
        return self._datetime_string


    # TODO: this may throw KeyError
    def get_stream_settings(self, id):
        return self._historical_stream_settings[id]


    def add_listener(self, on_change):
        self._listeners.add(on_change)


    def remove_listener(self, on_change):
        try:
            self._listeners.remove(on_change)
        except KeyError:
            pass


    def _notify_listeners(self):
        for listener in self._listeners:
            listener()
