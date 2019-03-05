import visdom.server
# from visdom import server
# import threading
from threading import Thread
import asyncio
import hashlib
import uuid
from os.path import expanduser

DEFAULT_ENV_PATH = '%s/.visdom/' % expanduser("~")

class VisdomWebServer(object):
    """
    This class runs a Visdom Server.

    Args:
        port (int): Port for server to run on.
        enable_login (bool): Whether to activate login screen for the server.
        username (str): The username for login.
        password (str): The password for login. A hashed version of the password will be stored in the Visdom settings.
        cookie_secret (str): A unique string to be used as a cookie for the server.

    """
    def __init__(self, port=8097, enable_login=False, username='visdom', password='visdom', cookie_secret='visdom@d1c11598d2fb'):
        self.port = port
        self.enable_login = enable_login
        self.username = username
        self.password = password
        self.cookie_secret = cookie_secret

        thread = Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()
    
    def _hash_password(self, password):
        """Hashing Password with SHA-256"""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()
    def _set_cookie(self, cookie_secret):
        with open(DEFAULT_ENV_PATH + "COOKIE_SECRET", "w") as cookie_file:
            cookie_file.write(cookie_secret)
    def _start_server(self):
        if self.enable_login:
            user_credential = {
                "username": self.username,
                "password": self._hash_password(self._hash_password(self.password))
            }
            self._set_cookie(self.cookie_secret)
        else:
            user_credential = None
        
        visdom.server.start_server(port=self.port,
            env_path=DEFAULT_ENV_PATH,
            user_credential=user_credential)

    def run(self):
        """ Method that runs forever """
        asyncio.set_event_loop(asyncio.new_event_loop())
        self._start_server()


