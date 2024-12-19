"""
    execute the base case, create copies

    parts of the code are copied from drlfoam, currently maintained by @AndreWeiner:

    https://github.com/OFDataCommittee/drlfoam
"""
import logging

from queue import Queue
from typing import Union
from threading import Thread
from subprocess import Popen
from os import remove, makedirs
from os.path import join, exists
from shutil import copytree, rmtree


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Executer:
    def __init__(self, base_path: str = "", simulation: str = "cylinder_2D_Re100",
                 run_directory: str = "test_optimization", n_runner: int = 2, buffer_size: int = 4,
                 start_time: Union[int, float] = 3, end_time: Union[int, float] = 3, timeout: Union[int, float] = 1e15):
        self._base_path = base_path
        self._run_directory = run_directory
        self._simulation = simulation
        self._n_runner = n_runner
        self._buffer_size = buffer_size
        self._end_time = end_time
        self._start_time = start_time
        self._run_script = "Allrun"
        self._solver = "pimpleFoam"
        self._timeout = timeout
        self._manager = TaskManager(self._n_runner)

        # copy the test case to the run directory
        makedirs(join(self._base_path, self._run_directory), exist_ok=True)
        if not exists(join(self._run_directory, "base")):
            copytree(join(self._base_path, "test_cases", self._simulation),
                     join(self._base_path, self._run_directory, "base"), dirs_exist_ok=True)

    def prepare(self):
        # execute the base case
        cwd = join(self._base_path, self._run_directory, "base")
        self._manager.add(submit_and_wait, f"./{self._run_script}.pre", cwd, self._timeout)
        self._manager.run()

        # create copies
        self._create_copies()

    def execute(self):
        # replace the start and end times
        self._set_times_and_clean()

        # execute the simulations in the copy_* directories
        for c in range(self._buffer_size):
            self._manager.add(submit_and_wait, f"./{self._run_script}",
                              join(self._base_path, self._run_directory, f"copy_{c}"), self._timeout)
        self._manager.run()

    def clean_simulation(self, dir_path: str):
        # we only need to remove the log.pimpleFoam file to re-run the simulations with new settings
        if exists(join(dir_path, f"log.{self._solver}")):
            remove(join(dir_path, f"log.{self._solver}"))
            try:
                rmtree(join(dir_path, "postProcessing"))
            except FileNotFoundError:
                pass

    def _create_copies(self):
        for i in range(self._buffer_size):
            dest = join(self._base_path, self._run_directory, f"copy_{i}")
            if not exists(dest):
                copytree(join(self._base_path, self._run_directory, "base"), dest, dirs_exist_ok=True)

    def _replace_times(self, pwd: str, key: str, value: str, file: str = "controlDict"):
        # load the file and replace the line
        with open(join(self._base_path, self._run_directory, pwd, "system", file), "r") as f:
            lines = f.readlines()
        lines = [f"{key}\t\t{value};\n" if line.startswith(key) else line for line in lines]

        # write the modified lines back to the file
        with open(join(self._base_path, self._run_directory, pwd, "system", file), "w") as f:
            f.writelines(lines)

    def _set_times_and_clean(self):
        for c in range(self._buffer_size):
            # set the times
            self._replace_times(f"copy_{c}", "startFrom", "startTime")
            self._replace_times(f"copy_{c}", "startTime", str(self._start_time))
            self._replace_times(f"copy_{c}", "endTime", str(self._end_time))

            # remove the pimpleFoam log and postProcessing folder
            self.clean_simulation(join(self._base_path, self._run_directory, f"copy_{c}"))

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, t_start):
        self._start_time = t_start

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, t_end):
        self._end_time = t_end

    @property
    def run_directory(self):
        return self._run_directory

    @property
    def buffer_size(self):
        return self._buffer_size


# everything below taken from drlfoam.execution
def submit_and_wait(cmd: str, cwd: str, timeout: int = 1e15):
    proc = Popen([cmd], cwd=cwd)
    proc.wait(timeout)


def string_args(args: list, kwargs: dict) -> str:
    args_str = ", ".join([str(arg) for arg in args])
    kwargs_str = ", ".join(f"{key}={str(value)}" for key, value in kwargs.items())
    if args_str and kwargs_str:
        return args_str + ", " + kwargs_str
    elif args_str and not kwargs_str:
        return args_str
    elif not args_str and kwargs_str:
        return kwargs_str
    else:
        return ""


class Runner(Thread):
    def __init__(self, tasks: Queue, name: str):
        super(Runner, self).__init__()
        self._tasks = tasks
        self._name = name
        self.daemon = True
        self.start()

    def run(self) -> None:
        while not self._tasks.empty():
            try:
                func, args, kwargs = self._tasks.get()
                logger.info(f"{self._name}: {func.__name__}({string_args(args, kwargs)})")
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{self._name}: " + str(e))
            finally:
                self._tasks.task_done()

        logger.info(f"{self._name}: all tasks done")


class TaskManager(Queue):
    def __init__(self, n_runners_max: int):
        super(TaskManager, self).__init__()
        self._n_runners_max = n_runners_max
        self._runners = None

    def add(self, task, *args, **kwargs) -> None:
        self.put((task, args, kwargs))

    def run(self, wait: bool = True) -> None:
        n_runners = min(self._n_runners_max, self.qsize())
        self._runners = [Runner(self, f"Runner {i}") for i in range(n_runners)]
        if wait:
            self.wait()

    def wait(self) -> None:
        self.join()


if __name__ == "__main__":
    pass
