"""
    here are all possible solver settings along with their default values and types, function for manipulating these
    settings
"""
from os.path import join


class ManipulateSolverSettings:
    def __init__(self, simulation_dir: str, target_parameter: str = "p"):
        self._path = simulation_dir
        self._cwd = "base"
        self._target_parameter = target_parameter
        self._header = None
        self._footer = None
        self._current_dict = None

    def replace_settings(self, new_settings: dict, cwd: str) -> None:
        self._cwd = cwd
        self._get_dict_from_fvSolution()
        self._replace_dict(new_settings)
        self._write_fvSolution()

    def _get_dict_from_fvSolution(self) -> None:
        # open file
        with open(join(self._path, self._cwd, "system", "fvSolution"), "r") as f:
            lines = f.readlines()

        # extract solver dict
        start, end, bracket_counter = None, None, 1337
        for i, line in enumerate(lines):
            if line.strip().lower() == self._target_parameter.lower():
                start = i
                bracket_counter = 0
                continue

            # count the brackets
            if start is not None and line.strip().startswith("{"):
                bracket_counter += 1
            elif start is not None and line.strip().startswith("}"):
                bracket_counter -= 1

            # if the bracket counter is zero, we know that the dict ended
            if start is not None and i > start and bracket_counter == 0:
                end = i
                break

        if self._header is None:
            self._header = lines[:start]
            self._footer = lines[end + 1:]

        # store the dict currently written in the fvSolution
        self._current_dict = lines[start:end + 1]

    def _replace_dict(self, new_dict: dict) -> None:
        # replace entries based on keywords, if not found, then add them to the dict
        for i, line in enumerate(self._current_dict):
            check = [line.strip().startswith(k) for k in new_dict.keys()]
            if any(check):
                # replace and remove, assuming exactly one match since we have one setting per line
                idx = [j for j, x in enumerate(check) if x][0]
                key = list(new_dict.keys())[idx]
                self._current_dict[i] = line.replace(line.split()[-1].strip(";"), str(new_dict[key]))
                new_dict.pop(key)

        # if we still have settings left, add them (we know that subdict endswith a bracket, so put them at the end of
        # the dict)
        if new_dict:
            self._current_dict = (self._current_dict[:-1] +
                                  [f"\t{key}\t\t{str(value)};\n" for key, value in new_dict.items()] +
                                  [self._current_dict[-1]])

    def _write_fvSolution(self) -> None:
        with open(join(self._path, self._cwd, "system", "fvSolution"), "w") as f:
            f.writelines(self._header)
            f.writelines(self._current_dict)
            f.writelines(self._footer)


if __name__ == "__main__":
    pass
