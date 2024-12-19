"""
    here are all options for the solver dict stored
"""
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GAMGSolverOptions:
    def __init__(self):
        self.keys = {"solver": "GAMG",
                     "smoother": {"FDIC", "DIC", "DICGaussSeidel", "symGaussSeidel", "nonBlockingGaussSeidel",
                                  "GaussSeidel"},
                     "tolerance": {"continuous value range"},
                     "relTol": {"continuous value range"},
                     "interpolateCorrection": {"yes", "no"},
                     "nFinestSweeps": {"discrete value range"},
                     "nCellsInCoarsestLevel": {"discrete value range"},
                     "nPreSweeps": {"discrete value range"},
                     "maxPreSweeps": {"discrete value range"},
                     "nPostSweeps": {"discrete value range"},
                     "maxPostSweeps": {"discrete value range"},
                     "directSolveCoarsest": {"yes", "no"},
                     "cacheAgglomeration": {"yes", "no"},
                     "postSweepsLevelMultiplier": {"discrete value range"},
                     "preSweepsLevelMultiplier": {"discrete value range"}
                     }

    def show_options(self):
        logger.info("Available options for solver settings:")
        for key, value in self.keys.items():
            value = next(iter(value)) if len(value) == 1 else value
            logger.info(f"\t{key}:\t{value}")

    @staticmethod
    def create_default_dict() -> dict:
        return {"solver": "GAMG", "smoother": "DICGaussSeidel", "tolerance": 1e-06, "relTol": 0.01}


if __name__ == "__main__":
    pass
