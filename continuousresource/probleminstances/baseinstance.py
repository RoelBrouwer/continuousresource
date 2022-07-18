from abc import ABC
from abc import abstractmethod


class BaseInstance(ABC):
    """Class template for grouping functions related to problem instances
    of a specific type.
    """
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def from_binary(path):
        """Read a problem instance from a binary file.

        Parameters
        ----------
        path : str
            Path to the binary file containing the instance.

        Returns
        -------
        Dict of ndarray
            Dictionary containing the instance data, in the format
            expected for a particular instance type.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_csv(path):
        """Read a problem instance from three csv files.

        Parameters
        ----------
        path : str
            Path to the folder containing the csv files describing the
            instance.

        Returns
        -------
        Dict of ndarray
            Dictionary containing the instance data, in the format
            expected for a particular instance type.
        """
        pass

    @staticmethod
    @abstractmethod
    def to_binary(path, instance):
        """Write instance data to a binary file.

        Parameters
        ----------
        path : str
            Filename of outputfile (IMPORTANT: without extension).
        instance : Dict of ndarray
            Dictionary containing the instance data, in the format
            expected for a particular instance type.
        """
        pass

    @staticmethod
    @abstractmethod
    def to_csv(path, instance):
        """Write instance data to five csv files.

        Parameters
        ----------
        path : str
            Path to output folder.
        instance : Dict of ndarray
            Dictionary containing the instance data, in the format
            expected for a particular instance type.
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_instance(*args, **kwargs):
        """Generate a single instance of a specific type.

        Parameters
        ----------
        *args :
            Should contain exactly the (non-keyword) arguments required
            for the construction of an instance of this type.
        **kwargs :
            Should contain exactly the keyword arguments required for the
            construction of an instance of this type.
        """
        pass
