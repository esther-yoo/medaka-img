import json
import os
import warnings
from dataclasses import asdict, field
from typing import Any, Dict, Union, Tuple
from collections import OrderedDict
import torch
import torch.nn as nn

from pydantic import ValidationError
from pydantic.dataclasses import dataclass

class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

@dataclass
class BaseConfig:
    """This is the BaseConfig class which defines all the useful loading and saving methods
    of the configs"""

    name: str = field(init=False)

    def __post_init__(self):
        self.name = self.__class__.__name__

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Creates a :class:`~pythae.config.BaseConfig` instance from a dictionnary

        Args:
            config_dict (dict): The Python dictionnary containing all the parameters

        Returns:
            :class:`BaseConfig`: The created instance
        """
        try:
            config = cls(**config_dict)
        except (ValidationError, TypeError) as e:
            raise e
        return config

    @classmethod
    def _dict_from_json(cls, json_path: Union[str, os.PathLike]) -> Dict[str, Any]:
        try:
            with open(json_path) as f:
                try:
                    config_dict = json.load(f)
                    return config_dict

                except (TypeError, json.JSONDecodeError) as e:
                    raise TypeError(
                        f"File {json_path} not loadable. Maybe not json ? \n"
                        f"Catch Exception {type(e)} with message: " + str(e)
                    ) from e

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found. Please check path '{json_path}'"
            )

    @classmethod
    def from_json_file(cls, json_path: str) -> "BaseConfig":
        """Creates a :class:`~pythae.config.BaseConfig` instance from a JSON config file

        Args:
            json_path (str): The path to the json file containing all the parameters

        Returns:
            :class:`BaseConfig`: The created instance
        """
        config_dict = cls._dict_from_json(json_path)

        config_name = config_dict.pop("name")

        if cls.__name__ != config_name:
            warnings.warn(
                f"You are trying to load a "
                f"`{ cls.__name__}` while a "
                f"`{config_name}` is given."
            )

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Transforms object into a Python dictionnary

        Returns:
            (dict): The dictionnary containing all the parameters"""
        return asdict(self)

    def to_json_string(self):
        """Transforms object into a JSON string

        Returns:
            (str): The JSON str containing all the parameters"""
        return json.dumps(self.to_dict())

    def save_json(self, dir_path, filename):
        """Saves a ``.json`` file from the dataclass

        Args:
            dir_path (str): path to the folder
            filename (str): the name of the file

        """
        with open(
            os.path.join(dir_path, f"{filename}.json"), "w", encoding="utf-8"
        ) as fp:
            fp.write(self.to_json_string())


class BaseEncoder(nn.Module):
    """This is a base class for Encoders neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own encoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseEncoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Encoder(BaseEncoder):
            ...
            ...     def __init__(self):
            ...         BaseEncoder.__init__(self)
            ...         # your code
            ...
            ...     def forward(self, x: torch.Tensor):
            ...         # your code
            ...         output = ModelOutput(
            ...             embedding=embedding,
            ...             log_covariance=log_var # for VAE based models
            ...         )
            ...         return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()


class BaseDecoder(nn.Module):
    """This is a base class for Decoders neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, z: torch.Tensor):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own decoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseDecoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_decoder(BaseDecoder):
            ...
            ...    def __init__(self):
            ...        BaseDecoder.__init__(self)
            ...        # your code
            ...
            ...    def forward(self, z: torch.Tensor):
            ...        # your code
            ...        output = ModelOutput(
            ...             reconstruction=reconstruction
            ...         )
            ...        return output

        Parameters:
            z (torch.Tensor): The latent data that must be decoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the decoder

        .. note::

            By convention, the reconstruction tensors should be in [0, 1] and of shape
            BATCH x channels x ...

        """
        raise NotImplementedError()