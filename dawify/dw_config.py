from dataclasses import dataclass
import yaml
from pathlib import Path
from typing import Any, Tuple, Type

from dawify.mis_utils import rprint

## base classes taken from nerfstudio

# Pretty printing class
class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)

    def save_config(self, filename: str) -> None:
        """Save the config to a YAML file."""
        with open(filename, 'w') as f:
            yaml.dump(self, f)
        rprint(f"[yellow]config saved to: {filename}[/yellow]")


def load_config(filename: str, inp_conf = None) -> InstantiateConfig:
    """load and overwrite config from file"""
    config = yaml.load(Path(filename).read_text(), Loader=yaml.Loader)
    return config
