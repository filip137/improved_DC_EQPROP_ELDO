# __init__.py

from .old_bidirectional_amp_dict import old_bidir_dict
from .three_terminal_bidirectional_amp import three_bidir_dict
from .bidir_amp_with_nonlin      import bidir_amp_with_nonlin

dicts = {
    "ThreeTerminalBiDirAmp": three_bidir_dict,
    "OldBiDirAmp"          : old_bidir_dict,
    "BiDirWithNonLin"      : bidir_amp_with_nonlin,
}

__all__ = ["dicts"]

