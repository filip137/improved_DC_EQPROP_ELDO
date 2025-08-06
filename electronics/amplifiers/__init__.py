# __init__.py

from .old_bidirectional_amp_dict import old_bidir_dict
from .three_terminal_bidirectional_amp import three_bidir_dict
from .bidir_amp_with_nonlin      import bidir_amp_with_nonlin
from .bidir_amp_without_nonlin      import bidir_amp_without_nonlin
from .perfect_amp_nonlin import perfect_amp_nonlin
dicts = {
    "ThreeTerminalBiDirAmp": three_bidir_dict,
    "OldBiDirAmp"          : old_bidir_dict,
    "BiDirWithNonLin"      : bidir_amp_with_nonlin,
    "BiDirWithOutNonLin"      : bidir_amp_without_nonlin,
    "PerfectAmpWithNonlin" : perfect_amp_nonlin
}

__all__ = ["dicts"]

