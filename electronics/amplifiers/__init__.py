# __init__.py

from .old_bidirectional_amp_dict import old_bidir_dict
from .three_terminal_bidirectional_amp import three_bidir_dict
from .bidir_amp_with_nonlin      import bidir_amp_with_nonlin
from .bidir_amp_without_nonlin      import bidir_amp_without_nonlin
from .perfect_amp_nonlin import perfect_amp_nonlin
from .bidir_amp_with_nonlin_cap import bidir_amp_with_nonlin_cap
from .bidir_amp_without_nonlin_test import bidir_amp_without_nonlin_test
from .perfect_amp import perfect_amp
dicts = {
    "ThreeTerminalBiDirAmp": three_bidir_dict,
    "OldBiDirAmp"          : old_bidir_dict,
    "BiDirWithNonLin"      : bidir_amp_with_nonlin,
    "BiDirWithOutNonLin"      : bidir_amp_without_nonlin,
    "BiDirWithOutNonLinTest"      : bidir_amp_without_nonlin_test,
    "PerfectAmpWithNonlin" : perfect_amp_nonlin,
    "PerfectAmp" : perfect_amp,
    "BiDirWithNonLinCAP" : bidir_amp_with_nonlin_cap
}

__all__ = ["dicts"]

