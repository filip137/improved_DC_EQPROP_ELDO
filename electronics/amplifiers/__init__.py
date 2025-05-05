from .bidirectional_amp_dict import new_bidir_dict
from .old_bidirectional_amp_dict import old_bidir_dict
from .three_terminal_bidirectional_amp import three_bidir_dict
dicts = {"NewBiDirAmp" : new_bidir_dict,
         "ThreeTerminalBiDirAmp" : three_bidir_dict,
         "OldBiDirAmp" : old_bidir_dict}

__all__ = ["dicts", "new_bidir_dict", "old_bidir_dict"]
