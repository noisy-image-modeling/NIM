from dataclasses import dataclass

from umei.omega import CrossValConf, SegExpConf

@dataclass(kw_only=True)
class BTCVExpConf(SegExpConf, CrossValConf):
    pass
