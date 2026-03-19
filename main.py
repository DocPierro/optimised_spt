from cal.Optimizer_PT import Optimizer_PT
from language.Eventlog import *


if __name__ == '__main__':

    ev = Eventlog("rl_data/BPIC13_closed/BPIC13_closed.xes")
    pt = ev.discover_pt_inductive()
    opt_pt = Optimizer_PT(ev, pt)

    opt_pt.estimate_exact("KLD", "L-BFGS-B", None, 100)
    #opt_pt.estimate_exact("KLD", "TNC", None, 100)
    #opt_pt.estimate_exact("KLD", "Powell", None, 100)
    #opt_pt.estimate_exact("KLD", "Nelder-Mead", None, 100)
    opt_pt.estimate_exact("rEMD", "Powell", None, 100)
    #opt_pt.estimate_exact("rEMD", "Nelder-Mead", None, 100)
