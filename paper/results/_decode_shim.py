import sys, runpy
sys.path.insert(0, '/home/sogang/jaehoon/beatFCOS_new')
from beat_this.model.pl_module import PLBeatThis
# these checkpoints predate --decode, so pin the rule at load time
import functools
_init = PLBeatThis.__init__
@functools.wraps(_init)
def _patched(self, *a, **k):
    _init(self, *a, **k)
    if self.decode_kwargs is not None:
        self.decode_kwargs = dict(self.decode_kwargs,
                                         detect_tau=0.5, refine_meter=False)
PLBeatThis.__init__ = _patched
sys.argv = ['compute_paper_metrics.py'] + sys.argv[1:]
runpy.run_path('/home/sogang/jaehoon/beatFCOS_new/launch_scripts/compute_paper_metrics.py', run_name='__main__')
