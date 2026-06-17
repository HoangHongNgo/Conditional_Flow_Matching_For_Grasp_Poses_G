from graspnetAPI import GraspNetEval, GraspGroup
import inspect

ge = GraspNetEval(root='/media/dsp520/Grasp_2T/graspnet', camera='realsense', split='test')
print([m for m in dir(ge) if not m.startswith('_')])
