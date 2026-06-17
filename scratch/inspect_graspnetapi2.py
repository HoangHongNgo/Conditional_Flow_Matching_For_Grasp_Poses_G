from graspnetAPI import GraspNetEval
import inspect

print(inspect.signature(GraspNetEval.eval_scene))
print(inspect.getdoc(GraspNetEval.eval_scene))
