from gym.envs.registration import register
from myenv import ECFS_Env

# register(
# 	id = 'ECFS-v0', # 环境名,版本号v0必须有
# 	entry_point = 'myenv.ecfsenv:ECFSEnv' # 文件夹名.文件名:类名
# )

# register(
# 	id = 'ECFS-v1', # 环境名,版本号v0必须有
# 	entry_point = 'myenv.ecfsenv:ECFSEnv_dis' # 文件夹名.文件名:类名
# )

register(
	id = 'ECFS-v1', # 环境名,版本号v0必须有
	entry_point = 'myenv.ECFS_Env:ECS_Env' # 文件夹名.文件名:类名
)

__all__ = ['ECFSH_Env']