conda install package_name #安装包
source activate my_env #进入环境
source deactivate #离开环境
conda env export > environment.yaml #将包保存为 YAML
conda env create -f environment.yaml #通过环境文件创建环境
conda env list #列出环境
conda env remove -n env_name #删除指定的环境
