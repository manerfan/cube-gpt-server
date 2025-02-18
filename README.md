# MODU Server

## 开发

### 包管理

- [Configure a Poetry environment in PyCharm](https://www.jetbrains.com/help/pycharm/poetry.html)

### 创建环境
```shell
python3 -m venv .venv
source .venv/bin/activate
```

### 安装依赖
```shell
pip3 install poetry
poetry install
```

### 启动服务    
```shell
cd deploy/docker
docker-compose -f docker-compose-dev.yml -p modu-dev up
```

### 初始化数据库
```shell
cd app
alembic upgrade head
```

### 启动服务
```shell
cd app
python3 start.py
```
