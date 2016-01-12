# 安装

```
brew install opencv3 --with-contrib --with-python --with-python3
mkvirtualenv -p python2.7 cv
pip install -r requirements.txt
```

使用之前，必须修改 dataset 下的 NewImages 软链接，使其指向图片数据文件夹。

```
cd dataset
ln -s <到图片数据所在的路径> NewImages
```

## python 找不到 cv2 模块

```
ln -s /usr/local/opt/opencv3/lib/python2.7/site-packages/cv2.so ~/.virtualenvs/cv/lib/python2.7/site-packages/cv2.so
```

## 装完

```
workon cv
cd VRR
python engine.py
```