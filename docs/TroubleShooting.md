# Trouble Shootings

## Tensorflow version
Though Tensorpack offical code requires Tensorflow version >= 1.5, we found 1.5 will cause OOM issue when training big models while 1.4 works well. But we did not verify other versions >1.5.

## libtensorflow_framework
```
tensorflow.python.framework.errors_impl.NotFoundError: libtensorflow_framework.so: cannot open shared object file: No such file or directory
```
Please deactivate and uninstall/delete horovod package
```
pip3 uninstall horovod
sudo rm -rf /usr/local/lib/python3.5/dist-packages/horovod
```

## \__Unicode Error\__ for pycocotools
Sometimes, the installed pycocotools does not fully support Python 3. The reason is still unknown so far. Please check
```
python3.5/dist-packages/pycocotools/coco.py
```
If you line 308 looks like
```
if type(resFile) == str or type(resFile) == unicode:
```
replace it with,
```
if type(resFile) == str or type(resFile) == bytes:
```

## Speed
Increase `DATA.NUM_WORKERS` can speedup significantly if you have enought CPU cores.
