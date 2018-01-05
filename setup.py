#nsml: floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.17

from distutils.core import setup

setup(
    name='WaveNet example for NSML',
    version='0.1',
    description='WaveNet for NSML',
    install_requires=[
        'librosa',
        'line_notify'
    ]
)