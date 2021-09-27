from setuptools import setup

setup(
	name='fscpy',
	version='0.1.0',    
	description='Python package for Field State Classification',
	url='https://github.com/Davit98/fscpy',
	author='Davit Martirosyan',
	author_email='dvmartirosyan@gmail.com',
	license='BSD 2-clause',
	packages=['fscpy'],
	install_requires=[
	'numpy==1.20.1',
	'pandas==1.2.4',
	'tqdm==4.59.0',
	'geopandas==0.9.0',
	'rasterio==1.2.8',
	'torchvision==0.5.0',
	'torch==1.4.0'
	]
)