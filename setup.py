import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()



def get_version(short=False):
    with open('README.md') as f:
        for line in f:
            if 'Version:' in line:
                ver = line.split(':')[1].strip()
                if short:
                    subver = ver.split('.')
                    return '%s.%s' % tuple(subver[:2])
                else:
                    return ver

NAME = 'twentyoneflow'
DESCRIPTION = '21cm signal analysis and prediction using tensorflow.'
URL = 'https://github.com/SuperKam91/twentyoneflow'
EMAIL = 'kj316@mrao.cam.ac.uk'
AUTHOR = 'Kamran Javid'
VERSION = get_version()
# What packages are required for this module to be executed?
REQUIRED = ['numpy', 'sklearn', 'matplotlib', 'seaborn', 'tensorflow']
EXTRAS = []

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
)
