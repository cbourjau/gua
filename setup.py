from setuptools import setup

install_requires = ['rootpy']
tests_require = ['nose']

setup(
    name='gua',
    version='1.0.1',
    description="Post analysis",
    author='Christian Bourjau',
    author_email='christian.bourjau@cern.ch',
    packages=['gua'],
    include_package_data=True,
    # package_data={
    #     'nittygriddy': [
    #         'non-python-files/datasets.yml',
    #         'non-python-files/run.C',
    #         'non-python-files/GetSetting.C',
    #         # 'examples/flame_graph.png'
    #     ],
    # },
    # long_description=open('README.rst').read(),
    # url='https://github.com/cbourjau/nittygriddy',
    # keywords=['alice', 'cern', 'grid', 'proof'],
    # scripts=glob('scripts/*'),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Development Status :: 4 - Beta",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=install_requires,
    extras_require={'test': tests_require},
    test_suite='nose.collector',
)
