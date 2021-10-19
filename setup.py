"""Setup continuous resource package."""
from setuptools import setup, find_packages


# Start simple, maybe adapt later
setup(
    name='continuousresource',
    version='0.0.1',
    description="""TBD""",
    author="R.J.J. Brouwer",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'instancegeneration = continuousresource.scripts.'
                'instancegeneration:main',
            'jobarrayinstancegeneration = continuousresource.scripts.'
                'jobarrayinstancegeneration:main',
            'lp_stresstest = continuousresource.scripts.lp_stresstest:main',
            'multipleinstancegeneration = continuousresource.scripts.'
                'multipleinstancegeneration:main',
            'multiplejobarrayinstancegeneration = continuousresource.scripts.'
                'multiplejobarrayinstancegeneration:main',
            'runtests = continuousresource.scripts.runtests:main',
            'solveinstance = continuousresource.scripts.solveinstance:main',
            'test_simulated_annealing = continuousresource.scripts.'
                'test_simulated_annealing:main'
        ]
    }
)
