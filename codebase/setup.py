from setuptools import setup, find_packages

setup(
    name='codebase',
    version='0.1.23',
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        # List your package dependencies here
        # 'dependency1',
        # 'dependency2',
    ],
    entry_points={
        # 'console_scripts': [
        #     'your_script_name = your_package.module:main_function',
        # ],
    },
    author='James Grant',
    author_email='your.email@example.com',
    description='A short description of your package',
    url='https://github.com/your_username/your_package',
)
