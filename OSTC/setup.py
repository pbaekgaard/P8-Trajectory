import os
import platform
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)
            self.generate_stubs(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), 
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

    def generate_stubs(self, ext):
        stubs_dir = os.path.join(self.build_lib, "ostc")
        os.makedirs(stubs_dir, exist_ok=True)
        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.abspath(self.build_lib)
        else:
            env['PYTHONPATH'] += ':' + os.path.abspath(self.build_lib)

        try:
            subprocess.check_call(['pybind11-stubgen', 'ostc', '-o', self.build_lib], env=env)
            print(f"Generated stubs for {ext.name} in '{self.build_lib}'.")
        except subprocess.CalledProcessError:
            print(f"Failed to generate stubs for {ext.name}", file=sys.stderr)
        py_typed_path = os.path.join(stubs_dir, 'py.typed')
        open(py_typed_path, 'w').close()  # Create an empty file


setup(
    name='OSTC',
    version='1.0.0',
    author='P8Project AAU 2025',
    author_email='cs-25-sw-8-06@student.aau.dk',
    description='A package for performing OSTC Compression',
    long_description='',
    setup_requires=["pybind11_stubgen"],
    packages=find_packages(),
    package_data={'ostc': ['*']},
    ext_modules=[CMakeExtension('ostc', ".")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
