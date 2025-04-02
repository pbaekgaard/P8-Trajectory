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
        stubs_dir = os.path.join(self.build_lib)
        os.makedirs(stubs_dir, exist_ok=True)
        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.abspath(self.build_lib)
        else:
            env['PYTHONPATH'] += ':' + os.path.abspath(self.build_lib)

        try:
            subprocess.check_call(['pybind11-stubgen', 'ostc', '-o', stubs_dir], env=env)
            print(f"Generated stubs for {ext.name} in '{stubs_dir}'.")
        except subprocess.CalledProcessError:
            print(f"Failed to generate stubs for {ext.name}", file=sys.stderr)
        py_typed_path = os.path.join(self.build_lib, 'py.typed')
        open(py_typed_path, 'w').close()  # Create an empty file


setup(
    name='ostc',
    version='1.0.0',
    author='Geoffrey Hunter',
    author_email='gbmhunter@gmail.com',
    description='A test project using pybind11 and CMake',
    long_description='',
    packages=find_packages(),
    package_data={'ostc': ['*']},
    ext_modules=[CMakeExtension('ostc', ".")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
