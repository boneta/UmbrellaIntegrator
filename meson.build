project('UmbrellaIntegrator', 'c',
  version: '0.6.1',
  license: 'GPLv3')

add_languages('fortran', native: false)

py = import('python').find_installation(pure: false)

openmp_dep = dependency('openmp', required: false)
incdir_numpy = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
incdir_f2py = run_command(py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()
fortranobject_c = incdir_f2py / 'fortranobject.c'

umbrellaint_fortran_c = custom_target('umbrellaint_fortran-module',
  input: ['umbrellaint_fortran.f90'],
  output: ['umbrellaint_fortranmodule.c'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'umbrellaint_fortran', '--lower']
)

py.extension_module('umbrellaint_fortran',
  ['umbrellaint_fortran.f90', umbrellaint_fortran_c, fortranobject_c],
  include_directories: [incdir_numpy, incdir_f2py],
  dependencies: [openmp_dep],
  install: true
)

py.install_sources(['umbrellaint.py'], subdir: '.')