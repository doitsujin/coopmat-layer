project('VkLayer_FROG_cooperative_matrix', ['cpp'], version : '0.1', meson_version : '>= 1.0', default_options : [ 'cpp_std=c++20' ])

data_dir = get_option('datadir')
prefix = get_option('prefix')
lib_dir = get_option('libdir')
out_lib_dir = join_paths(prefix, lib_dir)

source_files = files(
  'main.cpp',
  'lower_coopmat.cpp',
  'spirv_builder.cpp'
)

shared_library('VkLayer_FROG_cooperative_matrix', source_files,
  include_directories : include_directories('SPIRV-Headers/include'),
  install             : true
)

configure_file(
  input         : 'VkLayer_FROG_cooperative_matrix.json.in',
  output        : 'VkLayer_FROG_cooperative_matrix.' + build_machine.cpu_family() + '.json',
  configuration : {'family' : build_machine.cpu_family(), 'lib_dir' : out_lib_dir },
  install       : true,
  install_dir   : join_paths(data_dir, 'vulkan', 'implicit_layer.d'),
)