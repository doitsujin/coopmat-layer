project('VkLayer_FROG_cooperative_matrix', ['cpp'], version : '0.1', meson_version : '>= 1.0', default_options : [ 'cpp_std=c++20' ])

source_files = files(
  'main.cpp',
  'lower_coopmat.cpp',
  'spirv_builder.cpp'
)

layer_files = files(
  'VkLayer_FROG_cooperative_matrix.json'
)

shared_library('VkLayer_FROG_cooperative_matrix', source_files,
  include_directories : include_directories('SPIRV-Headers/include'),
  name_prefix         : '',
  install             : true
)

install_data(layer_files,
  install_dir : get_option('datadir') / 'vulkan/implicit_layer.d'
)
