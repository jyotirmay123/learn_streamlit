# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: optimizer.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='optimizer.proto',
  package='optimizer',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0foptimizer.proto\x12\toptimizer\x1a\x1cgoogle/protobuf/struct.proto\"B\n\x06\x46ilter\x12\x0b\n\x03kpi\x18\x01 \x01(\t\x12+\n\x07options\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.ListValue\"E\n\x05Scope\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\"\n\x07\x66ilters\x18\x02 \x03(\x0b\x32\x11.optimizer.Filter\x12\n\n\x02id\x18\x03 \x01(\x03\"Y\n\x04Rule\x12\x11\n\tpredicate\x18\x01 \x01(\t\x12\x1a\n\x12interpolated_price\x18\x02 \x01(\t\x12\"\n\x07\x66ilters\x18\x03 \x03(\x0b\x32\x11.optimizer.Filter\"\'\n\x06Target\x12\r\n\x05value\x18\x01 \x01(\x01\x12\x0e\n\x06weight\x18\x02 \x01(\x05\"<\n\x0bPreviewData\x12\x0e\n\x06profit\x18\x01 \x01(\x01\x12\x0f\n\x07revenue\x18\x02 \x01(\x01\x12\x0c\n\x04name\x18\x03 \x01(\t\"9\n\rTargetPreview\x12(\n\x08previews\x18\x01 \x03(\x0b\x32\x16.optimizer.PreviewData\"\x97\x01\n\x0cTargetResult\x12\x12\n\nscope_name\x18\x01 \x01(\t\x12\x12\n\nnum_prices\x18\x02 \x01(\x05\x12\x15\n\rprofit_actual\x18\x03 \x01(\x01\x12\x16\n\x0erevenue_actual\x18\x04 \x01(\x01\x12\x16\n\x0erevenue_target\x18\x05 \x01(\x01\x12\x18\n\x10total_num_prices\x18\x06 \x01(\x05\"\xd4\x01\n\x0bRequestData\x12\x16\n\x0e\x66orecast_table\x18\x01 \x01(\t\x12\x15\n\rforecast_date\x18\x02 \x01(\t\x12\x12\n\nstart_date\x18\x03 \x01(\t\x12\x10\n\x08\x65nd_date\x18\x04 \x01(\t\x12 \n\x06scopes\x18\x06 \x03(\x0b\x32\x10.optimizer.Scope\x12\x1e\n\x05rules\x18\x07 \x03(\x0b\x32\x0f.optimizer.Rule\x12\"\n\x07targets\x18\x08 \x03(\x0b\x32\x11.optimizer.TargetJ\x04\x08\x05\x10\x06R\x04kpis\"6\n\x0ePreviewRequest\x12$\n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\x16.optimizer.RequestData\"9\n\x0fPreviewResponse\x12&\n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32\x18.optimizer.TargetPreview\"k\n\x13OptimizationRequest\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x14\n\x0c\x63\x61llback_url\x18\x03 \x01(\t\x12$\n\x04\x64\x61ta\x18\x04 \x01(\x0b\x32\x16.optimizer.RequestData\"z\n\x14OptimizationResponse\x12\n\n\x02id\x18\x01 \x01(\x03\x12\x19\n\x11\x66iltered_forecast\x18\x02 \x01(\t\x12(\n\x07results\x18\x03 \x03(\x0b\x32\x17.optimizer.TargetResult\x12\x11\n\tartifacts\x18\x04 \x03(\tb\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_struct__pb2.DESCRIPTOR,])




_FILTER = _descriptor.Descriptor(
  name='Filter',
  full_name='optimizer.Filter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='kpi', full_name='optimizer.Filter.kpi', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='options', full_name='optimizer.Filter.options', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=126,
)


_SCOPE = _descriptor.Descriptor(
  name='Scope',
  full_name='optimizer.Scope',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='optimizer.Scope.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='filters', full_name='optimizer.Scope.filters', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='id', full_name='optimizer.Scope.id', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=128,
  serialized_end=197,
)


_RULE = _descriptor.Descriptor(
  name='Rule',
  full_name='optimizer.Rule',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='predicate', full_name='optimizer.Rule.predicate', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='interpolated_price', full_name='optimizer.Rule.interpolated_price', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='filters', full_name='optimizer.Rule.filters', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=199,
  serialized_end=288,
)


_TARGET = _descriptor.Descriptor(
  name='Target',
  full_name='optimizer.Target',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='optimizer.Target.value', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='weight', full_name='optimizer.Target.weight', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=290,
  serialized_end=329,
)


_PREVIEWDATA = _descriptor.Descriptor(
  name='PreviewData',
  full_name='optimizer.PreviewData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='profit', full_name='optimizer.PreviewData.profit', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='revenue', full_name='optimizer.PreviewData.revenue', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='optimizer.PreviewData.name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=331,
  serialized_end=391,
)


_TARGETPREVIEW = _descriptor.Descriptor(
  name='TargetPreview',
  full_name='optimizer.TargetPreview',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='previews', full_name='optimizer.TargetPreview.previews', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=393,
  serialized_end=450,
)


_TARGETRESULT = _descriptor.Descriptor(
  name='TargetResult',
  full_name='optimizer.TargetResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='scope_name', full_name='optimizer.TargetResult.scope_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num_prices', full_name='optimizer.TargetResult.num_prices', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='profit_actual', full_name='optimizer.TargetResult.profit_actual', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='revenue_actual', full_name='optimizer.TargetResult.revenue_actual', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='revenue_target', full_name='optimizer.TargetResult.revenue_target', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_num_prices', full_name='optimizer.TargetResult.total_num_prices', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=453,
  serialized_end=604,
)


_REQUESTDATA = _descriptor.Descriptor(
  name='RequestData',
  full_name='optimizer.RequestData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='forecast_table', full_name='optimizer.RequestData.forecast_table', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='forecast_date', full_name='optimizer.RequestData.forecast_date', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='start_date', full_name='optimizer.RequestData.start_date', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end_date', full_name='optimizer.RequestData.end_date', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scopes', full_name='optimizer.RequestData.scopes', index=4,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rules', full_name='optimizer.RequestData.rules', index=5,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='targets', full_name='optimizer.RequestData.targets', index=6,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=607,
  serialized_end=819,
)


_PREVIEWREQUEST = _descriptor.Descriptor(
  name='PreviewRequest',
  full_name='optimizer.PreviewRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='optimizer.PreviewRequest.data', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=821,
  serialized_end=875,
)


_PREVIEWRESPONSE = _descriptor.Descriptor(
  name='PreviewResponse',
  full_name='optimizer.PreviewResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='optimizer.PreviewResponse.data', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=877,
  serialized_end=934,
)


_OPTIMIZATIONREQUEST = _descriptor.Descriptor(
  name='OptimizationRequest',
  full_name='optimizer.OptimizationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='optimizer.OptimizationRequest.id', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='optimizer.OptimizationRequest.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='callback_url', full_name='optimizer.OptimizationRequest.callback_url', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='optimizer.OptimizationRequest.data', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=936,
  serialized_end=1043,
)


_OPTIMIZATIONRESPONSE = _descriptor.Descriptor(
  name='OptimizationResponse',
  full_name='optimizer.OptimizationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='optimizer.OptimizationResponse.id', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='filtered_forecast', full_name='optimizer.OptimizationResponse.filtered_forecast', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='results', full_name='optimizer.OptimizationResponse.results', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='artifacts', full_name='optimizer.OptimizationResponse.artifacts', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1045,
  serialized_end=1167,
)

_FILTER.fields_by_name['options'].message_type = google_dot_protobuf_dot_struct__pb2._LISTVALUE
_SCOPE.fields_by_name['filters'].message_type = _FILTER
_RULE.fields_by_name['filters'].message_type = _FILTER
_TARGETPREVIEW.fields_by_name['previews'].message_type = _PREVIEWDATA
_REQUESTDATA.fields_by_name['scopes'].message_type = _SCOPE
_REQUESTDATA.fields_by_name['rules'].message_type = _RULE
_REQUESTDATA.fields_by_name['targets'].message_type = _TARGET
_PREVIEWREQUEST.fields_by_name['data'].message_type = _REQUESTDATA
_PREVIEWRESPONSE.fields_by_name['data'].message_type = _TARGETPREVIEW
_OPTIMIZATIONREQUEST.fields_by_name['data'].message_type = _REQUESTDATA
_OPTIMIZATIONRESPONSE.fields_by_name['results'].message_type = _TARGETRESULT
DESCRIPTOR.message_types_by_name['Filter'] = _FILTER
DESCRIPTOR.message_types_by_name['Scope'] = _SCOPE
DESCRIPTOR.message_types_by_name['Rule'] = _RULE
DESCRIPTOR.message_types_by_name['Target'] = _TARGET
DESCRIPTOR.message_types_by_name['PreviewData'] = _PREVIEWDATA
DESCRIPTOR.message_types_by_name['TargetPreview'] = _TARGETPREVIEW
DESCRIPTOR.message_types_by_name['TargetResult'] = _TARGETRESULT
DESCRIPTOR.message_types_by_name['RequestData'] = _REQUESTDATA
DESCRIPTOR.message_types_by_name['PreviewRequest'] = _PREVIEWREQUEST
DESCRIPTOR.message_types_by_name['PreviewResponse'] = _PREVIEWRESPONSE
DESCRIPTOR.message_types_by_name['OptimizationRequest'] = _OPTIMIZATIONREQUEST
DESCRIPTOR.message_types_by_name['OptimizationResponse'] = _OPTIMIZATIONRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Filter = _reflection.GeneratedProtocolMessageType('Filter', (_message.Message,), {
  'DESCRIPTOR' : _FILTER,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.Filter)
  })
_sym_db.RegisterMessage(Filter)

Scope = _reflection.GeneratedProtocolMessageType('Scope', (_message.Message,), {
  'DESCRIPTOR' : _SCOPE,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.Scope)
  })
_sym_db.RegisterMessage(Scope)

Rule = _reflection.GeneratedProtocolMessageType('Rule', (_message.Message,), {
  'DESCRIPTOR' : _RULE,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.Rule)
  })
_sym_db.RegisterMessage(Rule)

Target = _reflection.GeneratedProtocolMessageType('Target', (_message.Message,), {
  'DESCRIPTOR' : _TARGET,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.Target)
  })
_sym_db.RegisterMessage(Target)

PreviewData = _reflection.GeneratedProtocolMessageType('PreviewData', (_message.Message,), {
  'DESCRIPTOR' : _PREVIEWDATA,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.PreviewData)
  })
_sym_db.RegisterMessage(PreviewData)

TargetPreview = _reflection.GeneratedProtocolMessageType('TargetPreview', (_message.Message,), {
  'DESCRIPTOR' : _TARGETPREVIEW,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.TargetPreview)
  })
_sym_db.RegisterMessage(TargetPreview)

TargetResult = _reflection.GeneratedProtocolMessageType('TargetResult', (_message.Message,), {
  'DESCRIPTOR' : _TARGETRESULT,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.TargetResult)
  })
_sym_db.RegisterMessage(TargetResult)

RequestData = _reflection.GeneratedProtocolMessageType('RequestData', (_message.Message,), {
  'DESCRIPTOR' : _REQUESTDATA,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.RequestData)
  })
_sym_db.RegisterMessage(RequestData)

PreviewRequest = _reflection.GeneratedProtocolMessageType('PreviewRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREVIEWREQUEST,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.PreviewRequest)
  })
_sym_db.RegisterMessage(PreviewRequest)

PreviewResponse = _reflection.GeneratedProtocolMessageType('PreviewResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREVIEWRESPONSE,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.PreviewResponse)
  })
_sym_db.RegisterMessage(PreviewResponse)

OptimizationRequest = _reflection.GeneratedProtocolMessageType('OptimizationRequest', (_message.Message,), {
  'DESCRIPTOR' : _OPTIMIZATIONREQUEST,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.OptimizationRequest)
  })
_sym_db.RegisterMessage(OptimizationRequest)

OptimizationResponse = _reflection.GeneratedProtocolMessageType('OptimizationResponse', (_message.Message,), {
  'DESCRIPTOR' : _OPTIMIZATIONRESPONSE,
  '__module__' : 'optimizer_pb2'
  # @@protoc_insertion_point(class_scope:optimizer.OptimizationResponse)
  })
_sym_db.RegisterMessage(OptimizationResponse)


# @@protoc_insertion_point(module_scope)
