òò
®
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.12v2.5.0-160-g8222c1cfc868¹

normal-dense1_actor/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_namenormal-dense1_actor/kernel

.normal-dense1_actor/kernel/Read/ReadVariableOpReadVariableOpnormal-dense1_actor/kernel*
_output_shapes
:	*
dtype0

normal-dense1_actor/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namenormal-dense1_actor/bias

,normal-dense1_actor/bias/Read/ReadVariableOpReadVariableOpnormal-dense1_actor/bias*
_output_shapes	
:*
dtype0

normal-dense2_actor/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_namenormal-dense2_actor/kernel

.normal-dense2_actor/kernel/Read/ReadVariableOpReadVariableOpnormal-dense2_actor/kernel* 
_output_shapes
:
*
dtype0

normal-dense2_actor/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namenormal-dense2_actor/bias

,normal-dense2_actor/bias/Read/ReadVariableOpReadVariableOpnormal-dense2_actor/bias*
_output_shapes	
:*
dtype0

normal-dense3_actor/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*+
shared_namenormal-dense3_actor/kernel

.normal-dense3_actor/kernel/Read/ReadVariableOpReadVariableOpnormal-dense3_actor/kernel*
_output_shapes
:	@*
dtype0

normal-dense3_actor/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namenormal-dense3_actor/bias

,normal-dense3_actor/bias/Read/ReadVariableOpReadVariableOpnormal-dense3_actor/bias*
_output_shapes
:@*
dtype0

normal-action_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_namenormal-action_output/kernel

/normal-action_output/kernel/Read/ReadVariableOpReadVariableOpnormal-action_output/kernel*
_output_shapes

:@*
dtype0

normal-action_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namenormal-action_output/bias

-normal-action_output/bias/Read/ReadVariableOpReadVariableOpnormal-action_output/bias*
_output_shapes
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ã
value¹B¶ B¯

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
trainable_variables
	variables
regularization_losses
		keras_api


signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
­
#layer_metrics
trainable_variables
$metrics
%layer_regularization_losses
	variables
&non_trainable_variables

'layers
regularization_losses
 
fd
VARIABLE_VALUEnormal-dense1_actor/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnormal-dense1_actor/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
(layer_metrics
trainable_variables
)metrics

*layers
	variables
+non_trainable_variables
,layer_regularization_losses
regularization_losses
fd
VARIABLE_VALUEnormal-dense2_actor/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnormal-dense2_actor/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
-layer_metrics
trainable_variables
.metrics

/layers
	variables
0non_trainable_variables
1layer_regularization_losses
regularization_losses
fd
VARIABLE_VALUEnormal-dense3_actor/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEnormal-dense3_actor/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
2layer_metrics
trainable_variables
3metrics

4layers
	variables
5non_trainable_variables
6layer_regularization_losses
regularization_losses
ge
VARIABLE_VALUEnormal-action_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEnormal-action_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
7layer_metrics
trainable_variables
8metrics

9layers
 	variables
:non_trainable_variables
;layer_regularization_losses
!regularization_losses
 
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

)serving_default_normal-observations_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
²
StatefulPartitionedCallStatefulPartitionedCall)serving_default_normal-observations_inputnormal-dense1_actor/kernelnormal-dense1_actor/biasnormal-dense2_actor/kernelnormal-dense2_actor/biasnormal-dense3_actor/kernelnormal-dense3_actor/biasnormal-action_output/kernelnormal-action_output/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference_signature_wrapper_26083
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¡
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.normal-dense1_actor/kernel/Read/ReadVariableOp,normal-dense1_actor/bias/Read/ReadVariableOp.normal-dense2_actor/kernel/Read/ReadVariableOp,normal-dense2_actor/bias/Read/ReadVariableOp.normal-dense3_actor/kernel/Read/ReadVariableOp,normal-dense3_actor/bias/Read/ReadVariableOp/normal-action_output/kernel/Read/ReadVariableOp-normal-action_output/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *'
f"R 
__inference__traced_save_26318
ü
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamenormal-dense1_actor/kernelnormal-dense1_actor/biasnormal-dense2_actor/kernelnormal-dense2_actor/biasnormal-dense3_actor/kernelnormal-dense3_actor/biasnormal-action_output/kernelnormal-action_output/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__traced_restore_26352ë
»
£
3__inference_normal-dense2_actor_layer_call_fn_26220

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_258222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
>
´	
 __inference__wrapped_model_25786
normal_observations_inputR
?normal_actor_normal_dense1_actor_matmul_readvariableop_resource:	O
@normal_actor_normal_dense1_actor_biasadd_readvariableop_resource:	S
?normal_actor_normal_dense2_actor_matmul_readvariableop_resource:
O
@normal_actor_normal_dense2_actor_biasadd_readvariableop_resource:	R
?normal_actor_normal_dense3_actor_matmul_readvariableop_resource:	@N
@normal_actor_normal_dense3_actor_biasadd_readvariableop_resource:@R
@normal_actor_normal_action_output_matmul_readvariableop_resource:@O
Anormal_actor_normal_action_output_biasadd_readvariableop_resource:
identity¢8normal-actor/normal-action_output/BiasAdd/ReadVariableOp¢7normal-actor/normal-action_output/MatMul/ReadVariableOp¢7normal-actor/normal-dense1_actor/BiasAdd/ReadVariableOp¢6normal-actor/normal-dense1_actor/MatMul/ReadVariableOp¢7normal-actor/normal-dense2_actor/BiasAdd/ReadVariableOp¢6normal-actor/normal-dense2_actor/MatMul/ReadVariableOp¢7normal-actor/normal-dense3_actor/BiasAdd/ReadVariableOp¢6normal-actor/normal-dense3_actor/MatMul/ReadVariableOp²
%normal-actor/normal-dense1_actor/CastCastnormal_observations_input*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%normal-actor/normal-dense1_actor/Castñ
6normal-actor/normal-dense1_actor/MatMul/ReadVariableOpReadVariableOp?normal_actor_normal_dense1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype028
6normal-actor/normal-dense1_actor/MatMul/ReadVariableOpú
'normal-actor/normal-dense1_actor/MatMulMatMul)normal-actor/normal-dense1_actor/Cast:y:0>normal-actor/normal-dense1_actor/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'normal-actor/normal-dense1_actor/MatMulð
7normal-actor/normal-dense1_actor/BiasAdd/ReadVariableOpReadVariableOp@normal_actor_normal_dense1_actor_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7normal-actor/normal-dense1_actor/BiasAdd/ReadVariableOp
(normal-actor/normal-dense1_actor/BiasAddBiasAdd1normal-actor/normal-dense1_actor/MatMul:product:0?normal-actor/normal-dense1_actor/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(normal-actor/normal-dense1_actor/BiasAdd¼
%normal-actor/normal-dense1_actor/ReluRelu1normal-actor/normal-dense1_actor/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%normal-actor/normal-dense1_actor/Reluò
6normal-actor/normal-dense2_actor/MatMul/ReadVariableOpReadVariableOp?normal_actor_normal_dense2_actor_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype028
6normal-actor/normal-dense2_actor/MatMul/ReadVariableOp
'normal-actor/normal-dense2_actor/MatMulMatMul3normal-actor/normal-dense1_actor/Relu:activations:0>normal-actor/normal-dense2_actor/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'normal-actor/normal-dense2_actor/MatMulð
7normal-actor/normal-dense2_actor/BiasAdd/ReadVariableOpReadVariableOp@normal_actor_normal_dense2_actor_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype029
7normal-actor/normal-dense2_actor/BiasAdd/ReadVariableOp
(normal-actor/normal-dense2_actor/BiasAddBiasAdd1normal-actor/normal-dense2_actor/MatMul:product:0?normal-actor/normal-dense2_actor/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(normal-actor/normal-dense2_actor/BiasAdd¼
%normal-actor/normal-dense2_actor/ReluRelu1normal-actor/normal-dense2_actor/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%normal-actor/normal-dense2_actor/Reluñ
6normal-actor/normal-dense3_actor/MatMul/ReadVariableOpReadVariableOp?normal_actor_normal_dense3_actor_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype028
6normal-actor/normal-dense3_actor/MatMul/ReadVariableOp
'normal-actor/normal-dense3_actor/MatMulMatMul3normal-actor/normal-dense2_actor/Relu:activations:0>normal-actor/normal-dense3_actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2)
'normal-actor/normal-dense3_actor/MatMulï
7normal-actor/normal-dense3_actor/BiasAdd/ReadVariableOpReadVariableOp@normal_actor_normal_dense3_actor_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7normal-actor/normal-dense3_actor/BiasAdd/ReadVariableOp
(normal-actor/normal-dense3_actor/BiasAddBiasAdd1normal-actor/normal-dense3_actor/MatMul:product:0?normal-actor/normal-dense3_actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(normal-actor/normal-dense3_actor/BiasAdd»
%normal-actor/normal-dense3_actor/ReluRelu1normal-actor/normal-dense3_actor/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2'
%normal-actor/normal-dense3_actor/Reluó
7normal-actor/normal-action_output/MatMul/ReadVariableOpReadVariableOp@normal_actor_normal_action_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype029
7normal-actor/normal-action_output/MatMul/ReadVariableOp
(normal-actor/normal-action_output/MatMulMatMul3normal-actor/normal-dense3_actor/Relu:activations:0?normal-actor/normal-action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(normal-actor/normal-action_output/MatMulò
8normal-actor/normal-action_output/BiasAdd/ReadVariableOpReadVariableOpAnormal_actor_normal_action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8normal-actor/normal-action_output/BiasAdd/ReadVariableOp
)normal-actor/normal-action_output/BiasAddBiasAdd2normal-actor/normal-action_output/MatMul:product:0@normal-actor/normal-action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)normal-actor/normal-action_output/BiasAdd¾
&normal-actor/normal-action_output/TanhTanh2normal-actor/normal-action_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&normal-actor/normal-action_output/TanhÌ
IdentityIdentity*normal-actor/normal-action_output/Tanh:y:09^normal-actor/normal-action_output/BiasAdd/ReadVariableOp8^normal-actor/normal-action_output/MatMul/ReadVariableOp8^normal-actor/normal-dense1_actor/BiasAdd/ReadVariableOp7^normal-actor/normal-dense1_actor/MatMul/ReadVariableOp8^normal-actor/normal-dense2_actor/BiasAdd/ReadVariableOp7^normal-actor/normal-dense2_actor/MatMul/ReadVariableOp8^normal-actor/normal-dense3_actor/BiasAdd/ReadVariableOp7^normal-actor/normal-dense3_actor/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2t
8normal-actor/normal-action_output/BiasAdd/ReadVariableOp8normal-actor/normal-action_output/BiasAdd/ReadVariableOp2r
7normal-actor/normal-action_output/MatMul/ReadVariableOp7normal-actor/normal-action_output/MatMul/ReadVariableOp2r
7normal-actor/normal-dense1_actor/BiasAdd/ReadVariableOp7normal-actor/normal-dense1_actor/BiasAdd/ReadVariableOp2p
6normal-actor/normal-dense1_actor/MatMul/ReadVariableOp6normal-actor/normal-dense1_actor/MatMul/ReadVariableOp2r
7normal-actor/normal-dense2_actor/BiasAdd/ReadVariableOp7normal-actor/normal-dense2_actor/BiasAdd/ReadVariableOp2p
6normal-actor/normal-dense2_actor/MatMul/ReadVariableOp6normal-actor/normal-dense2_actor/MatMul/ReadVariableOp2r
7normal-actor/normal-dense3_actor/BiasAdd/ReadVariableOp7normal-actor/normal-dense3_actor/BiasAdd/ReadVariableOp2p
6normal-actor/normal-dense3_actor/MatMul/ReadVariableOp6normal-actor/normal-dense3_actor/MatMul/ReadVariableOp:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namenormal-observations_input
¾


N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_25805

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
½
__inference__traced_save_26318
file_prefix9
5savev2_normal_dense1_actor_kernel_read_readvariableop7
3savev2_normal_dense1_actor_bias_read_readvariableop9
5savev2_normal_dense2_actor_kernel_read_readvariableop7
3savev2_normal_dense2_actor_bias_read_readvariableop9
5savev2_normal_dense3_actor_kernel_read_readvariableop7
3savev2_normal_dense3_actor_bias_read_readvariableop:
6savev2_normal_action_output_kernel_read_readvariableop8
4savev2_normal_action_output_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÙ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ë
valueáBÞ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesô
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_normal_dense1_actor_kernel_read_readvariableop3savev2_normal_dense1_actor_bias_read_readvariableop5savev2_normal_dense2_actor_kernel_read_readvariableop3savev2_normal_dense2_actor_bias_read_readvariableop5savev2_normal_dense3_actor_kernel_read_readvariableop3savev2_normal_dense3_actor_bias_read_readvariableop6savev2_normal_action_output_kernel_read_readvariableop4savev2_normal_action_output_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*]
_input_shapesL
J: :	::
::	@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::	

_output_shapes
: 
³

G__inference_normal-actor_layer_call_and_return_conditional_losses_26035
normal_observations_input,
normal_dense1_actor_26014:	(
normal_dense1_actor_26016:	-
normal_dense2_actor_26019:
(
normal_dense2_actor_26021:	,
normal_dense3_actor_26024:	@'
normal_dense3_actor_26026:@,
normal_action_output_26029:@(
normal_action_output_26031:
identity¢,normal-action_output/StatefulPartitionedCall¢+normal-dense1_actor/StatefulPartitionedCall¢+normal-dense2_actor/StatefulPartitionedCall¢+normal-dense3_actor/StatefulPartitionedCall
normal-dense1_actor/CastCastnormal_observations_input*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/Castä
+normal-dense1_actor/StatefulPartitionedCallStatefulPartitionedCallnormal-dense1_actor/Cast:y:0normal_dense1_actor_26014normal_dense1_actor_26016*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_258052-
+normal-dense1_actor/StatefulPartitionedCallü
+normal-dense2_actor/StatefulPartitionedCallStatefulPartitionedCall4normal-dense1_actor/StatefulPartitionedCall:output:0normal_dense2_actor_26019normal_dense2_actor_26021*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_258222-
+normal-dense2_actor/StatefulPartitionedCallû
+normal-dense3_actor/StatefulPartitionedCallStatefulPartitionedCall4normal-dense2_actor/StatefulPartitionedCall:output:0normal_dense3_actor_26024normal_dense3_actor_26026*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_258392-
+normal-dense3_actor/StatefulPartitionedCall
,normal-action_output/StatefulPartitionedCallStatefulPartitionedCall4normal-dense3_actor/StatefulPartitionedCall:output:0normal_action_output_26029normal_action_output_26031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_normal-action_output_layer_call_and_return_conditional_losses_258562.
,normal-action_output/StatefulPartitionedCallÂ
IdentityIdentity5normal-action_output/StatefulPartitionedCall:output:0-^normal-action_output/StatefulPartitionedCall,^normal-dense1_actor/StatefulPartitionedCall,^normal-dense2_actor/StatefulPartitionedCall,^normal-dense3_actor/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2\
,normal-action_output/StatefulPartitionedCall,normal-action_output/StatefulPartitionedCall2Z
+normal-dense1_actor/StatefulPartitionedCall+normal-dense1_actor/StatefulPartitionedCall2Z
+normal-dense2_actor/StatefulPartitionedCall+normal-dense2_actor/StatefulPartitionedCall2Z
+normal-dense3_actor/StatefulPartitionedCall+normal-dense3_actor/StatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namenormal-observations_input
á	
Ô
,__inference_normal-actor_layer_call_fn_25882
normal_observations_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallnormal_observations_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_normal-actor_layer_call_and_return_conditional_losses_258632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namenormal-observations_input
­


O__inference_normal-action_output_layer_call_and_return_conditional_losses_25856

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý2
ø
G__inference_normal-actor_layer_call_and_return_conditional_losses_26158

inputsE
2normal_dense1_actor_matmul_readvariableop_resource:	B
3normal_dense1_actor_biasadd_readvariableop_resource:	F
2normal_dense2_actor_matmul_readvariableop_resource:
B
3normal_dense2_actor_biasadd_readvariableop_resource:	E
2normal_dense3_actor_matmul_readvariableop_resource:	@A
3normal_dense3_actor_biasadd_readvariableop_resource:@E
3normal_action_output_matmul_readvariableop_resource:@B
4normal_action_output_biasadd_readvariableop_resource:
identity¢+normal-action_output/BiasAdd/ReadVariableOp¢*normal-action_output/MatMul/ReadVariableOp¢*normal-dense1_actor/BiasAdd/ReadVariableOp¢)normal-dense1_actor/MatMul/ReadVariableOp¢*normal-dense2_actor/BiasAdd/ReadVariableOp¢)normal-dense2_actor/MatMul/ReadVariableOp¢*normal-dense3_actor/BiasAdd/ReadVariableOp¢)normal-dense3_actor/MatMul/ReadVariableOp
normal-dense1_actor/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/CastÊ
)normal-dense1_actor/MatMul/ReadVariableOpReadVariableOp2normal_dense1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)normal-dense1_actor/MatMul/ReadVariableOpÆ
normal-dense1_actor/MatMulMatMulnormal-dense1_actor/Cast:y:01normal-dense1_actor/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/MatMulÉ
*normal-dense1_actor/BiasAdd/ReadVariableOpReadVariableOp3normal_dense1_actor_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*normal-dense1_actor/BiasAdd/ReadVariableOpÒ
normal-dense1_actor/BiasAddBiasAdd$normal-dense1_actor/MatMul:product:02normal-dense1_actor/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/BiasAdd
normal-dense1_actor/ReluRelu$normal-dense1_actor/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/ReluË
)normal-dense2_actor/MatMul/ReadVariableOpReadVariableOp2normal_dense2_actor_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)normal-dense2_actor/MatMul/ReadVariableOpÐ
normal-dense2_actor/MatMulMatMul&normal-dense1_actor/Relu:activations:01normal-dense2_actor/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense2_actor/MatMulÉ
*normal-dense2_actor/BiasAdd/ReadVariableOpReadVariableOp3normal_dense2_actor_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*normal-dense2_actor/BiasAdd/ReadVariableOpÒ
normal-dense2_actor/BiasAddBiasAdd$normal-dense2_actor/MatMul:product:02normal-dense2_actor/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense2_actor/BiasAdd
normal-dense2_actor/ReluRelu$normal-dense2_actor/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense2_actor/ReluÊ
)normal-dense3_actor/MatMul/ReadVariableOpReadVariableOp2normal_dense3_actor_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02+
)normal-dense3_actor/MatMul/ReadVariableOpÏ
normal-dense3_actor/MatMulMatMul&normal-dense2_actor/Relu:activations:01normal-dense3_actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
normal-dense3_actor/MatMulÈ
*normal-dense3_actor/BiasAdd/ReadVariableOpReadVariableOp3normal_dense3_actor_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*normal-dense3_actor/BiasAdd/ReadVariableOpÑ
normal-dense3_actor/BiasAddBiasAdd$normal-dense3_actor/MatMul:product:02normal-dense3_actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
normal-dense3_actor/BiasAdd
normal-dense3_actor/ReluRelu$normal-dense3_actor/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
normal-dense3_actor/ReluÌ
*normal-action_output/MatMul/ReadVariableOpReadVariableOp3normal_action_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*normal-action_output/MatMul/ReadVariableOpÒ
normal-action_output/MatMulMatMul&normal-dense3_actor/Relu:activations:02normal-action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-action_output/MatMulË
+normal-action_output/BiasAdd/ReadVariableOpReadVariableOp4normal_action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+normal-action_output/BiasAdd/ReadVariableOpÕ
normal-action_output/BiasAddBiasAdd%normal-action_output/MatMul:product:03normal-action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-action_output/BiasAdd
normal-action_output/TanhTanh%normal-action_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-action_output/Tanh×
IdentityIdentitynormal-action_output/Tanh:y:0,^normal-action_output/BiasAdd/ReadVariableOp+^normal-action_output/MatMul/ReadVariableOp+^normal-dense1_actor/BiasAdd/ReadVariableOp*^normal-dense1_actor/MatMul/ReadVariableOp+^normal-dense2_actor/BiasAdd/ReadVariableOp*^normal-dense2_actor/MatMul/ReadVariableOp+^normal-dense3_actor/BiasAdd/ReadVariableOp*^normal-dense3_actor/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+normal-action_output/BiasAdd/ReadVariableOp+normal-action_output/BiasAdd/ReadVariableOp2X
*normal-action_output/MatMul/ReadVariableOp*normal-action_output/MatMul/ReadVariableOp2X
*normal-dense1_actor/BiasAdd/ReadVariableOp*normal-dense1_actor/BiasAdd/ReadVariableOp2V
)normal-dense1_actor/MatMul/ReadVariableOp)normal-dense1_actor/MatMul/ReadVariableOp2X
*normal-dense2_actor/BiasAdd/ReadVariableOp*normal-dense2_actor/BiasAdd/ReadVariableOp2V
)normal-dense2_actor/MatMul/ReadVariableOp)normal-dense2_actor/MatMul/ReadVariableOp2X
*normal-dense3_actor/BiasAdd/ReadVariableOp*normal-dense3_actor/BiasAdd/ReadVariableOp2V
)normal-dense3_actor/MatMul/ReadVariableOp)normal-dense3_actor/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±	
Ë
#__inference_signature_wrapper_26083
normal_observations_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallnormal_observations_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__wrapped_model_257862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namenormal-observations_input
º


N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_25839

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾


N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_26211

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á	
Ô
,__inference_normal-actor_layer_call_fn_26010
normal_observations_input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallnormal_observations_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_normal-actor_layer_call_and_return_conditional_losses_259702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namenormal-observations_input
ú
ÿ
G__inference_normal-actor_layer_call_and_return_conditional_losses_25970

inputs,
normal_dense1_actor_25949:	(
normal_dense1_actor_25951:	-
normal_dense2_actor_25954:
(
normal_dense2_actor_25956:	,
normal_dense3_actor_25959:	@'
normal_dense3_actor_25961:@,
normal_action_output_25964:@(
normal_action_output_25966:
identity¢,normal-action_output/StatefulPartitionedCall¢+normal-dense1_actor/StatefulPartitionedCall¢+normal-dense2_actor/StatefulPartitionedCall¢+normal-dense3_actor/StatefulPartitionedCall
normal-dense1_actor/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/Castä
+normal-dense1_actor/StatefulPartitionedCallStatefulPartitionedCallnormal-dense1_actor/Cast:y:0normal_dense1_actor_25949normal_dense1_actor_25951*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_258052-
+normal-dense1_actor/StatefulPartitionedCallü
+normal-dense2_actor/StatefulPartitionedCallStatefulPartitionedCall4normal-dense1_actor/StatefulPartitionedCall:output:0normal_dense2_actor_25954normal_dense2_actor_25956*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_258222-
+normal-dense2_actor/StatefulPartitionedCallû
+normal-dense3_actor/StatefulPartitionedCallStatefulPartitionedCall4normal-dense2_actor/StatefulPartitionedCall:output:0normal_dense3_actor_25959normal_dense3_actor_25961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_258392-
+normal-dense3_actor/StatefulPartitionedCall
,normal-action_output/StatefulPartitionedCallStatefulPartitionedCall4normal-dense3_actor/StatefulPartitionedCall:output:0normal_action_output_25964normal_action_output_25966*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_normal-action_output_layer_call_and_return_conditional_losses_258562.
,normal-action_output/StatefulPartitionedCallÂ
IdentityIdentity5normal-action_output/StatefulPartitionedCall:output:0-^normal-action_output/StatefulPartitionedCall,^normal-dense1_actor/StatefulPartitionedCall,^normal-dense2_actor/StatefulPartitionedCall,^normal-dense3_actor/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2\
,normal-action_output/StatefulPartitionedCall,normal-action_output/StatefulPartitionedCall2Z
+normal-dense1_actor/StatefulPartitionedCall+normal-dense1_actor/StatefulPartitionedCall2Z
+normal-dense2_actor/StatefulPartitionedCall+normal-dense2_actor/StatefulPartitionedCall2Z
+normal-dense3_actor/StatefulPartitionedCall+normal-dense3_actor/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

G__inference_normal-actor_layer_call_and_return_conditional_losses_26060
normal_observations_input,
normal_dense1_actor_26039:	(
normal_dense1_actor_26041:	-
normal_dense2_actor_26044:
(
normal_dense2_actor_26046:	,
normal_dense3_actor_26049:	@'
normal_dense3_actor_26051:@,
normal_action_output_26054:@(
normal_action_output_26056:
identity¢,normal-action_output/StatefulPartitionedCall¢+normal-dense1_actor/StatefulPartitionedCall¢+normal-dense2_actor/StatefulPartitionedCall¢+normal-dense3_actor/StatefulPartitionedCall
normal-dense1_actor/CastCastnormal_observations_input*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/Castä
+normal-dense1_actor/StatefulPartitionedCallStatefulPartitionedCallnormal-dense1_actor/Cast:y:0normal_dense1_actor_26039normal_dense1_actor_26041*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_258052-
+normal-dense1_actor/StatefulPartitionedCallü
+normal-dense2_actor/StatefulPartitionedCallStatefulPartitionedCall4normal-dense1_actor/StatefulPartitionedCall:output:0normal_dense2_actor_26044normal_dense2_actor_26046*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_258222-
+normal-dense2_actor/StatefulPartitionedCallû
+normal-dense3_actor/StatefulPartitionedCallStatefulPartitionedCall4normal-dense2_actor/StatefulPartitionedCall:output:0normal_dense3_actor_26049normal_dense3_actor_26051*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_258392-
+normal-dense3_actor/StatefulPartitionedCall
,normal-action_output/StatefulPartitionedCallStatefulPartitionedCall4normal-dense3_actor/StatefulPartitionedCall:output:0normal_action_output_26054normal_action_output_26056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_normal-action_output_layer_call_and_return_conditional_losses_258562.
,normal-action_output/StatefulPartitionedCallÂ
IdentityIdentity5normal-action_output/StatefulPartitionedCall:output:0-^normal-action_output/StatefulPartitionedCall,^normal-dense1_actor/StatefulPartitionedCall,^normal-dense2_actor/StatefulPartitionedCall,^normal-dense3_actor/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2\
,normal-action_output/StatefulPartitionedCall,normal-action_output/StatefulPartitionedCall2Z
+normal-dense1_actor/StatefulPartitionedCall+normal-dense1_actor/StatefulPartitionedCall2Z
+normal-dense2_actor/StatefulPartitionedCall+normal-dense2_actor/StatefulPartitionedCall2Z
+normal-dense3_actor/StatefulPartitionedCall+normal-dense3_actor/StatefulPartitionedCall:b ^
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
_user_specified_namenormal-observations_input
Ý2
ø
G__inference_normal-actor_layer_call_and_return_conditional_losses_26191

inputsE
2normal_dense1_actor_matmul_readvariableop_resource:	B
3normal_dense1_actor_biasadd_readvariableop_resource:	F
2normal_dense2_actor_matmul_readvariableop_resource:
B
3normal_dense2_actor_biasadd_readvariableop_resource:	E
2normal_dense3_actor_matmul_readvariableop_resource:	@A
3normal_dense3_actor_biasadd_readvariableop_resource:@E
3normal_action_output_matmul_readvariableop_resource:@B
4normal_action_output_biasadd_readvariableop_resource:
identity¢+normal-action_output/BiasAdd/ReadVariableOp¢*normal-action_output/MatMul/ReadVariableOp¢*normal-dense1_actor/BiasAdd/ReadVariableOp¢)normal-dense1_actor/MatMul/ReadVariableOp¢*normal-dense2_actor/BiasAdd/ReadVariableOp¢)normal-dense2_actor/MatMul/ReadVariableOp¢*normal-dense3_actor/BiasAdd/ReadVariableOp¢)normal-dense3_actor/MatMul/ReadVariableOp
normal-dense1_actor/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/CastÊ
)normal-dense1_actor/MatMul/ReadVariableOpReadVariableOp2normal_dense1_actor_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)normal-dense1_actor/MatMul/ReadVariableOpÆ
normal-dense1_actor/MatMulMatMulnormal-dense1_actor/Cast:y:01normal-dense1_actor/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/MatMulÉ
*normal-dense1_actor/BiasAdd/ReadVariableOpReadVariableOp3normal_dense1_actor_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*normal-dense1_actor/BiasAdd/ReadVariableOpÒ
normal-dense1_actor/BiasAddBiasAdd$normal-dense1_actor/MatMul:product:02normal-dense1_actor/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/BiasAdd
normal-dense1_actor/ReluRelu$normal-dense1_actor/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/ReluË
)normal-dense2_actor/MatMul/ReadVariableOpReadVariableOp2normal_dense2_actor_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)normal-dense2_actor/MatMul/ReadVariableOpÐ
normal-dense2_actor/MatMulMatMul&normal-dense1_actor/Relu:activations:01normal-dense2_actor/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense2_actor/MatMulÉ
*normal-dense2_actor/BiasAdd/ReadVariableOpReadVariableOp3normal_dense2_actor_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*normal-dense2_actor/BiasAdd/ReadVariableOpÒ
normal-dense2_actor/BiasAddBiasAdd$normal-dense2_actor/MatMul:product:02normal-dense2_actor/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense2_actor/BiasAdd
normal-dense2_actor/ReluRelu$normal-dense2_actor/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense2_actor/ReluÊ
)normal-dense3_actor/MatMul/ReadVariableOpReadVariableOp2normal_dense3_actor_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02+
)normal-dense3_actor/MatMul/ReadVariableOpÏ
normal-dense3_actor/MatMulMatMul&normal-dense2_actor/Relu:activations:01normal-dense3_actor/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
normal-dense3_actor/MatMulÈ
*normal-dense3_actor/BiasAdd/ReadVariableOpReadVariableOp3normal_dense3_actor_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*normal-dense3_actor/BiasAdd/ReadVariableOpÑ
normal-dense3_actor/BiasAddBiasAdd$normal-dense3_actor/MatMul:product:02normal-dense3_actor/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
normal-dense3_actor/BiasAdd
normal-dense3_actor/ReluRelu$normal-dense3_actor/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
normal-dense3_actor/ReluÌ
*normal-action_output/MatMul/ReadVariableOpReadVariableOp3normal_action_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*normal-action_output/MatMul/ReadVariableOpÒ
normal-action_output/MatMulMatMul&normal-dense3_actor/Relu:activations:02normal-action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-action_output/MatMulË
+normal-action_output/BiasAdd/ReadVariableOpReadVariableOp4normal_action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+normal-action_output/BiasAdd/ReadVariableOpÕ
normal-action_output/BiasAddBiasAdd%normal-action_output/MatMul:product:03normal-action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-action_output/BiasAdd
normal-action_output/TanhTanh%normal-action_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-action_output/Tanh×
IdentityIdentitynormal-action_output/Tanh:y:0,^normal-action_output/BiasAdd/ReadVariableOp+^normal-action_output/MatMul/ReadVariableOp+^normal-dense1_actor/BiasAdd/ReadVariableOp*^normal-dense1_actor/MatMul/ReadVariableOp+^normal-dense2_actor/BiasAdd/ReadVariableOp*^normal-dense2_actor/MatMul/ReadVariableOp+^normal-dense3_actor/BiasAdd/ReadVariableOp*^normal-dense3_actor/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+normal-action_output/BiasAdd/ReadVariableOp+normal-action_output/BiasAdd/ReadVariableOp2X
*normal-action_output/MatMul/ReadVariableOp*normal-action_output/MatMul/ReadVariableOp2X
*normal-dense1_actor/BiasAdd/ReadVariableOp*normal-dense1_actor/BiasAdd/ReadVariableOp2V
)normal-dense1_actor/MatMul/ReadVariableOp)normal-dense1_actor/MatMul/ReadVariableOp2X
*normal-dense2_actor/BiasAdd/ReadVariableOp*normal-dense2_actor/BiasAdd/ReadVariableOp2V
)normal-dense2_actor/MatMul/ReadVariableOp)normal-dense2_actor/MatMul/ReadVariableOp2X
*normal-dense3_actor/BiasAdd/ReadVariableOp*normal-dense3_actor/BiasAdd/ReadVariableOp2V
)normal-dense3_actor/MatMul/ReadVariableOp)normal-dense3_actor/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â


N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_26231

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­


O__inference_normal-action_output_layer_call_and_return_conditional_losses_26271

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¶
¡
4__inference_normal-action_output_layer_call_fn_26260

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_normal-action_output_layer_call_and_return_conditional_losses_258562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨	
Á
,__inference_normal-actor_layer_call_fn_26104

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_normal-actor_layer_call_and_return_conditional_losses_258632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â


N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_25822

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
¢
3__inference_normal-dense1_actor_layer_call_fn_26200

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_258052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
¡
3__inference_normal-dense3_actor_layer_call_fn_26240

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_258392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º


N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_26251

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
ÿ
G__inference_normal-actor_layer_call_and_return_conditional_losses_25863

inputs,
normal_dense1_actor_25806:	(
normal_dense1_actor_25808:	-
normal_dense2_actor_25823:
(
normal_dense2_actor_25825:	,
normal_dense3_actor_25840:	@'
normal_dense3_actor_25842:@,
normal_action_output_25857:@(
normal_action_output_25859:
identity¢,normal-action_output/StatefulPartitionedCall¢+normal-dense1_actor/StatefulPartitionedCall¢+normal-dense2_actor/StatefulPartitionedCall¢+normal-dense3_actor/StatefulPartitionedCall
normal-dense1_actor/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
normal-dense1_actor/Castä
+normal-dense1_actor/StatefulPartitionedCallStatefulPartitionedCallnormal-dense1_actor/Cast:y:0normal_dense1_actor_25806normal_dense1_actor_25808*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_258052-
+normal-dense1_actor/StatefulPartitionedCallü
+normal-dense2_actor/StatefulPartitionedCallStatefulPartitionedCall4normal-dense1_actor/StatefulPartitionedCall:output:0normal_dense2_actor_25823normal_dense2_actor_25825*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_258222-
+normal-dense2_actor/StatefulPartitionedCallû
+normal-dense3_actor/StatefulPartitionedCallStatefulPartitionedCall4normal-dense2_actor/StatefulPartitionedCall:output:0normal_dense3_actor_25840normal_dense3_actor_25842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_258392-
+normal-dense3_actor/StatefulPartitionedCall
,normal-action_output/StatefulPartitionedCallStatefulPartitionedCall4normal-dense3_actor/StatefulPartitionedCall:output:0normal_action_output_25857normal_action_output_25859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *X
fSRQ
O__inference_normal-action_output_layer_call_and_return_conditional_losses_258562.
,normal-action_output/StatefulPartitionedCallÂ
IdentityIdentity5normal-action_output/StatefulPartitionedCall:output:0-^normal-action_output/StatefulPartitionedCall,^normal-dense1_actor/StatefulPartitionedCall,^normal-dense2_actor/StatefulPartitionedCall,^normal-dense3_actor/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2\
,normal-action_output/StatefulPartitionedCall,normal-action_output/StatefulPartitionedCall2Z
+normal-dense1_actor/StatefulPartitionedCall+normal-dense1_actor/StatefulPartitionedCall2Z
+normal-dense2_actor/StatefulPartitionedCall+normal-dense2_actor/StatefulPartitionedCall2Z
+normal-dense3_actor/StatefulPartitionedCall+normal-dense3_actor/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨	
Á
,__inference_normal-actor_layer_call_fn_26125

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_normal-actor_layer_call_and_return_conditional_losses_259702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î'
ã
!__inference__traced_restore_26352
file_prefix>
+assignvariableop_normal_dense1_actor_kernel:	:
+assignvariableop_1_normal_dense1_actor_bias:	A
-assignvariableop_2_normal_dense2_actor_kernel:
:
+assignvariableop_3_normal_dense2_actor_bias:	@
-assignvariableop_4_normal_dense3_actor_kernel:	@9
+assignvariableop_5_normal_dense3_actor_bias:@@
.assignvariableop_6_normal_action_output_kernel:@:
,assignvariableop_7_normal_action_output_bias:

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7ß
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ë
valueáBÞ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesØ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityª
AssignVariableOpAssignVariableOp+assignvariableop_normal_dense1_actor_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1°
AssignVariableOp_1AssignVariableOp+assignvariableop_1_normal_dense1_actor_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2²
AssignVariableOp_2AssignVariableOp-assignvariableop_2_normal_dense2_actor_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3°
AssignVariableOp_3AssignVariableOp+assignvariableop_3_normal_dense2_actor_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4²
AssignVariableOp_4AssignVariableOp-assignvariableop_4_normal_dense3_actor_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5°
AssignVariableOp_5AssignVariableOp+assignvariableop_5_normal_dense3_actor_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6³
AssignVariableOp_6AssignVariableOp.assignvariableop_6_normal_action_output_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_normal_action_output_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Û
serving_defaultÇ
_
normal-observations_inputB
+serving_default_normal-observations_input:0ÿÿÿÿÿÿÿÿÿH
normal-action_output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¨
Ø2
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
trainable_variables
	variables
regularization_losses
		keras_api


signatures
<__call__
=_default_save_signature
*>&call_and_return_all_conditional_losses"ó/
_tf_keras_network×/{"name": "normal-actor", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "normal-actor", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float16", "sparse": false, "ragged": false, "name": "normal-observations_input"}, "name": "normal-observations_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "normal-dense1_actor", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "normal-dense1_actor", "inbound_nodes": [[["normal-observations_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "normal-dense2_actor", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "normal-dense2_actor", "inbound_nodes": [[["normal-dense1_actor", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "normal-dense3_actor", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "normal-dense3_actor", "inbound_nodes": [[["normal-dense2_actor", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "normal-action_output", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "normal-action_output", "inbound_nodes": [[["normal-dense3_actor", 0, 0, {}]]]}], "input_layers": [["normal-observations_input", 0, 0]], "output_layers": [["normal-action_output", 0, 0]]}, "shared_object_id": 13, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 8]}, "float16", "normal-observations_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "normal-actor", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float16", "sparse": false, "ragged": false, "name": "normal-observations_input"}, "name": "normal-observations_input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "normal-dense1_actor", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "normal-dense1_actor", "inbound_nodes": [[["normal-observations_input", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "normal-dense2_actor", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "normal-dense2_actor", "inbound_nodes": [[["normal-dense1_actor", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "normal-dense3_actor", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "normal-dense3_actor", "inbound_nodes": [[["normal-dense2_actor", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "normal-action_output", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "normal-action_output", "inbound_nodes": [[["normal-dense3_actor", 0, 0, {}]]], "shared_object_id": 12}], "input_layers": [["normal-observations_input", 0, 0]], "output_layers": [["normal-action_output", 0, 0]]}}}
"
_tf_keras_input_layerê{"class_name": "InputLayer", "name": "normal-observations_input", "dtype": "float16", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float16", "sparse": false, "ragged": false, "name": "normal-observations_input"}}
¢	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"ý
_tf_keras_layerã{"name": "normal-dense1_actor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "normal-dense1_actor", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["normal-observations_input", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
 	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"û
_tf_keras_layerá{"name": "normal-dense2_actor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "normal-dense2_actor", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["normal-dense1_actor", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"ú
_tf_keras_layerà{"name": "normal-dense3_actor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "normal-dense3_actor", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["normal-dense2_actor", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
¡	

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
E__call__
*F&call_and_return_all_conditional_losses"ü
_tf_keras_layerâ{"name": "normal-action_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "normal-action_output", "trainable": true, "dtype": "float32", "units": 4, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["normal-dense3_actor", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
#layer_metrics
trainable_variables
$metrics
%layer_regularization_losses
	variables
&non_trainable_variables

'layers
regularization_losses
<__call__
=_default_save_signature
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
,
Gserving_default"
signature_map
-:+	2normal-dense1_actor/kernel
':%2normal-dense1_actor/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
(layer_metrics
trainable_variables
)metrics

*layers
	variables
+non_trainable_variables
,layer_regularization_losses
regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
.:,
2normal-dense2_actor/kernel
':%2normal-dense2_actor/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
-layer_metrics
trainable_variables
.metrics

/layers
	variables
0non_trainable_variables
1layer_regularization_losses
regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
-:+	@2normal-dense3_actor/kernel
&:$@2normal-dense3_actor/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
2layer_metrics
trainable_variables
3metrics

4layers
	variables
5non_trainable_variables
6layer_regularization_losses
regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
-:+@2normal-action_output/kernel
':%2normal-action_output/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
7layer_metrics
trainable_variables
8metrics

9layers
 	variables
:non_trainable_variables
;layer_regularization_losses
!regularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
þ2û
,__inference_normal-actor_layer_call_fn_25882
,__inference_normal-actor_layer_call_fn_26104
,__inference_normal-actor_layer_call_fn_26125
,__inference_normal-actor_layer_call_fn_26010À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
 __inference__wrapped_model_25786È
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *8¢5
30
normal-observations_inputÿÿÿÿÿÿÿÿÿ
ê2ç
G__inference_normal-actor_layer_call_and_return_conditional_losses_26158
G__inference_normal-actor_layer_call_and_return_conditional_losses_26191
G__inference_normal-actor_layer_call_and_return_conditional_losses_26035
G__inference_normal-actor_layer_call_and_return_conditional_losses_26060À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ý2Ú
3__inference_normal-dense1_actor_layer_call_fn_26200¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_26211¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ý2Ú
3__inference_normal-dense2_actor_layer_call_fn_26220¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_26231¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ý2Ú
3__inference_normal-dense3_actor_layer_call_fn_26240¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_26251¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Þ2Û
4__inference_normal-action_output_layer_call_fn_26260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ù2ö
O__inference_normal-action_output_layer_call_and_return_conditional_losses_26271¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÜBÙ
#__inference_signature_wrapper_26083normal-observations_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 À
 __inference__wrapped_model_25786B¢?
8¢5
30
normal-observations_inputÿÿÿÿÿÿÿÿÿ
ª "KªH
F
normal-action_output.+
normal-action_outputÿÿÿÿÿÿÿÿÿ¯
O__inference_normal-action_output_layer_call_and_return_conditional_losses_26271\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
4__inference_normal-action_output_layer_call_fn_26260O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÈ
G__inference_normal-actor_layer_call_and_return_conditional_losses_26035}J¢G
@¢=
30
normal-observations_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 È
G__inference_normal-actor_layer_call_and_return_conditional_losses_26060}J¢G
@¢=
30
normal-observations_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_normal-actor_layer_call_and_return_conditional_losses_26158j7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_normal-actor_layer_call_and_return_conditional_losses_26191j7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
  
,__inference_normal-actor_layer_call_fn_25882pJ¢G
@¢=
30
normal-observations_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
,__inference_normal-actor_layer_call_fn_26010pJ¢G
@¢=
30
normal-observations_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_normal-actor_layer_call_fn_26104]7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_normal-actor_layer_call_fn_26125]7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¯
N__inference_normal-dense1_actor_layer_call_and_return_conditional_losses_26211]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
3__inference_normal-dense1_actor_layer_call_fn_26200P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
N__inference_normal-dense2_actor_layer_call_and_return_conditional_losses_26231^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
3__inference_normal-dense2_actor_layer_call_fn_26220Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¯
N__inference_normal-dense3_actor_layer_call_and_return_conditional_losses_26251]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
3__inference_normal-dense3_actor_layer_call_fn_26240P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@à
#__inference_signature_wrapper_26083¸_¢\
¢ 
UªR
P
normal-observations_input30
normal-observations_inputÿÿÿÿÿÿÿÿÿ"KªH
F
normal-action_output.+
normal-action_outputÿÿÿÿÿÿÿÿÿ