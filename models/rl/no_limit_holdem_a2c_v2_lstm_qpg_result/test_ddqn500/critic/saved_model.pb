��$
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02unknown8��"
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	�*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
w
lstm_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	6�*
shared_namelstm_2/kernel
p
!lstm_2/kernel/Read/ReadVariableOpReadVariableOplstm_2/kernel*
_output_shapes
:	6�*
dtype0
�
lstm_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_namelstm_2/recurrent_kernel
�
+lstm_2/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_2/recurrent_kernel* 
_output_shapes
:
��*
dtype0
o
lstm_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelstm_2/bias
h
lstm_2/bias/Read/ReadVariableOpReadVariableOplstm_2/bias*
_output_shapes	
:�*
dtype0
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
��*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_6/gamma
�
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_6/beta
�
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:�*
dtype0
z
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_9/kernel
s
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel* 
_output_shapes
:
��*
dtype0
q
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_9/bias
j
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_7/gamma
�
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_7/beta
�
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:�*
dtype0
|
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_10/kernel
u
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel* 
_output_shapes
:
��*
dtype0
s
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_10/bias
l
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_8/gamma
�
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_8/beta
�
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_6/moving_mean
�
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_6/moving_variance
�
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_7/moving_mean
�
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_7/moving_variance
�
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_8/moving_mean
�
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_8/moving_variance
�
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:�*
dtype0

NoOpNoOp
�7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�6
value�6B�6 B�6
�
input_layer
lstm_layers
hidden_layers
output_layer

signatures
	variables
trainable_variables
regularization_losses
		keras_api
 


0

0
1
2
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
 
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
21
22
~
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
15
16
 
�
)metrics
*non_trainable_variables
	variables

+layers
,layer_regularization_losses
trainable_variables
regularization_losses
l
-cell
.
state_spec
/	variables
0trainable_variables
1regularization_losses
2	keras_api
s
3dense_layer
4
norm_layer
5	variables
6trainable_variables
7regularization_losses
8	keras_api
s
9dense_layer
:
norm_layer
;	variables
<trainable_variables
=regularization_losses
>	keras_api
s
?dense_layer
@
norm_layer
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
SQ
VARIABLE_VALUEdense_11/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdense_11/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Emetrics
Fnon_trainable_variables
	variables

Glayers
Hlayer_regularization_losses
trainable_variables
regularization_losses
IG
VARIABLE_VALUElstm_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_2/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUElstm_2/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_8/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_8/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_6/gamma&variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_6/beta&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_9/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_9/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_7/gamma&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_7/beta'variables/10/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_10/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_10/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_8/gamma'variables/13/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_8/beta'variables/14/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_6/moving_mean'variables/15/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_6/moving_variance'variables/16/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_7/moving_mean'variables/17/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_7/moving_variance'variables/18/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_8/moving_mean'variables/19/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_8/moving_variance'variables/20/.ATTRIBUTES/VARIABLE_VALUE
 
*
#0
$1
%2
&3
'4
(5
*
0

1
2
3
4
5
 
~

kernel
recurrent_kernel
bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
 

0
1
2

0
1
2
 
�
Mmetrics
Nnon_trainable_variables
/	variables

Olayers
Player_regularization_losses
0trainable_variables
1regularization_losses
h

kernel
bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
�
Uaxis
	gamma
beta
#moving_mean
$moving_variance
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
*
0
1
2
3
#4
$5

0
1
2
3
 
�
Zmetrics
[non_trainable_variables
5	variables

\layers
]layer_regularization_losses
6trainable_variables
7regularization_losses
h

kernel
bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
�
baxis
	gamma
beta
%moving_mean
&moving_variance
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
*
0
1
2
3
%4
&5

0
1
2
3
 
�
gmetrics
hnon_trainable_variables
;	variables

ilayers
jlayer_regularization_losses
<trainable_variables
=regularization_losses
h

kernel
 bias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
�
oaxis
	!gamma
"beta
'moving_mean
(moving_variance
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
*
0
 1
!2
"3
'4
(5

0
 1
!2
"3
 
�
tmetrics
unon_trainable_variables
A	variables

vlayers
wlayer_regularization_losses
Btrainable_variables
Cregularization_losses
 
 
 
 

0
1
2

0
1
2
 
�
xmetrics
ynon_trainable_variables
I	variables

zlayers
{layer_regularization_losses
Jtrainable_variables
Kregularization_losses
 
 

-0
 

0
1

0
1
 
�
|metrics
}non_trainable_variables
Q	variables

~layers
layer_regularization_losses
Rtrainable_variables
Sregularization_losses
 

0
1
#2
$3

0
1
 
�
�metrics
�non_trainable_variables
V	variables
�layers
 �layer_regularization_losses
Wtrainable_variables
Xregularization_losses
 

#0
$1

30
41
 

0
1

0
1
 
�
�metrics
�non_trainable_variables
^	variables
�layers
 �layer_regularization_losses
_trainable_variables
`regularization_losses
 

0
1
%2
&3

0
1
 
�
�metrics
�non_trainable_variables
c	variables
�layers
 �layer_regularization_losses
dtrainable_variables
eregularization_losses
 

%0
&1

90
:1
 

0
 1

0
 1
 
�
�metrics
�non_trainable_variables
k	variables
�layers
 �layer_regularization_losses
ltrainable_variables
mregularization_losses
 

!0
"1
'2
(3

!0
"1
 
�
�metrics
�non_trainable_variables
p	variables
�layers
 �layer_regularization_losses
qtrainable_variables
rregularization_losses
 

'0
(1

?0
@1
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

#0
$1
 
 
 
 
 
 
 

%0
&1
 
 
 
 
 
 
 

'0
(1
 
 
�
serving_default_input_1Placeholder*+
_output_shapes
:���������6*
dtype0* 
shape:���������6
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm_2/kernellstm_2/recurrent_kernellstm_2/biasdense_8/kerneldense_8/bias!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_6/betabatch_normalization_6/gammadense_9/kerneldense_9/bias!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancebatch_normalization_7/betabatch_normalization_7/gammadense_10/kerneldense_10/bias!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancebatch_normalization_8/betabatch_normalization_8/gammadense_11/kerneldense_11/bias*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*/
f*R(
&__inference_signature_wrapper_74975466
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp!lstm_2/kernel/Read/ReadVariableOp+lstm_2/recurrent_kernel/Read/ReadVariableOplstm_2/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOpConst*$
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_save_74975559
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kerneldense_11/biaslstm_2/kernellstm_2/recurrent_kernellstm_2/biasdense_8/kerneldense_8/biasbatch_normalization_6/gammabatch_normalization_6/betadense_9/kerneldense_9/biasbatch_normalization_7/gammabatch_normalization_7/betadense_10/kerneldense_10/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference__traced_restore_74975640�!
�
�
#__inference__wrapped_model_74975437
input_1>
:lstm_neural_network_model_2_statefulpartitionedcall_args_1>
:lstm_neural_network_model_2_statefulpartitionedcall_args_2>
:lstm_neural_network_model_2_statefulpartitionedcall_args_3>
:lstm_neural_network_model_2_statefulpartitionedcall_args_4>
:lstm_neural_network_model_2_statefulpartitionedcall_args_5>
:lstm_neural_network_model_2_statefulpartitionedcall_args_6>
:lstm_neural_network_model_2_statefulpartitionedcall_args_7>
:lstm_neural_network_model_2_statefulpartitionedcall_args_8>
:lstm_neural_network_model_2_statefulpartitionedcall_args_9?
;lstm_neural_network_model_2_statefulpartitionedcall_args_10?
;lstm_neural_network_model_2_statefulpartitionedcall_args_11?
;lstm_neural_network_model_2_statefulpartitionedcall_args_12?
;lstm_neural_network_model_2_statefulpartitionedcall_args_13?
;lstm_neural_network_model_2_statefulpartitionedcall_args_14?
;lstm_neural_network_model_2_statefulpartitionedcall_args_15?
;lstm_neural_network_model_2_statefulpartitionedcall_args_16?
;lstm_neural_network_model_2_statefulpartitionedcall_args_17?
;lstm_neural_network_model_2_statefulpartitionedcall_args_18?
;lstm_neural_network_model_2_statefulpartitionedcall_args_19?
;lstm_neural_network_model_2_statefulpartitionedcall_args_20?
;lstm_neural_network_model_2_statefulpartitionedcall_args_21?
;lstm_neural_network_model_2_statefulpartitionedcall_args_22?
;lstm_neural_network_model_2_statefulpartitionedcall_args_23
identity��3lstm_neural_network_model_2/StatefulPartitionedCall�
3lstm_neural_network_model_2/StatefulPartitionedCallStatefulPartitionedCallinput_1:lstm_neural_network_model_2_statefulpartitionedcall_args_1:lstm_neural_network_model_2_statefulpartitionedcall_args_2:lstm_neural_network_model_2_statefulpartitionedcall_args_3:lstm_neural_network_model_2_statefulpartitionedcall_args_4:lstm_neural_network_model_2_statefulpartitionedcall_args_5:lstm_neural_network_model_2_statefulpartitionedcall_args_6:lstm_neural_network_model_2_statefulpartitionedcall_args_7:lstm_neural_network_model_2_statefulpartitionedcall_args_8:lstm_neural_network_model_2_statefulpartitionedcall_args_9;lstm_neural_network_model_2_statefulpartitionedcall_args_10;lstm_neural_network_model_2_statefulpartitionedcall_args_11;lstm_neural_network_model_2_statefulpartitionedcall_args_12;lstm_neural_network_model_2_statefulpartitionedcall_args_13;lstm_neural_network_model_2_statefulpartitionedcall_args_14;lstm_neural_network_model_2_statefulpartitionedcall_args_15;lstm_neural_network_model_2_statefulpartitionedcall_args_16;lstm_neural_network_model_2_statefulpartitionedcall_args_17;lstm_neural_network_model_2_statefulpartitionedcall_args_18;lstm_neural_network_model_2_statefulpartitionedcall_args_19;lstm_neural_network_model_2_statefulpartitionedcall_args_20;lstm_neural_network_model_2_statefulpartitionedcall_args_21;lstm_neural_network_model_2_statefulpartitionedcall_args_22;lstm_neural_network_model_2_statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*4
f/R-
+__inference_restored_function_body_2704032525
3lstm_neural_network_model_2/StatefulPartitionedCall�
IdentityIdentity<lstm_neural_network_model_2/StatefulPartitionedCall:output:04^lstm_neural_network_model_2/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2j
3lstm_neural_network_model_2/StatefulPartitionedCall3lstm_neural_network_model_2/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�.
�
(__inference_while_body_10005875_27031841
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
8__inference_batch_normalization_6_layer_call_fn_27018094

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_270180852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�.
�
(__inference_while_body_10002938_27011547
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
D__inference_lstm_2_layer_call_and_return_conditional_losses_27024221

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*j
_output_shapesX
V:����������:����������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270241882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������6:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�G
�
"__inference_standard_lstm_26997217

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*n
_output_shapes\
Z: : : : :����������:����������: : : : : : : : : : : : : : : : : : : : : *4
body,R*
(__inference_while_body_10004269_26988676*4
cond,R*
(__inference_while_cond_10004268_26997142*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:����������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*^
_input_shapesM
K:���������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�
�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27002456

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
(__inference_while_body_10035087_27023990
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�)
�
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_27029219

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource6
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource8
4batch_normalization_6_cast_2_readvariableop_resource8
4batch_normalization_6_cast_3_readvariableop_resource
identity��)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�+batch_normalization_6/Cast_2/ReadVariableOp�+batch_normalization_6/Cast_3/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAdd�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOp�
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/BiasAdd:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1r
TanhTanh)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2Z
+batch_normalization_6/Cast_2/ReadVariableOp+batch_normalization_6/Cast_2/ReadVariableOp2Z
+batch_normalization_6/Cast_3/ReadVariableOp+batch_normalization_6/Cast_3/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�O
�
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_27018947

inputs*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource2
.batch_normalization_7_assignmovingavg_100047904
0batch_normalization_7_assignmovingavg_1_100047966
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource
identity��9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAdd�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indices�
"batch_normalization_7/moments/meanMeandense_9/BiasAdd:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_7/moments/mean�
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_7/moments/StopGradient�
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_9/BiasAdd:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_7/moments/SquaredDifference�
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indices�
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_7/moments/variance�
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze�
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1�
+batch_normalization_7/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10004790*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_7/AssignMovingAvg/decay�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_10004790*
_output_shapes	
:�*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp�
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10004790*
_output_shapes	
:�2+
)batch_normalization_7/AssignMovingAvg/sub�
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10004790*
_output_shapes	
:�2+
)batch_normalization_7/AssignMovingAvg/mul�
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_10004790-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10004790*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_7/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10004796*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_7/AssignMovingAvg_1/decay�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_7_assignmovingavg_1_10004796*
_output_shapes	
:�*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10004796*
_output_shapes	
:�2-
+batch_normalization_7/AssignMovingAvg_1/sub�
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10004796*
_output_shapes	
:�2-
+batch_normalization_7/AssignMovingAvg_1/mul�
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_7_assignmovingavg_1_10004796/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10004796*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1r
TanhTanh)batch_normalization_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�)
�
__inference_call_27006112

inputs*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource6
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource8
4batch_normalization_7_cast_2_readvariableop_resource8
4batch_normalization_7_cast_3_readvariableop_resource
identity��)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�+batch_normalization_7/Cast_2/ReadVariableOp�+batch_normalization_7/Cast_3/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAdd�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOp�
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1r
TanhTanh)batch_normalization_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2Z
+batch_normalization_7/Cast_2/ReadVariableOp+batch_normalization_7/Cast_2/ReadVariableOp2Z
+batch_normalization_7/Cast_3/ReadVariableOp+batch_normalization_7/Cast_3/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27001945

inputs
assignmovingavg_10003541
assignmovingavg_1_10003547 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/10003541*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10003541*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10003541*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10003541*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10003541AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10003541*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/10003547*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10003547*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10003547*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10003547*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10003547AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10003547*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_10005263_27026900
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10005263___redundant_placeholder00
,while_cond_10005263___redundant_placeholder10
,while_cond_10005263___redundant_placeholder20
,while_cond_10005263___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�)
�
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_27013033

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource6
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource8
4batch_normalization_6_cast_2_readvariableop_resource8
4batch_normalization_6_cast_3_readvariableop_resource
identity��)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�+batch_normalization_6/Cast_2/ReadVariableOp�+batch_normalization_6/Cast_3/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAdd�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOp�
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/BiasAdd:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1r
TanhTanh)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2Z
+batch_normalization_6/Cast_2/ReadVariableOp+batch_normalization_6/Cast_2/ReadVariableOp2Z
+batch_normalization_6/Cast_3/ReadVariableOp+batch_normalization_6/Cast_3/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_lstm_2_layer_call_fn_27011663
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_2_layer_call_and_return_conditional_losses_270116552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������6:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
�	
�
F__inference_simple_neural_network_layer_block_6_layer_call_fn_27013044

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_270130332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
F__inference_simple_neural_network_layer_block_8_layer_call_fn_27003154

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_270031432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
D__inference_lstm_2_layer_call_and_return_conditional_losses_27034162
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCallF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*s
_output_shapesa
_:����������:�������������������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270341292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������6:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
�O
�
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_26999506

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource2
.batch_normalization_8_assignmovingavg_100416904
0batch_normalization_8_assignmovingavg_1_100416966
2batch_normalization_8_cast_readvariableop_resource8
4batch_normalization_8_cast_1_readvariableop_resource
identity��9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_8/AssignMovingAvg/ReadVariableOp�;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAdd�
"batch_normalization_8/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/x�
"batch_normalization_8/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/y�
 batch_normalization_8/LogicalAnd
LogicalAnd+batch_normalization_8/LogicalAnd/x:output:0+batch_normalization_8/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_8/LogicalAnd�
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indices�
"batch_normalization_8/moments/meanMeandense_10/BiasAdd:output:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_8/moments/mean�
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_8/moments/StopGradient�
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_10/BiasAdd:output:03batch_normalization_8/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_8/moments/SquaredDifference�
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices�
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_8/moments/variance�
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze�
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1�
+batch_normalization_8/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10041690*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_8/AssignMovingAvg/decay�
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_8_assignmovingavg_10041690*
_output_shapes	
:�*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp�
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10041690*
_output_shapes	
:�2+
)batch_normalization_8/AssignMovingAvg/sub�
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10041690*
_output_shapes	
:�2+
)batch_normalization_8/AssignMovingAvg/mul�
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_8_assignmovingavg_10041690-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10041690*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_8/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10041696*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_8/AssignMovingAvg_1/decay�
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_8_assignmovingavg_1_10041696*
_output_shapes	
:�*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10041696*
_output_shapes	
:�2-
+batch_normalization_8/AssignMovingAvg_1/sub�
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10041696*
_output_shapes	
:�2-
+batch_normalization_8/AssignMovingAvg_1/mul�
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_8_assignmovingavg_1_10041696/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10041696*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_10/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/add_1r
TanhTanh)batch_normalization_8/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_8/AssignMovingAvg/ReadVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�3
�
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27019078
input_1)
%lstm_2_statefulpartitionedcall_args_1)
%lstm_2_statefulpartitionedcall_args_2)
%lstm_2_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6+
'dense_11_statefulpartitionedcall_args_1+
'dense_11_statefulpartitionedcall_args_2
identity�� dense_11/StatefulPartitionedCall�lstm_2/StatefulPartitionedCall�;simple_neural_network_layer_block_6/StatefulPartitionedCall�;simple_neural_network_layer_block_7/StatefulPartitionedCall�;simple_neural_network_layer_block_8/StatefulPartitionedCall�
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinput_1%lstm_2_statefulpartitionedcall_args_1%lstm_2_statefulpartitionedcall_args_2%lstm_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_2_layer_call_and_return_conditional_losses_270122342 
lstm_2/StatefulPartitionedCall�
;simple_neural_network_layer_block_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_269911882=
;simple_neural_network_layer_block_6/StatefulPartitionedCall�
;simple_neural_network_layer_block_7/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_6/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_270189472=
;simple_neural_network_layer_block_7/StatefulPartitionedCall�
;simple_neural_network_layer_block_8/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_7/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_270073762=
;simple_neural_network_layer_block_8/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_8/StatefulPartitionedCall:output:0'dense_11_statefulpartitionedcall_args_1'dense_11_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_270117342"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_11/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall<^simple_neural_network_layer_block_6/StatefulPartitionedCall<^simple_neural_network_layer_block_7/StatefulPartitionedCall<^simple_neural_network_layer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2z
;simple_neural_network_layer_block_6/StatefulPartitionedCall;simple_neural_network_layer_block_6/StatefulPartitionedCall2z
;simple_neural_network_layer_block_7/StatefulPartitionedCall;simple_neural_network_layer_block_7/StatefulPartitionedCall2z
;simple_neural_network_layer_block_8/StatefulPartitionedCall;simple_neural_network_layer_block_8/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�O
�
__inference_call_26993300

inputs*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource2
.batch_normalization_7_assignmovingavg_100057084
0batch_normalization_7_assignmovingavg_1_100057146
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource
identity��9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAdd�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indices�
"batch_normalization_7/moments/meanMeandense_9/BiasAdd:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_7/moments/mean�
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_7/moments/StopGradient�
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_9/BiasAdd:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_7/moments/SquaredDifference�
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indices�
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_7/moments/variance�
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze�
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1�
+batch_normalization_7/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10005708*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_7/AssignMovingAvg/decay�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_10005708*
_output_shapes	
:�*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp�
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10005708*
_output_shapes	
:�2+
)batch_normalization_7/AssignMovingAvg/sub�
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10005708*
_output_shapes	
:�2+
)batch_normalization_7/AssignMovingAvg/mul�
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_10005708-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10005708*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_7/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10005714*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_7/AssignMovingAvg_1/decay�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_7_assignmovingavg_1_10005714*
_output_shapes	
:�*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10005714*
_output_shapes	
:�2-
+batch_normalization_7/AssignMovingAvg_1/sub�
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10005714*
_output_shapes	
:�2-
+batch_normalization_7/AssignMovingAvg_1/mul�
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_7_assignmovingavg_1_10005714/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10005714*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1r
TanhTanh)batch_normalization_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�)
�
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_27011428

inputs*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource6
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource8
4batch_normalization_7_cast_2_readvariableop_resource8
4batch_normalization_7_cast_3_readvariableop_resource
identity��)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�+batch_normalization_7/Cast_2/ReadVariableOp�+batch_normalization_7/Cast_3/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAdd�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOp�
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1r
TanhTanh)batch_normalization_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2Z
+batch_normalization_7/Cast_2/ReadVariableOp+batch_normalization_7/Cast_2/ReadVariableOp2Z
+batch_normalization_7/Cast_3/ReadVariableOp+batch_normalization_7/Cast_3/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�O
�
__inference_call_27008177

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource2
.batch_normalization_8_assignmovingavg_100057614
0batch_normalization_8_assignmovingavg_1_100057676
2batch_normalization_8_cast_readvariableop_resource8
4batch_normalization_8_cast_1_readvariableop_resource
identity��9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_8/AssignMovingAvg/ReadVariableOp�;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAdd�
"batch_normalization_8/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/x�
"batch_normalization_8/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/y�
 batch_normalization_8/LogicalAnd
LogicalAnd+batch_normalization_8/LogicalAnd/x:output:0+batch_normalization_8/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_8/LogicalAnd�
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indices�
"batch_normalization_8/moments/meanMeandense_10/BiasAdd:output:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_8/moments/mean�
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_8/moments/StopGradient�
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_10/BiasAdd:output:03batch_normalization_8/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_8/moments/SquaredDifference�
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices�
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_8/moments/variance�
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze�
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1�
+batch_normalization_8/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10005761*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_8/AssignMovingAvg/decay�
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_8_assignmovingavg_10005761*
_output_shapes	
:�*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp�
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10005761*
_output_shapes	
:�2+
)batch_normalization_8/AssignMovingAvg/sub�
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10005761*
_output_shapes	
:�2+
)batch_normalization_8/AssignMovingAvg/mul�
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_8_assignmovingavg_10005761-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10005761*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_8/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10005767*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_8/AssignMovingAvg_1/decay�
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_8_assignmovingavg_1_10005767*
_output_shapes	
:�*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10005767*
_output_shapes	
:�2-
+batch_normalization_8/AssignMovingAvg_1/sub�
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10005767*
_output_shapes	
:�2-
+batch_normalization_8/AssignMovingAvg_1/mul�
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_8_assignmovingavg_1_10005767/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10005767*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_10/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/add_1r
TanhTanh)batch_normalization_8/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_8/AssignMovingAvg/ReadVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�3
�
___forward_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_28230262
inputs_0*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource6
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource8
4batch_normalization_6_cast_2_readvariableop_resource8
4batch_normalization_6_cast_3_readvariableop_resource
identity
tanh)
%batch_normalization_6_batchnorm_mul_1'
#batch_normalization_6_batchnorm_sub
dense_8_biasadd'
#batch_normalization_6_batchnorm_mul-
)batch_normalization_6_cast_readvariableop!
dense_8_matmul_readvariableop

inputs/
+batch_normalization_6_cast_3_readvariableop)
%batch_normalization_6_batchnorm_rsqrt��)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�+batch_normalization_6/Cast_2/ReadVariableOp�+batch_normalization_6/Cast_3/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulinputs_0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAdd�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOp�
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/BiasAdd:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1r
TanhTanh)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"N
#batch_normalization_6_batchnorm_mul'batch_normalization_6/batchnorm/mul:z:0"R
%batch_normalization_6_batchnorm_mul_1)batch_normalization_6/batchnorm/mul_1:z:0"R
%batch_normalization_6_batchnorm_rsqrt)batch_normalization_6/batchnorm/Rsqrt:y:0"N
#batch_normalization_6_batchnorm_sub'batch_normalization_6/batchnorm/sub:z:0"b
+batch_normalization_6_cast_3_readvariableop3batch_normalization_6/Cast_3/ReadVariableOp:value:0"^
)batch_normalization_6_cast_readvariableop1batch_normalization_6/Cast/ReadVariableOp:value:0"+
dense_8_biasadddense_8/BiasAdd:output:0"F
dense_8_matmul_readvariableop%dense_8/MatMul/ReadVariableOp:value:0"
identityIdentity:output:0"
inputsinputs_0"
tanhTanh:y:0*?
_input_shapes.
,:����������::::::*�
backward_function_namewu__inference___backward_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_28230204_282302632V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2Z
+batch_normalization_6/Cast_2/ReadVariableOp+batch_normalization_6/Cast_2/ReadVariableOp2Z
+batch_normalization_6/Cast_3/ReadVariableOp+batch_normalization_6/Cast_3/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_7_layer_call_fn_27025395

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_270253862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�.
�
(__inference_while_body_10005264_27008708
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
8__inference_batch_normalization_6_layer_call_fn_26989149

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_269891402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�O
�
__inference_call_27031521

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource2
.batch_normalization_6_assignmovingavg_100056554
0batch_normalization_6_assignmovingavg_1_100056616
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource
identity��9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAdd�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indices�
"batch_normalization_6/moments/meanMeandense_8/BiasAdd:output:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_6/moments/mean�
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_6/moments/StopGradient�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_8/BiasAdd:output:03batch_normalization_6/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_6/moments/SquaredDifference�
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices�
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_6/moments/variance�
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze�
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1�
+batch_normalization_6/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10005655*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_6/AssignMovingAvg/decay�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_10005655*
_output_shapes	
:�*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp�
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10005655*
_output_shapes	
:�2+
)batch_normalization_6/AssignMovingAvg/sub�
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10005655*
_output_shapes	
:�2+
)batch_normalization_6/AssignMovingAvg/mul�
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_10005655-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10005655*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_6/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10005661*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_6/AssignMovingAvg_1/decay�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_6_assignmovingavg_1_10005661*
_output_shapes	
:�*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10005661*
_output_shapes	
:�2-
+batch_normalization_6/AssignMovingAvg_1/sub�
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10005661*
_output_shapes	
:�2-
+batch_normalization_6/AssignMovingAvg_1/mul�
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_6_assignmovingavg_1_10005661/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10005661*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/BiasAdd:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1r
TanhTanh)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_lstm_2_layer_call_fn_27006589
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_2_layer_call_and_return_conditional_losses_270065812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������6:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
�G
�
"__inference_standard_lstm_27006548

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10002485_26991144*4
cond,R*
(__inference_while_cond_10002484_27006473*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*g
_input_shapesV
T:������������������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�.
�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_26997000

inputs
assignmovingavg_10044862
assignmovingavg_1_10044868 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/10044862*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10044862*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10044862*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10044862*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10044862AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10044862*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/10044868*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10044868*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10044868*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10044868*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10044868AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10044868*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_10033735_26991798
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10033735___redundant_placeholder00
,while_cond_10033735___redundant_placeholder10
,while_cond_10033735___redundant_placeholder20
,while_cond_10033735___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�
�
&__inference_signature_wrapper_74975466
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__wrapped_model_749754372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�.
�
(__inference_while_body_10034181_27022964
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
D__inference_lstm_2_layer_call_and_return_conditional_losses_27024098

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*j
_output_shapesX
V:����������:����������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270240652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������6:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27025386

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:& "
 
_user_specified_nameinputs
�)
�
__inference_call_27025485

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource6
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource8
4batch_normalization_6_cast_2_readvariableop_resource8
4batch_normalization_6_cast_3_readvariableop_resource
identity��)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�+batch_normalization_6/Cast_2/ReadVariableOp�+batch_normalization_6/Cast_3/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAdd�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
+batch_normalization_6/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_6_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_2/ReadVariableOp�
+batch_normalization_6/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_6_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_3/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV23batch_normalization_6/Cast_1/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/BiasAdd:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul1batch_normalization_6/Cast/ReadVariableOp:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub3batch_normalization_6/Cast_2/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1r
TanhTanh)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp,^batch_normalization_6/Cast_2/ReadVariableOp,^batch_normalization_6/Cast_3/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2Z
+batch_normalization_6/Cast_2/ReadVariableOp+batch_normalization_6/Cast_2/ReadVariableOp2Z
+batch_normalization_6/Cast_3/ReadVariableOp+batch_normalization_6/Cast_3/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_27001851

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_26990909

inputs
assignmovingavg_10044944
assignmovingavg_1_10044950 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/10044944*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10044944*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10044944*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10044944*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10044944AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10044944*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/10044950*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10044950*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10044950*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10044950*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10044950AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10044950*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
(__inference_while_body_10004269_26988676
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
D__inference_lstm_2_layer_call_and_return_conditional_losses_27012234

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*j
_output_shapesX
V:����������:����������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270122012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������6:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
 __forward_standard_lstm_28230749

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4&
"tensorarrayv2stack_tensorliststack
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2
transpose_1_perm	
while
while_maximum_iterations
while_0
while_1
while_2
while_3
while_4
while_5
while_6
while_7
while_8
while_9
while_10
while_11
while_12
while_13
while_14
while_15
while_16
	transpose
transpose_perm��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeM
ShapeShapetranspose_0:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose_0:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose_0:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�	
while_17Whilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbiasmul_2_0/accumulator:handle:0"placeholder_0/accumulator:handle:0 Sigmoid_2_0/accumulator:handle:0Tanh_1_0/accumulator:handle:0mul_0/accumulator:handle:0mul_1_0/accumulator:handle:0 Sigmoid_1_0/accumulator:handle:0$placeholder_3_0/accumulator:handle:0Sigmoid_0/accumulator:handle:0Tanh_0/accumulator:handle:0MatMul_0/accumulator:handle:0MatMul_1_0/accumulator:handle:0,MatMul/ReadVariableOp_0/accumulator:handle:0:TensorArrayV2Read/TensorListGetItem_0/accumulator:handle:0.MatMul_1/ReadVariableOp_0/accumulator:handle:0$placeholder_2_0/accumulator:handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*$
T
2*
_lower_using_switch_merge(*
_num_original_outputs*>
body6R4
2__inference_while_body_10004269_26988676_rewritten*>
cond6R4
2__inference_while_cond_10004268_26997142_rewritten*m
output_shapes\
Z: : : : :����������:����������: : : : : : : : : : : : : : : : : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile_17:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_17*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_17*
T0*,
_output_shapes
:����������2

Identity_1�

Identity_2Identitywhile_17:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_17*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile_17:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_17*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp	^while_17*
T0*
_output_shapes
: 2

Identity_4�
!mul_2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2#
!mul_2_0/accumulator/element_shape�
mul_2_0/accumulatorEmptyTensorList*mul_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
mul_2_0/accumulatorU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
placeholder_0/accumulatorEmptyTensorListConst_1:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
placeholder_0/accumulator�
%Sigmoid_2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2'
%Sigmoid_2_0/accumulator/element_shape�
Sigmoid_2_0/accumulatorEmptyTensorList.Sigmoid_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
Sigmoid_2_0/accumulator�
"Tanh_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2$
"Tanh_1_0/accumulator/element_shape�
Tanh_1_0/accumulatorEmptyTensorList+Tanh_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
Tanh_1_0/accumulator�
mul_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2!
mul_0/accumulator/element_shape�
mul_0/accumulatorEmptyTensorList(mul_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
mul_0/accumulator�
!mul_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2#
!mul_1_0/accumulator/element_shape�
mul_1_0/accumulatorEmptyTensorList*mul_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
mul_1_0/accumulator�
%Sigmoid_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2'
%Sigmoid_1_0/accumulator/element_shape�
Sigmoid_1_0/accumulatorEmptyTensorList.Sigmoid_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
Sigmoid_1_0/accumulator�
)placeholder_3_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2+
)placeholder_3_0/accumulator/element_shape�
placeholder_3_0/accumulatorEmptyTensorList2placeholder_3_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
placeholder_3_0/accumulator�
#Sigmoid_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2%
#Sigmoid_0/accumulator/element_shape�
Sigmoid_0/accumulatorEmptyTensorList,Sigmoid_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
Sigmoid_0/accumulator�
 Tanh_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2"
 Tanh_0/accumulator/element_shape�
Tanh_0/accumulatorEmptyTensorList)Tanh_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
Tanh_0/accumulator�
"MatMul_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2$
"MatMul_0/accumulator/element_shape�
MatMul_0/accumulatorEmptyTensorList+MatMul_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
MatMul_0/accumulator�
$MatMul_1_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"��������2&
$MatMul_1_0/accumulator/element_shape�
MatMul_1_0/accumulatorEmptyTensorList-MatMul_1_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
MatMul_1_0/accumulator�
1MatMul/ReadVariableOp_0/accumulator/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1MatMul/ReadVariableOp_0/accumulator/element_shape�
#MatMul/ReadVariableOp_0/accumulatorEmptyTensorList:MatMul/ReadVariableOp_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#MatMul/ReadVariableOp_0/accumulator�
?TensorArrayV2Read/TensorListGetItem_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   2A
?TensorArrayV2Read/TensorListGetItem_0/accumulator/element_shape�
1TensorArrayV2Read/TensorListGetItem_0/accumulatorEmptyTensorListHTensorArrayV2Read/TensorListGetItem_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1TensorArrayV2Read/TensorListGetItem_0/accumulator�
3MatMul_1/ReadVariableOp_0/accumulator/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������25
3MatMul_1/ReadVariableOp_0/accumulator/element_shape�
%MatMul_1/ReadVariableOp_0/accumulatorEmptyTensorList<MatMul_1/ReadVariableOp_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%MatMul_1/ReadVariableOp_0/accumulator�
)placeholder_2_0/accumulator/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2+
)placeholder_2_0/accumulator/element_shape�
placeholder_2_0/accumulatorEmptyTensorList2placeholder_2_0/accumulator/element_shape:output:0!while/maximum_iterations:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
placeholder_2_0/accumulator"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"Q
"tensorarrayv2stack_tensorliststack+TensorArrayV2Stack/TensorListStack:tensor:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0")
transpose_permtranspose/perm:output:0"
whilewhile_17:output:7"
while_0while_17:output:0"
while_1while_17:output:11"
while_10while_17:output:20"
while_11while_17:output:21"
while_12while_17:output:22"
while_13while_17:output:23"
while_14while_17:output:24"
while_15while_17:output:25"
while_16while_17:output:26"
while_2while_17:output:12"
while_3while_17:output:13"
while_4while_17:output:14"
while_5while_17:output:15"
while_6while_17:output:16"
while_7while_17:output:17"
while_8while_17:output:18"
while_9while_17:output:19"=
while_maximum_iterations!while/maximum_iterations:output:0*^
_input_shapesM
K:���������6:����������:����������:::*R
backward_function_name86__inference___backward_standard_lstm_28230303_2823075020
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile_17:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�L
�
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27035701

inputs)
%lstm_2_statefulpartitionedcall_args_3)
%lstm_2_statefulpartitionedcall_args_4)
%lstm_2_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�lstm_2/StatefulPartitionedCall�;simple_neural_network_layer_block_6/StatefulPartitionedCall�;simple_neural_network_layer_block_7/StatefulPartitionedCall�;simple_neural_network_layer_block_8/StatefulPartitionedCallR
lstm_2/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_2/Shape�
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stack�
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1�
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2�
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicek
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros/mul/y�
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros/Less/y�
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessq
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros/packed/1�
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/Const�
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:����������2
lstm_2/zeroso
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros_1/mul/y�
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros_1/Less/y�
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lessu
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros_1/packed/1�
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/Const�
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
lstm_2/zeros_1�
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputslstm_2/zeros:output:0lstm_2/zeros_1:output:0%lstm_2_statefulpartitionedcall_args_3%lstm_2_statefulpartitionedcall_args_4%lstm_2_statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*j
_output_shapesX
V:����������:����������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270319162 
lstm_2/StatefulPartitionedCall�
;simple_neural_network_layer_block_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_call_270254852=
;simple_neural_network_layer_block_6/StatefulPartitionedCall�
;simple_neural_network_layer_block_7/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_6/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_call_270061122=
;simple_neural_network_layer_block_7/StatefulPartitionedCall�
;simple_neural_network_layer_block_8/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_7/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_call_270356402=
;simple_neural_network_layer_block_8/StatefulPartitionedCall�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMulDsimple_neural_network_layer_block_8/StatefulPartitionedCall:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdds
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_11/Tanh�
IdentityIdentitydense_11/Tanh:y:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^lstm_2/StatefulPartitionedCall<^simple_neural_network_layer_block_6/StatefulPartitionedCall<^simple_neural_network_layer_block_7/StatefulPartitionedCall<^simple_neural_network_layer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2z
;simple_neural_network_layer_block_6/StatefulPartitionedCall;simple_neural_network_layer_block_6/StatefulPartitionedCall2z
;simple_neural_network_layer_block_7/StatefulPartitionedCall;simple_neural_network_layer_block_7/StatefulPartitionedCall2z
;simple_neural_network_layer_block_8/StatefulPartitionedCall;simple_neural_network_layer_block_8/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�G
�
"__inference_standard_lstm_27034129

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10033736_27034054*4
cond,R*
(__inference_while_cond_10033735_26991798*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*g
_input_shapesV
T:������������������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�.
�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27023613

inputs
assignmovingavg_10044780
assignmovingavg_1_10044786 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/10044780*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10044780*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10044780*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10044780*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10044780AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10044780*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/10044786*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10044786*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10044786*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10044786*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10044786AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10044786*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:& "
 
_user_specified_nameinputs
�W
�
2__inference_while_body_10004269_26988676_rewritten
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0_0S
Otensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0&
"matmul_readvariableop_resource_0_0(
$matmul_1_readvariableop_resource_0_0'
#biasadd_readvariableop_resource_0_0*
&tensorlistpushback_mul_2_0_accumulator2
.tensorlistpushback_1_placeholder_0_accumulator0
,tensorlistpushback_2_sigmoid_2_0_accumulator-
)tensorlistpushback_3_tanh_1_0_accumulator*
&tensorlistpushback_4_mul_0_accumulator,
(tensorlistpushback_5_mul_1_0_accumulator0
,tensorlistpushback_6_sigmoid_1_0_accumulator4
0tensorlistpushback_7_placeholder_3_0_accumulator.
*tensorlistpushback_8_sigmoid_0_accumulator+
'tensorlistpushback_9_tanh_0_accumulator.
*tensorlistpushback_10_matmul_0_accumulator0
,tensorlistpushback_11_matmul_1_0_accumulator=
9tensorlistpushback_12_matmul_readvariableop_0_accumulatorK
Gtensorlistpushback_13_tensorarrayv2read_tensorlistgetitem_0_accumulator?
;tensorlistpushback_14_matmul_1_readvariableop_0_accumulator5
1tensorlistpushback_15_placeholder_2_0_accumulator
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
tensorlistpushback
tensorlistpushback_1
tensorlistpushback_2
tensorlistpushback_3
tensorlistpushback_4
tensorlistpushback_5
tensorlistpushback_6
tensorlistpushback_7
tensorlistpushback_8
tensorlistpushback_9
tensorlistpushback_10
tensorlistpushback_11
tensorlistpushback_12
tensorlistpushback_13
tensorlistpushback_14
tensorlistpushback_15��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemOtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_resource_0_0*
_output_shapes
:*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:������������������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp$matmul_1_readvariableop_resource_0_0*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:������������������2

MatMul_1t
addAddV2MatMul:product:0MatMul_1:product:0*
T0*0
_output_shapes
:������������������2
add�
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_resource_0_0*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:������������������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*�
_output_shapesr
p:������������������:������������������:������������������:������������������*
	num_split2
splith
SigmoidSigmoidsplit:output:0*
T0*0
_output_shapes
:������������������2	
Sigmoidl
	Sigmoid_1Sigmoidsplit:output:1*
T0*0
_output_shapes
:������������������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mul_
TanhTanhsplit:output:2*
T0*0
_output_shapes
:������������������2
Tanhg
mul_1MulSigmoid:y:0Tanh:y:0*
T0*0
_output_shapes
:������������������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1l
	Sigmoid_2Sigmoidsplit:output:3*
T0*0
_output_shapes
:������������������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5�
TensorListPushBackTensorListPushBack&tensorlistpushback_mul_2_0_accumulator	mul_2:z:0*
_output_shapes
: *
element_dtype02
TensorListPushBack�
TensorListPushBack_1TensorListPushBack.tensorlistpushback_1_placeholder_0_accumulatorplaceholder*
_output_shapes
: *
element_dtype02
TensorListPushBack_1�
TensorListPushBack_2TensorListPushBack,tensorlistpushback_2_sigmoid_2_0_accumulatorSigmoid_2:y:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_2�
TensorListPushBack_3TensorListPushBack)tensorlistpushback_3_tanh_1_0_accumulator
Tanh_1:y:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_3�
TensorListPushBack_4TensorListPushBack&tensorlistpushback_4_mul_0_accumulatormul:z:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_4�
TensorListPushBack_5TensorListPushBack(tensorlistpushback_5_mul_1_0_accumulator	mul_1:z:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_5�
TensorListPushBack_6TensorListPushBack,tensorlistpushback_6_sigmoid_1_0_accumulatorSigmoid_1:y:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_6�
TensorListPushBack_7TensorListPushBack0tensorlistpushback_7_placeholder_3_0_accumulatorplaceholder_3*
_output_shapes
: *
element_dtype02
TensorListPushBack_7�
TensorListPushBack_8TensorListPushBack*tensorlistpushback_8_sigmoid_0_accumulatorSigmoid:y:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_8�
TensorListPushBack_9TensorListPushBack'tensorlistpushback_9_tanh_0_accumulatorTanh:y:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_9�
TensorListPushBack_10TensorListPushBack*tensorlistpushback_10_matmul_0_accumulatorMatMul:product:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_10�
TensorListPushBack_11TensorListPushBack,tensorlistpushback_11_matmul_1_0_accumulatorMatMul_1:product:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_11�
TensorListPushBack_12TensorListPushBack9tensorlistpushback_12_matmul_readvariableop_0_accumulatorMatMul/ReadVariableOp:value:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_12�
TensorListPushBack_13TensorListPushBackGtensorlistpushback_13_tensorarrayv2read_tensorlistgetitem_0_accumulator*TensorArrayV2Read/TensorListGetItem:item:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_13�
TensorListPushBack_14TensorListPushBack;tensorlistpushback_14_matmul_1_readvariableop_0_accumulatorMatMul_1/ReadVariableOp:value:0*
_output_shapes
: *
element_dtype02
TensorListPushBack_14�
TensorListPushBack_15TensorListPushBack1tensorlistpushback_15_placeholder_2_0_accumulatorplaceholder_2*
_output_shapes
: *
element_dtype02
TensorListPushBack_15"H
!biasadd_readvariableop_resource_0#biasadd_readvariableop_resource_0_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"J
"matmul_1_readvariableop_resource_0$matmul_1_readvariableop_resource_0_0"F
 matmul_readvariableop_resource_0"matmul_readvariableop_resource_0_0"$
strided_slice_0strided_slice_0_0"�
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0Otensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0_0"8
tensorlistpushback"TensorListPushBack:output_handle:0"<
tensorlistpushback_1$TensorListPushBack_1:output_handle:0">
tensorlistpushback_10%TensorListPushBack_10:output_handle:0">
tensorlistpushback_11%TensorListPushBack_11:output_handle:0">
tensorlistpushback_12%TensorListPushBack_12:output_handle:0">
tensorlistpushback_13%TensorListPushBack_13:output_handle:0">
tensorlistpushback_14%TensorListPushBack_14:output_handle:0">
tensorlistpushback_15%TensorListPushBack_15:output_handle:0"<
tensorlistpushback_2$TensorListPushBack_2:output_handle:0"<
tensorlistpushback_3$TensorListPushBack_3:output_handle:0"<
tensorlistpushback_4$TensorListPushBack_4:output_handle:0"<
tensorlistpushback_5$TensorListPushBack_5:output_handle:0"<
tensorlistpushback_6$TensorListPushBack_6:output_handle:0"<
tensorlistpushback_7$TensorListPushBack_7:output_handle:0"<
tensorlistpushback_8$TensorListPushBack_8:output_handle:0"<
tensorlistpushback_9$TensorListPushBack_9:output_handle:0*s
_input_shapesb
`: : : : :����������:����������: : :::: : : : : : : : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�	
�
F__inference_simple_neural_network_layer_block_7_layer_call_fn_27011439

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_270114282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�)
�
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_27019108

inputs*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource6
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource8
4batch_normalization_7_cast_2_readvariableop_resource8
4batch_normalization_7_cast_3_readvariableop_resource
identity��)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�+batch_normalization_7/Cast_2/ReadVariableOp�+batch_normalization_7/Cast_3/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAdd�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOp�
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1r
TanhTanh)batch_normalization_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2Z
+batch_normalization_7/Cast_2/ReadVariableOp+batch_normalization_7/Cast_2/ReadVariableOp2Z
+batch_normalization_7/Cast_3/ReadVariableOp+batch_normalization_7/Cast_3/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_10035086_26998145
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10035086___redundant_placeholder00
,while_cond_10035086___redundant_placeholder10
,while_cond_10035086___redundant_placeholder20
,while_cond_10035086___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�*
�
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_27000036

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource6
2batch_normalization_8_cast_readvariableop_resource8
4batch_normalization_8_cast_1_readvariableop_resource8
4batch_normalization_8_cast_2_readvariableop_resource8
4batch_normalization_8_cast_3_readvariableop_resource
identity��)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�+batch_normalization_8/Cast_2/ReadVariableOp�+batch_normalization_8/Cast_3/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAdd�
"batch_normalization_8/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_8/LogicalAnd/x�
"batch_normalization_8/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/y�
 batch_normalization_8/LogicalAnd
LogicalAnd+batch_normalization_8/LogicalAnd/x:output:0+batch_normalization_8/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_8/LogicalAnd�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOp�
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_10/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/add_1r
TanhTanh)batch_normalization_8/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2Z
+batch_normalization_8/Cast_2/ReadVariableOp+batch_normalization_8/Cast_2/ReadVariableOp2Z
+batch_normalization_8/Cast_3/ReadVariableOp+batch_normalization_8/Cast_3/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
F__inference_simple_neural_network_layer_block_8_layer_call_fn_27007387

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_270073762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_8_layer_call_fn_27005829

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_270058202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_7_layer_call_fn_27001954

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_270019452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
F__inference_dense_11_layer_call_and_return_conditional_losses_27024864

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�G
�
"__inference_standard_lstm_27012201

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10003824_27012126*4
cond,R*
(__inference_while_cond_10003823_26988751*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:����������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*^
_input_shapesM
K:���������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�G
�
"__inference_standard_lstm_27031916

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10005875_27031841*4
cond,R*
(__inference_while_cond_10005874_26992398*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:����������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*^
_input_shapesM
K:���������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�
�
8__inference_batch_normalization_8_layer_call_fn_27001860

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_270018512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�O
�
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_27007376

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource2
.batch_normalization_8_assignmovingavg_100048974
0batch_normalization_8_assignmovingavg_1_100049036
2batch_normalization_8_cast_readvariableop_resource8
4batch_normalization_8_cast_1_readvariableop_resource
identity��9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_8/AssignMovingAvg/ReadVariableOp�;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAdd�
"batch_normalization_8/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/x�
"batch_normalization_8/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/y�
 batch_normalization_8/LogicalAnd
LogicalAnd+batch_normalization_8/LogicalAnd/x:output:0+batch_normalization_8/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_8/LogicalAnd�
4batch_normalization_8/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_8/moments/mean/reduction_indices�
"batch_normalization_8/moments/meanMeandense_10/BiasAdd:output:0=batch_normalization_8/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_8/moments/mean�
*batch_normalization_8/moments/StopGradientStopGradient+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_8/moments/StopGradient�
/batch_normalization_8/moments/SquaredDifferenceSquaredDifferencedense_10/BiasAdd:output:03batch_normalization_8/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_8/moments/SquaredDifference�
8batch_normalization_8/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_8/moments/variance/reduction_indices�
&batch_normalization_8/moments/varianceMean3batch_normalization_8/moments/SquaredDifference:z:0Abatch_normalization_8/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_8/moments/variance�
%batch_normalization_8/moments/SqueezeSqueeze+batch_normalization_8/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_8/moments/Squeeze�
'batch_normalization_8/moments/Squeeze_1Squeeze/batch_normalization_8/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_8/moments/Squeeze_1�
+batch_normalization_8/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10004897*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_8/AssignMovingAvg/decay�
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_8_assignmovingavg_10004897*
_output_shapes	
:�*
dtype026
4batch_normalization_8/AssignMovingAvg/ReadVariableOp�
)batch_normalization_8/AssignMovingAvg/subSub<batch_normalization_8/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_8/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10004897*
_output_shapes	
:�2+
)batch_normalization_8/AssignMovingAvg/sub�
)batch_normalization_8/AssignMovingAvg/mulMul-batch_normalization_8/AssignMovingAvg/sub:z:04batch_normalization_8/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10004897*
_output_shapes	
:�2+
)batch_normalization_8/AssignMovingAvg/mul�
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_8_assignmovingavg_10004897-batch_normalization_8/AssignMovingAvg/mul:z:05^batch_normalization_8/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_8/AssignMovingAvg/10004897*
_output_shapes
 *
dtype02;
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_8/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10004903*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_8/AssignMovingAvg_1/decay�
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_8_assignmovingavg_1_10004903*
_output_shapes	
:�*
dtype028
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_8/AssignMovingAvg_1/subSub>batch_normalization_8/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_8/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10004903*
_output_shapes	
:�2-
+batch_normalization_8/AssignMovingAvg_1/sub�
+batch_normalization_8/AssignMovingAvg_1/mulMul/batch_normalization_8/AssignMovingAvg_1/sub:z:06batch_normalization_8/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10004903*
_output_shapes	
:�2-
+batch_normalization_8/AssignMovingAvg_1/mul�
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_8_assignmovingavg_1_10004903/batch_normalization_8/AssignMovingAvg_1/mul:z:07^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_8/AssignMovingAvg_1/10004903*
_output_shapes
 *
dtype02=
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV20batch_normalization_8/moments/Squeeze_1:output:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_10/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul.batch_normalization_8/moments/Squeeze:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub1batch_normalization_8/Cast/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/add_1r
TanhTanh)batch_normalization_8/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_8/AssignMovingAvg/ReadVariableOp<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_8/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp9batch_normalization_8/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_8/AssignMovingAvg/ReadVariableOp4batch_normalization_8/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�G
�
"__inference_standard_lstm_27026975

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10005264_27008708*4
cond,R*
(__inference_while_cond_10005263_27026900*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:����������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*^
_input_shapesM
K:���������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�
�
D__inference_lstm_2_layer_call_and_return_conditional_losses_26997250

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*j
_output_shapesX
V:����������:����������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_269972172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������6:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
>__inference_lstm_neural_network_model_2_layer_call_fn_27019035
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_270189792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�	
�
2__inference_while_cond_10004268_26997142_rewritten
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10004268___redundant_placeholder00
,while_cond_10004268___redundant_placeholder10
,while_cond_10004268___redundant_placeholder20
,while_cond_10004268___redundant_placeholder3
mul_2_0_accumulator
placeholder_0_accumulator
sigmoid_2_0_accumulator
tanh_1_0_accumulator
mul_0_accumulator
mul_1_0_accumulator
sigmoid_1_0_accumulator
placeholder_3_0_accumulator
sigmoid_0_accumulator
tanh_0_accumulator
matmul_0_accumulator
matmul_1_0_accumulator'
#matmul_readvariableop_0_accumulator5
1tensorarrayv2read_tensorlistgetitem_0_accumulator)
%matmul_1_readvariableop_0_accumulator
placeholder_2_0_accumulator
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*s
_input_shapesb
`: : : : :����������:����������: : :::: : : : : : : : : : : : : : : : 
�3
�
___forward_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_28230175
inputs_0*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource6
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource8
4batch_normalization_7_cast_2_readvariableop_resource8
4batch_normalization_7_cast_3_readvariableop_resource
identity
tanh)
%batch_normalization_7_batchnorm_mul_1'
#batch_normalization_7_batchnorm_sub
dense_9_biasadd'
#batch_normalization_7_batchnorm_mul-
)batch_normalization_7_cast_readvariableop!
dense_9_matmul_readvariableop

inputs/
+batch_normalization_7_cast_3_readvariableop)
%batch_normalization_7_batchnorm_rsqrt��)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�+batch_normalization_7/Cast_2/ReadVariableOp�+batch_normalization_7/Cast_3/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulinputs_0%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAdd�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
+batch_normalization_7/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_7_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_2/ReadVariableOp�
+batch_normalization_7/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_7_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_3/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV23batch_normalization_7/Cast_1/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul1batch_normalization_7/Cast/ReadVariableOp:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub3batch_normalization_7/Cast_2/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1r
TanhTanh)batch_normalization_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp,^batch_normalization_7/Cast_2/ReadVariableOp,^batch_normalization_7/Cast_3/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"N
#batch_normalization_7_batchnorm_mul'batch_normalization_7/batchnorm/mul:z:0"R
%batch_normalization_7_batchnorm_mul_1)batch_normalization_7/batchnorm/mul_1:z:0"R
%batch_normalization_7_batchnorm_rsqrt)batch_normalization_7/batchnorm/Rsqrt:y:0"N
#batch_normalization_7_batchnorm_sub'batch_normalization_7/batchnorm/sub:z:0"b
+batch_normalization_7_cast_3_readvariableop3batch_normalization_7/Cast_3/ReadVariableOp:value:0"^
)batch_normalization_7_cast_readvariableop1batch_normalization_7/Cast/ReadVariableOp:value:0"+
dense_9_biasadddense_9/BiasAdd:output:0"F
dense_9_matmul_readvariableop%dense_9/MatMul/ReadVariableOp:value:0"
identityIdentity:output:0"
inputsinputs_0"
tanhTanh:y:0*?
_input_shapes.
,:����������::::::*�
backward_function_namewu__inference___backward_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_28230117_282301762V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2Z
+batch_normalization_7/Cast_2/ReadVariableOp+batch_normalization_7/Cast_2/ReadVariableOp2Z
+batch_normalization_7/Cast_3/ReadVariableOp+batch_normalization_7/Cast_3/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�O
�
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_26991188

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource2
.batch_normalization_6_assignmovingavg_100046834
0batch_normalization_6_assignmovingavg_1_100046896
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource
identity��9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAdd�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indices�
"batch_normalization_6/moments/meanMeandense_8/BiasAdd:output:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_6/moments/mean�
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_6/moments/StopGradient�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_8/BiasAdd:output:03batch_normalization_6/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_6/moments/SquaredDifference�
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices�
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_6/moments/variance�
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze�
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1�
+batch_normalization_6/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10004683*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_6/AssignMovingAvg/decay�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_10004683*
_output_shapes	
:�*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp�
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10004683*
_output_shapes	
:�2+
)batch_normalization_6/AssignMovingAvg/sub�
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10004683*
_output_shapes	
:�2+
)batch_normalization_6/AssignMovingAvg/mul�
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_10004683-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10004683*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_6/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10004689*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_6/AssignMovingAvg_1/decay�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_6_assignmovingavg_1_10004689*
_output_shapes	
:�*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10004689*
_output_shapes	
:�2-
+batch_normalization_6/AssignMovingAvg_1/sub�
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10004689*
_output_shapes	
:�2-
+batch_normalization_6/AssignMovingAvg_1/mul�
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_6_assignmovingavg_1_10004689/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10004689*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/BiasAdd:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1r
TanhTanh)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
(__inference_while_body_10002485_26991144
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
D__inference_lstm_2_layer_call_and_return_conditional_losses_27006581

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*s
_output_shapesa
_:����������:�������������������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270065482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������6:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�.
�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_26989140

inputs
assignmovingavg_10003397
assignmovingavg_1_10003403 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/10003397*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10003397*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10003397*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10003397*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10003397AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10003397*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/10003403*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10003403*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10003403*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10003403*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10003403AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10003403*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_10005874_26992398
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10005874___redundant_placeholder00
,while_cond_10005874___redundant_placeholder10
,while_cond_10005874___redundant_placeholder20
,while_cond_10005874___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�	
�
F__inference_simple_neural_network_layer_block_7_layer_call_fn_27019046

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_270189472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�3
�
!__inference__traced_save_74975559
file_prefix.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop,
(savev2_lstm_2_kernel_read_readvariableop6
2savev2_lstm_2_recurrent_kernel_read_readvariableop*
&savev2_lstm_2_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_44f28ffe9118401e9344fe42911daca5/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop(savev2_lstm_2_kernel_read_readvariableop2savev2_lstm_2_recurrent_kernel_read_readvariableop&savev2_lstm_2_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop"/device:CPU:0*
_output_shapes
 *%
dtypes
22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�::	6�:
��:�:
��:�:�:�:
��:�:�:�:
��:�:�:�:�:�:�:�:�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
(__inference_while_cond_10003823_26988751
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10003823___redundant_placeholder00
,while_cond_10003823___redundant_placeholder10
,while_cond_10003823___redundant_placeholder20
,while_cond_10003823___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�
�
(__inference_while_cond_10034641_27024113
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10034641___redundant_placeholder00
,while_cond_10034641___redundant_placeholder10
,while_cond_10034641___redundant_placeholder20
,while_cond_10034641___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�
�
>__inference_lstm_neural_network_model_2_layer_call_fn_27013104

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_270130762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�.
�
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_27005820

inputs
assignmovingavg_10003685
assignmovingavg_1_10003691 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	�2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/10003685*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10003685*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10003685*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10003685*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10003685AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10003685*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/10003691*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10003691*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10003691*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10003691*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10003691AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10003691*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:& "
 
_user_specified_nameinputs
�`
�
$__inference__traced_restore_74975640
file_prefix$
 assignvariableop_dense_11_kernel$
 assignvariableop_1_dense_11_bias$
 assignvariableop_2_lstm_2_kernel.
*assignvariableop_3_lstm_2_recurrent_kernel"
assignvariableop_4_lstm_2_bias%
!assignvariableop_5_dense_8_kernel#
assignvariableop_6_dense_8_bias2
.assignvariableop_7_batch_normalization_6_gamma1
-assignvariableop_8_batch_normalization_6_beta%
!assignvariableop_9_dense_9_kernel$
 assignvariableop_10_dense_9_bias3
/assignvariableop_11_batch_normalization_7_gamma2
.assignvariableop_12_batch_normalization_7_beta'
#assignvariableop_13_dense_10_kernel%
!assignvariableop_14_dense_10_bias3
/assignvariableop_15_batch_normalization_8_gamma2
.assignvariableop_16_batch_normalization_8_beta9
5assignvariableop_17_batch_normalization_6_moving_mean=
9assignvariableop_18_batch_normalization_6_moving_variance9
5assignvariableop_19_batch_normalization_7_moving_mean=
9assignvariableop_20_batch_normalization_7_moving_variance9
5assignvariableop_21_batch_normalization_8_moving_mean=
9assignvariableop_22_batch_normalization_8_moving_variance
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_lstm_2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp*assignvariableop_3_lstm_2_recurrent_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_lstm_2_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_8_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_8_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_6_gammaIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_6_betaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_9_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_9_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_7_gammaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_7_betaIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_10_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_10_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_8_gammaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_batch_normalization_8_betaIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp5assignvariableop_17_batch_normalization_6_moving_meanIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp9assignvariableop_18_batch_normalization_6_moving_varianceIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp5assignvariableop_19_batch_normalization_7_moving_meanIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp9assignvariableop_20_batch_normalization_7_moving_varianceIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_8_moving_meanIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_8_moving_varianceIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23�
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�*
�
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_27003143

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource6
2batch_normalization_8_cast_readvariableop_resource8
4batch_normalization_8_cast_1_readvariableop_resource8
4batch_normalization_8_cast_2_readvariableop_resource8
4batch_normalization_8_cast_3_readvariableop_resource
identity��)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�+batch_normalization_8/Cast_2/ReadVariableOp�+batch_normalization_8/Cast_3/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAdd�
"batch_normalization_8/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_8/LogicalAnd/x�
"batch_normalization_8/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/y�
 batch_normalization_8/LogicalAnd
LogicalAnd+batch_normalization_8/LogicalAnd/x:output:0+batch_normalization_8/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_8/LogicalAnd�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOp�
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_10/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/add_1r
TanhTanh)batch_normalization_8/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2Z
+batch_normalization_8/Cast_2/ReadVariableOp+batch_normalization_8/Cast_2/ReadVariableOp2Z
+batch_normalization_8/Cast_3/ReadVariableOp+batch_normalization_8/Cast_3/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�3
�
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27018979

inputs)
%lstm_2_statefulpartitionedcall_args_1)
%lstm_2_statefulpartitionedcall_args_2)
%lstm_2_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6+
'dense_11_statefulpartitionedcall_args_1+
'dense_11_statefulpartitionedcall_args_2
identity�� dense_11/StatefulPartitionedCall�lstm_2/StatefulPartitionedCall�;simple_neural_network_layer_block_6/StatefulPartitionedCall�;simple_neural_network_layer_block_7/StatefulPartitionedCall�;simple_neural_network_layer_block_8/StatefulPartitionedCall�
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputs%lstm_2_statefulpartitionedcall_args_1%lstm_2_statefulpartitionedcall_args_2%lstm_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_2_layer_call_and_return_conditional_losses_270122342 
lstm_2/StatefulPartitionedCall�
;simple_neural_network_layer_block_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_269911882=
;simple_neural_network_layer_block_6/StatefulPartitionedCall�
;simple_neural_network_layer_block_7/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_6/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_270189472=
;simple_neural_network_layer_block_7/StatefulPartitionedCall�
;simple_neural_network_layer_block_8/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_7/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_270073762=
;simple_neural_network_layer_block_8/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_8/StatefulPartitionedCall:output:0'dense_11_statefulpartitionedcall_args_1'dense_11_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_270117342"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_11/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall<^simple_neural_network_layer_block_6/StatefulPartitionedCall<^simple_neural_network_layer_block_7/StatefulPartitionedCall<^simple_neural_network_layer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2z
;simple_neural_network_layer_block_6/StatefulPartitionedCall;simple_neural_network_layer_block_6/StatefulPartitionedCall2z
;simple_neural_network_layer_block_7/StatefulPartitionedCall;simple_neural_network_layer_block_7/StatefulPartitionedCall2z
;simple_neural_network_layer_block_8/StatefulPartitionedCall;simple_neural_network_layer_block_8/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�)
�
__inference_call_27035640

inputs+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource6
2batch_normalization_8_cast_readvariableop_resource8
4batch_normalization_8_cast_1_readvariableop_resource8
4batch_normalization_8_cast_2_readvariableop_resource8
4batch_normalization_8_cast_3_readvariableop_resource
identity��)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�+batch_normalization_8/Cast_2/ReadVariableOp�+batch_normalization_8/Cast_3/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMulinputs&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAdd�
"batch_normalization_8/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_8/LogicalAnd/x�
"batch_normalization_8/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/y�
 batch_normalization_8/LogicalAnd
LogicalAnd+batch_normalization_8/LogicalAnd/x:output:0+batch_normalization_8/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_8/LogicalAnd�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOp�
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_10/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/add_1r
TanhTanh)batch_normalization_8/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2Z
+batch_normalization_8/Cast_2/ReadVariableOp+batch_normalization_8/Cast_2/ReadVariableOp2Z
+batch_normalization_8/Cast_3/ReadVariableOp+batch_normalization_8/Cast_3/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_10004268_26997142
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10004268___redundant_placeholder00
,while_cond_10004268___redundant_placeholder10
,while_cond_10004268___redundant_placeholder20
,while_cond_10004268___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�.
�
(__inference_while_body_10003824_27012126
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�
�
>__inference_lstm_neural_network_model_2_layer_call_fn_27013132
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_270130762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�3
�
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27013164
input_1)
%lstm_2_statefulpartitionedcall_args_1)
%lstm_2_statefulpartitionedcall_args_2)
%lstm_2_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6+
'dense_11_statefulpartitionedcall_args_1+
'dense_11_statefulpartitionedcall_args_2
identity�� dense_11/StatefulPartitionedCall�lstm_2/StatefulPartitionedCall�;simple_neural_network_layer_block_6/StatefulPartitionedCall�;simple_neural_network_layer_block_7/StatefulPartitionedCall�;simple_neural_network_layer_block_8/StatefulPartitionedCall�
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinput_1%lstm_2_statefulpartitionedcall_args_1%lstm_2_statefulpartitionedcall_args_2%lstm_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_2_layer_call_and_return_conditional_losses_269972502 
lstm_2/StatefulPartitionedCall�
;simple_neural_network_layer_block_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_270130332=
;simple_neural_network_layer_block_6/StatefulPartitionedCall�
;simple_neural_network_layer_block_7/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_6/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_270114282=
;simple_neural_network_layer_block_7/StatefulPartitionedCall�
;simple_neural_network_layer_block_8/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_7/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_270031432=
;simple_neural_network_layer_block_8/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_8/StatefulPartitionedCall:output:0'dense_11_statefulpartitionedcall_args_1'dense_11_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_270117342"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_11/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall<^simple_neural_network_layer_block_6/StatefulPartitionedCall<^simple_neural_network_layer_block_7/StatefulPartitionedCall<^simple_neural_network_layer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2z
;simple_neural_network_layer_block_6/StatefulPartitionedCall;simple_neural_network_layer_block_6/StatefulPartitionedCall2z
;simple_neural_network_layer_block_7/StatefulPartitionedCall;simple_neural_network_layer_block_7/StatefulPartitionedCall2z
;simple_neural_network_layer_block_8/StatefulPartitionedCall;simple_neural_network_layer_block_8/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
+__inference_dense_11_layer_call_fn_27011741

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_270117342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_10034180_26998190
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10034180___redundant_placeholder00
,while_cond_10034180___redundant_placeholder10
,while_cond_10034180___redundant_placeholder20
,while_cond_10034180___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�
�
)__inference_lstm_2_layer_call_fn_27012242

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_2_layer_call_and_return_conditional_losses_270122342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������6:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
F__inference_simple_neural_network_layer_block_6_layer_call_fn_26991199

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_269911882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_10002484_27006473
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10002484___redundant_placeholder00
,while_cond_10002484___redundant_placeholder10
,while_cond_10002484___redundant_placeholder20
,while_cond_10002484___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�G
�
"__inference_standard_lstm_27011622

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10002938_27011547*4
cond,R*
(__inference_while_cond_10002937_26988456*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*g
_input_shapesV
T:������������������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�.
�
(__inference_while_body_10033736_27034054
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�.
�
(__inference_while_body_10034642_26998656
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������6*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1b
mulMulSigmoid_1:y:0placeholder_3*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3�
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_4�

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :����������:����������: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
�3
�
___forward_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_28230088
inputs_0+
'dense_10_matmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource6
2batch_normalization_8_cast_readvariableop_resource8
4batch_normalization_8_cast_1_readvariableop_resource8
4batch_normalization_8_cast_2_readvariableop_resource8
4batch_normalization_8_cast_3_readvariableop_resource
identity
tanh)
%batch_normalization_8_batchnorm_mul_1'
#batch_normalization_8_batchnorm_sub
dense_10_biasadd'
#batch_normalization_8_batchnorm_mul-
)batch_normalization_8_cast_readvariableop"
dense_10_matmul_readvariableop

inputs/
+batch_normalization_8_cast_3_readvariableop)
%batch_normalization_8_batchnorm_rsqrt��)batch_normalization_8/Cast/ReadVariableOp�+batch_normalization_8/Cast_1/ReadVariableOp�+batch_normalization_8/Cast_2/ReadVariableOp�+batch_normalization_8/Cast_3/ReadVariableOp�dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_10/MatMul/ReadVariableOp�
dense_10/MatMulMatMulinputs_0&dense_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/MatMul�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_10/BiasAdd/ReadVariableOp�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_10/BiasAdd�
"batch_normalization_8/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_8/LogicalAnd/x�
"batch_normalization_8/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_8/LogicalAnd/y�
 batch_normalization_8/LogicalAnd
LogicalAnd+batch_normalization_8/LogicalAnd/x:output:0+batch_normalization_8/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_8/LogicalAnd�
)batch_normalization_8/Cast/ReadVariableOpReadVariableOp2batch_normalization_8_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_8/Cast/ReadVariableOp�
+batch_normalization_8/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_8_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_1/ReadVariableOp�
+batch_normalization_8/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_8_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_2/ReadVariableOp�
+batch_normalization_8/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_8_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_8/Cast_3/ReadVariableOp�
%batch_normalization_8/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_8/batchnorm/add/y�
#batch_normalization_8/batchnorm/addAddV23batch_normalization_8/Cast_1/ReadVariableOp:value:0.batch_normalization_8/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/add�
%batch_normalization_8/batchnorm/RsqrtRsqrt'batch_normalization_8/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/Rsqrt�
#batch_normalization_8/batchnorm/mulMul)batch_normalization_8/batchnorm/Rsqrt:y:03batch_normalization_8/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/mul�
%batch_normalization_8/batchnorm/mul_1Muldense_10/BiasAdd:output:0'batch_normalization_8/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/mul_1�
%batch_normalization_8/batchnorm/mul_2Mul1batch_normalization_8/Cast/ReadVariableOp:value:0'batch_normalization_8/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_8/batchnorm/mul_2�
#batch_normalization_8/batchnorm/subSub3batch_normalization_8/Cast_2/ReadVariableOp:value:0)batch_normalization_8/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_8/batchnorm/sub�
%batch_normalization_8/batchnorm/add_1AddV2)batch_normalization_8/batchnorm/mul_1:z:0'batch_normalization_8/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_8/batchnorm/add_1r
TanhTanh)batch_normalization_8/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_8/Cast/ReadVariableOp,^batch_normalization_8/Cast_1/ReadVariableOp,^batch_normalization_8/Cast_2/ReadVariableOp,^batch_normalization_8/Cast_3/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"N
#batch_normalization_8_batchnorm_mul'batch_normalization_8/batchnorm/mul:z:0"R
%batch_normalization_8_batchnorm_mul_1)batch_normalization_8/batchnorm/mul_1:z:0"R
%batch_normalization_8_batchnorm_rsqrt)batch_normalization_8/batchnorm/Rsqrt:y:0"N
#batch_normalization_8_batchnorm_sub'batch_normalization_8/batchnorm/sub:z:0"b
+batch_normalization_8_cast_3_readvariableop3batch_normalization_8/Cast_3/ReadVariableOp:value:0"^
)batch_normalization_8_cast_readvariableop1batch_normalization_8/Cast/ReadVariableOp:value:0"-
dense_10_biasadddense_10/BiasAdd:output:0"H
dense_10_matmul_readvariableop&dense_10/MatMul/ReadVariableOp:value:0"
identityIdentity:output:0"
inputsinputs_0"
tanhTanh:y:0*?
_input_shapes.
,:����������::::::*�
backward_function_namewu__inference___backward_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_28230030_282300892V
)batch_normalization_8/Cast/ReadVariableOp)batch_normalization_8/Cast/ReadVariableOp2Z
+batch_normalization_8/Cast_1/ReadVariableOp+batch_normalization_8/Cast_1/ReadVariableOp2Z
+batch_normalization_8/Cast_2/ReadVariableOp+batch_normalization_8/Cast_2/ReadVariableOp2Z
+batch_normalization_8/Cast_3/ReadVariableOp+batch_normalization_8/Cast_3/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
F__inference_dense_11_layer_call_and_return_conditional_losses_27011734

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
D__inference_lstm_2_layer_call_and_return_conditional_losses_27023072
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCallF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*s
_output_shapesa
_:����������:�������������������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270230392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������6:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
�
�
)__inference_lstm_2_layer_call_fn_26997258

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_2_layer_call_and_return_conditional_losses_269972502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������6:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
>__inference_lstm_neural_network_model_2_layer_call_fn_27019007

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_270189792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27005127

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:& "
 
_user_specified_nameinputs
�G
�
"__inference_standard_lstm_27024188

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10034642_26998656*4
cond,R*
(__inference_while_cond_10034641_27024113*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:����������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*^
_input_shapesM
K:���������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�O
�
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_27004547

inputs*
&dense_9_matmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource2
.batch_normalization_7_assignmovingavg_100385944
0batch_normalization_7_assignmovingavg_1_100386006
2batch_normalization_7_cast_readvariableop_resource8
4batch_normalization_7_cast_1_readvariableop_resource
identity��9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_7/Cast/ReadVariableOp�+batch_normalization_7/Cast_1/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMulinputs%dense_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_9/BiasAdd�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_7/moments/mean/reduction_indices�
"batch_normalization_7/moments/meanMeandense_9/BiasAdd:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_7/moments/mean�
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_7/moments/StopGradient�
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferencedense_9/BiasAdd:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_7/moments/SquaredDifference�
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_7/moments/variance/reduction_indices�
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_7/moments/variance�
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze�
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1�
+batch_normalization_7/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10038594*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_7/AssignMovingAvg/decay�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_10038594*
_output_shapes	
:�*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp�
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10038594*
_output_shapes	
:�2+
)batch_normalization_7/AssignMovingAvg/sub�
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10038594*
_output_shapes	
:�2+
)batch_normalization_7/AssignMovingAvg/mul�
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_10038594-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/10038594*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_7/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10038600*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_7/AssignMovingAvg_1/decay�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_7_assignmovingavg_1_10038600*
_output_shapes	
:�*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10038600*
_output_shapes	
:�2-
+batch_normalization_7/AssignMovingAvg_1/sub�
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10038600*
_output_shapes	
:�2-
+batch_normalization_7/AssignMovingAvg_1/mul�
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_7_assignmovingavg_1_10038600/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/10038600*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_7/Cast/ReadVariableOpReadVariableOp2batch_normalization_7_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_7/Cast/ReadVariableOp�
+batch_normalization_7/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_7_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_7/Cast_1/ReadVariableOp�
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_7/batchnorm/add/y�
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/add�
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/Rsqrt�
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:03batch_normalization_7/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/mul�
%batch_normalization_7/batchnorm/mul_1Muldense_9/BiasAdd:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/mul_1�
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_7/batchnorm/mul_2�
#batch_normalization_7/batchnorm/subSub1batch_normalization_7/Cast/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_7/batchnorm/sub�
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_7/batchnorm/add_1r
TanhTanh)batch_normalization_7/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_7/Cast/ReadVariableOp,^batch_normalization_7/Cast_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_7/Cast/ReadVariableOp)batch_normalization_7/Cast/ReadVariableOp2Z
+batch_normalization_7/Cast_1/ReadVariableOp+batch_normalization_7/Cast_1/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
D__inference_lstm_2_layer_call_and_return_conditional_losses_27011655

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity��StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*s
_output_shapesa
_:����������:�������������������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270116222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������6:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�3
�
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27013076

inputs)
%lstm_2_statefulpartitionedcall_args_1)
%lstm_2_statefulpartitionedcall_args_2)
%lstm_2_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6+
'dense_11_statefulpartitionedcall_args_1+
'dense_11_statefulpartitionedcall_args_2
identity�� dense_11/StatefulPartitionedCall�lstm_2/StatefulPartitionedCall�;simple_neural_network_layer_block_6/StatefulPartitionedCall�;simple_neural_network_layer_block_7/StatefulPartitionedCall�;simple_neural_network_layer_block_8/StatefulPartitionedCall�
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputs%lstm_2_statefulpartitionedcall_args_1%lstm_2_statefulpartitionedcall_args_2%lstm_2_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_lstm_2_layer_call_and_return_conditional_losses_269972502 
lstm_2/StatefulPartitionedCall�
;simple_neural_network_layer_block_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_270130332=
;simple_neural_network_layer_block_6/StatefulPartitionedCall�
;simple_neural_network_layer_block_7/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_6/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_270114282=
;simple_neural_network_layer_block_7/StatefulPartitionedCall�
;simple_neural_network_layer_block_8/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_7/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*j
feRc
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_270031432=
;simple_neural_network_layer_block_8/StatefulPartitionedCall�
 dense_11/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_8/StatefulPartitionedCall:output:0'dense_11_statefulpartitionedcall_args_1'dense_11_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_270117342"
 dense_11/StatefulPartitionedCall�
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^dense_11/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall<^simple_neural_network_layer_block_6/StatefulPartitionedCall<^simple_neural_network_layer_block_7/StatefulPartitionedCall<^simple_neural_network_layer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2z
;simple_neural_network_layer_block_6/StatefulPartitionedCall;simple_neural_network_layer_block_6/StatefulPartitionedCall2z
;simple_neural_network_layer_block_7/StatefulPartitionedCall;simple_neural_network_layer_block_7/StatefulPartitionedCall2z
;simple_neural_network_layer_block_8/StatefulPartitionedCall;simple_neural_network_layer_block_8/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
+__inference_restored_function_body_27040325

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������*-
config_proto

CPU

GPU2*0J 8*b
f]R[
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_270131642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
D__forward_dense_11_layer_call_and_return_conditional_losses_28230019
inputs_0"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
tanh
matmul_readvariableop

inputs��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
tanhTanh:y:0*/
_input_shapes
:����������::*v
backward_function_name\Z__inference___backward_dense_11_layer_call_and_return_conditional_losses_28230005_2823002020
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�3
�	
B__forward_lstm_2_layer_call_and_return_conditional_losses_28230807

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity
statefulpartitionedcall
statefulpartitionedcall_0
statefulpartitionedcall_1
statefulpartitionedcall_2
statefulpartitionedcall_3
statefulpartitionedcall_4
statefulpartitionedcall_5
statefulpartitionedcall_6
statefulpartitionedcall_7
statefulpartitionedcall_8
statefulpartitionedcall_9
statefulpartitionedcall_10
statefulpartitionedcall_11
statefulpartitionedcall_12
statefulpartitionedcall_13
statefulpartitionedcall_14
statefulpartitionedcall_15
statefulpartitionedcall_16
statefulpartitionedcall_17
statefulpartitionedcall_18
statefulpartitionedcall_19
statefulpartitionedcall_20
statefulpartitionedcall_21
statefulpartitionedcall_22
statefulpartitionedcall_23
statefulpartitionedcall_24
statefulpartitionedcall_25
statefulpartitionedcall_26
statefulpartitionedcall_27
statefulpartitionedcall_28��StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:����������2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2	
zeros_1�
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*+
Tout#
!2*,
_gradient_op_typePartitionedCallUnused*�
_output_shapes�
�:����������:����������:����������:����������: :����������::::: : : : : : : : : : : : : : : : : : : :���������6:*-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __forward_standard_lstm_282307492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0";
statefulpartitionedcall StatefulPartitionedCall:output:1"=
statefulpartitionedcall_0 StatefulPartitionedCall:output:2"=
statefulpartitionedcall_1 StatefulPartitionedCall:output:3"?
statefulpartitionedcall_10!StatefulPartitionedCall:output:12"?
statefulpartitionedcall_11!StatefulPartitionedCall:output:13"?
statefulpartitionedcall_12!StatefulPartitionedCall:output:14"?
statefulpartitionedcall_13!StatefulPartitionedCall:output:15"?
statefulpartitionedcall_14!StatefulPartitionedCall:output:16"?
statefulpartitionedcall_15!StatefulPartitionedCall:output:17"?
statefulpartitionedcall_16!StatefulPartitionedCall:output:18"?
statefulpartitionedcall_17!StatefulPartitionedCall:output:19"?
statefulpartitionedcall_18!StatefulPartitionedCall:output:20"?
statefulpartitionedcall_19!StatefulPartitionedCall:output:21"=
statefulpartitionedcall_2 StatefulPartitionedCall:output:4"?
statefulpartitionedcall_20!StatefulPartitionedCall:output:22"?
statefulpartitionedcall_21!StatefulPartitionedCall:output:23"?
statefulpartitionedcall_22!StatefulPartitionedCall:output:24"?
statefulpartitionedcall_23!StatefulPartitionedCall:output:25"?
statefulpartitionedcall_24!StatefulPartitionedCall:output:26"?
statefulpartitionedcall_25!StatefulPartitionedCall:output:27"?
statefulpartitionedcall_26!StatefulPartitionedCall:output:28"?
statefulpartitionedcall_27!StatefulPartitionedCall:output:29"?
statefulpartitionedcall_28!StatefulPartitionedCall:output:30"=
statefulpartitionedcall_3 StatefulPartitionedCall:output:5"=
statefulpartitionedcall_4 StatefulPartitionedCall:output:6"=
statefulpartitionedcall_5 StatefulPartitionedCall:output:7"=
statefulpartitionedcall_6 StatefulPartitionedCall:output:8"=
statefulpartitionedcall_7 StatefulPartitionedCall:output:9">
statefulpartitionedcall_8!StatefulPartitionedCall:output:10">
statefulpartitionedcall_9!StatefulPartitionedCall:output:11*6
_input_shapes%
#:���������6:::*t
backward_function_nameZX__inference___backward_lstm_2_layer_call_and_return_conditional_losses_28230291_2823080822
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�O
�
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_27006778

inputs*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource2
.batch_normalization_6_assignmovingavg_100354984
0batch_normalization_6_assignmovingavg_1_100355046
2batch_normalization_6_cast_readvariableop_resource8
4batch_normalization_6_cast_1_readvariableop_resource
identity��9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_6/Cast/ReadVariableOp�+batch_normalization_6/Cast_1/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_8/BiasAdd�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indices�
"batch_normalization_6/moments/meanMeandense_8/BiasAdd:output:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_6/moments/mean�
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_6/moments/StopGradient�
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense_8/BiasAdd:output:03batch_normalization_6/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_6/moments/SquaredDifference�
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indices�
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_6/moments/variance�
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze�
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1�
+batch_normalization_6/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10035498*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_6/AssignMovingAvg/decay�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_10035498*
_output_shapes	
:�*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp�
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10035498*
_output_shapes	
:�2+
)batch_normalization_6/AssignMovingAvg/sub�
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10035498*
_output_shapes	
:�2+
)batch_normalization_6/AssignMovingAvg/mul�
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_10035498-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/10035498*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_6/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10035504*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_6/AssignMovingAvg_1/decay�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_6_assignmovingavg_1_10035504*
_output_shapes	
:�*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10035504*
_output_shapes	
:�2-
+batch_normalization_6/AssignMovingAvg_1/sub�
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10035504*
_output_shapes	
:�2-
+batch_normalization_6/AssignMovingAvg_1/mul�
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_6_assignmovingavg_1_10035504/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/10035504*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_6/Cast/ReadVariableOpReadVariableOp2batch_normalization_6_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_6/Cast/ReadVariableOp�
+batch_normalization_6/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_6_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_6/Cast_1/ReadVariableOp�
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_6/batchnorm/add/y�
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/add�
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/Rsqrt�
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:03batch_normalization_6/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/mul�
%batch_normalization_6/batchnorm/mul_1Muldense_8/BiasAdd:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/mul_1�
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_6/batchnorm/mul_2�
#batch_normalization_6/batchnorm/subSub1batch_normalization_6/Cast/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_6/batchnorm/sub�
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_6/batchnorm/add_1r
TanhTanh)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_6/Cast/ReadVariableOp,^batch_normalization_6/Cast_1/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_6/Cast/ReadVariableOp)batch_normalization_6/Cast/ReadVariableOp2Z
+batch_normalization_6/Cast_1/ReadVariableOp+batch_normalization_6/Cast_1/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�G
�
"__inference_standard_lstm_27024065

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10035087_27023990*4
cond,R*
(__inference_while_cond_10035086_26998145*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:����������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:����������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*,
_output_shapes
:����������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*^
_input_shapesM
K:���������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�G
�
"__inference_standard_lstm_27023039

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������62
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����6   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������6*
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	6�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel* 
_output_shapes
:
��*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:����������:����������:����������:����������*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:����������2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:����������2
	Sigmoid_1[
mulMulSigmoid_1:y:0init_c*
T0*(
_output_shapes
:����������2
mulW
TanhTanhsplit:output:2*
T0*(
_output_shapes
:����������2
Tanh_
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:����������2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:����������2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:����������2
	Sigmoid_2V
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh_1c
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:����������2
mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :����������:����������: : : : : *4
body,R*
(__inference_while_body_10034181_27022964*4
cond,R*
(__inference_while_cond_10034180_26998190*M
output_shapes<
:: : : : :����������:����������: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:�������������������*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:�������������������2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
runtime�
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity�

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*5
_output_shapes#
!:�������������������2

Identity_1�

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_2�

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*(
_output_shapes
:����������2

Identity_3�

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*g
_input_shapesV
T:������������������6:����������:����������:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
�
�
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_27030001

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_10002937_26988456
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_10002937___redundant_placeholder00
,while_cond_10002937___redundant_placeholder10
,while_cond_10002937___redundant_placeholder20
,while_cond_10002937___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*U
_input_shapesD
B: : : : :����������:����������: ::::
�L
�
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27031582

inputs)
%lstm_2_statefulpartitionedcall_args_3)
%lstm_2_statefulpartitionedcall_args_4)
%lstm_2_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6+
'dense_11_matmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�lstm_2/StatefulPartitionedCall�;simple_neural_network_layer_block_6/StatefulPartitionedCall�;simple_neural_network_layer_block_7/StatefulPartitionedCall�;simple_neural_network_layer_block_8/StatefulPartitionedCallR
lstm_2/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_2/Shape�
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stack�
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1�
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2�
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicek
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros/mul/y�
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros/Less/y�
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessq
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros/packed/1�
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/Const�
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:����������2
lstm_2/zeroso
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros_1/mul/y�
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros_1/Less/y�
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lessu
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_2/zeros_1/packed/1�
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/Const�
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
lstm_2/zeros_1�
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputslstm_2/zeros:output:0lstm_2/zeros_1:output:0%lstm_2_statefulpartitionedcall_args_3%lstm_2_statefulpartitionedcall_args_4%lstm_2_statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*j
_output_shapesX
V:����������:����������:����������:����������: *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference_standard_lstm_270269752 
lstm_2/StatefulPartitionedCall�
;simple_neural_network_layer_block_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_6_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_call_270315212=
;simple_neural_network_layer_block_6/StatefulPartitionedCall�
;simple_neural_network_layer_block_7/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_6/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_7_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_call_269933002=
;simple_neural_network_layer_block_7/StatefulPartitionedCall�
;simple_neural_network_layer_block_8/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_7/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_8_statefulpartitionedcall_args_6*
Tin
	2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_call_270081772=
;simple_neural_network_layer_block_8/StatefulPartitionedCall�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_11/MatMul/ReadVariableOp�
dense_11/MatMulMatMulDsimple_neural_network_layer_block_8/StatefulPartitionedCall:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/MatMul�
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_11/BiasAdds
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_11/Tanh�
IdentityIdentitydense_11/Tanh:y:0 ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^lstm_2/StatefulPartitionedCall<^simple_neural_network_layer_block_6/StatefulPartitionedCall<^simple_neural_network_layer_block_7/StatefulPartitionedCall<^simple_neural_network_layer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2z
;simple_neural_network_layer_block_6/StatefulPartitionedCall;simple_neural_network_layer_block_6/StatefulPartitionedCall2z
;simple_neural_network_layer_block_7/StatefulPartitionedCall;simple_neural_network_layer_block_7/StatefulPartitionedCall2z
;simple_neural_network_layer_block_8/StatefulPartitionedCall;simple_neural_network_layer_block_8/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27018085

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource"
cast_2_readvariableop_resource"
cast_3_readvariableop_resource
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOp^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAnd�
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast/ReadVariableOp�
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_1/ReadVariableOp�
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_2/ReadVariableOp�
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������6<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
input_layer
lstm_layers
hidden_layers
output_layer

signatures
	variables
trainable_variables
regularization_losses
		keras_api
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "LSTMNeuralNetworkModel", "name": "lstm_neural_network_model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "LSTMNeuralNetworkModel"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 5, 54], "config": {"batch_input_shape": [null, 5, 54], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
'

0"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
-
�serving_default"
signature_map
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
#15
$16
%17
&18
'19
(20
21
22"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
 12
!13
"14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
�
)metrics
*non_trainable_variables
	variables

+layers
,layer_regularization_losses
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

-cell
.
state_spec
/	variables
0trainable_variables
1regularization_losses
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTM", "name": "lstm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 512, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 54], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
�
3dense_layer
4
norm_layer
5	variables
6trainable_variables
7regularization_losses
8	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SimpleNeuralNetworkLayerBlock", "name": "simple_neural_network_layer_block_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "simple_neural_network_layer_block_6", "trainable": true, "dtype": "float32", "units": 512, "activation_func": "tanh", "kernel_initializer": "glorot_uniform"}}
�
9dense_layer
:
norm_layer
;	variables
<trainable_variables
=regularization_losses
>	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SimpleNeuralNetworkLayerBlock", "name": "simple_neural_network_layer_block_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "simple_neural_network_layer_block_7", "trainable": true, "dtype": "float32", "units": 512, "activation_func": "tanh", "kernel_initializer": "glorot_uniform"}}
�
?dense_layer
@
norm_layer
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "SimpleNeuralNetworkLayerBlock", "name": "simple_neural_network_layer_block_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "simple_neural_network_layer_block_8", "trainable": true, "dtype": "float32", "units": 512, "activation_func": "tanh", "kernel_initializer": "glorot_uniform"}}
": 	�2dense_11/kernel
:2dense_11/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Emetrics
Fnon_trainable_variables
	variables

Glayers
Hlayer_regularization_losses
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	6�2lstm_2/kernel
+:)
��2lstm_2/recurrent_kernel
:�2lstm_2/bias
": 
��2dense_8/kernel
:�2dense_8/bias
*:(�2batch_normalization_6/gamma
):'�2batch_normalization_6/beta
": 
��2dense_9/kernel
:�2dense_9/bias
*:(�2batch_normalization_7/gamma
):'�2batch_normalization_7/beta
#:!
��2dense_10/kernel
:�2dense_10/bias
*:(�2batch_normalization_8/gamma
):'�2batch_normalization_8/beta
2:0� (2!batch_normalization_6/moving_mean
6:4� (2%batch_normalization_6/moving_variance
2:0� (2!batch_normalization_7/moving_mean
6:4� (2%batch_normalization_7/moving_variance
2:0� (2!batch_normalization_8/moving_mean
6:4� (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
J
#0
$1
%2
&3
'4
(5"
trackable_list_wrapper
J
0

1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�

kernel
recurrent_kernel
bias
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mmetrics
Nnon_trainable_variables
/	variables

Olayers
Player_regularization_losses
0trainable_variables
1regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

kernel
bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
�
Uaxis
	gamma
beta
#moving_mean
$moving_variance
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}}
J
0
1
2
3
#4
$5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Zmetrics
[non_trainable_variables
5	variables

\layers
]layer_regularization_losses
6trainable_variables
7regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

kernel
bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
�
baxis
	gamma
beta
%moving_mean
&moving_variance
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}}
J
0
1
2
3
%4
&5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gmetrics
hnon_trainable_variables
;	variables

ilayers
jlayer_regularization_losses
<trainable_variables
=regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

kernel
 bias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
�
oaxis
	!gamma
"beta
'moving_mean
(moving_variance
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}}
J
0
 1
!2
"3
'4
(5"
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tmetrics
unon_trainable_variables
A	variables

vlayers
wlayer_regularization_losses
Btrainable_variables
Cregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
xmetrics
ynon_trainable_variables
I	variables

zlayers
{layer_regularization_losses
Jtrainable_variables
Kregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
-0"
trackable_list_wrapper
 "
trackable_list_wrapper
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
�
|metrics
}non_trainable_variables
Q	variables

~layers
layer_regularization_losses
Rtrainable_variables
Sregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
#2
$3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�non_trainable_variables
V	variables
�layers
 �layer_regularization_losses
Wtrainable_variables
Xregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�non_trainable_variables
^	variables
�layers
 �layer_regularization_losses
_trainable_variables
`regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
%2
&3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�non_trainable_variables
c	variables
�layers
 �layer_regularization_losses
dtrainable_variables
eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�non_trainable_variables
k	variables
�layers
 �layer_regularization_losses
ltrainable_variables
mregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
!0
"1
'2
(3"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�non_trainable_variables
p	variables
�layers
 �layer_regularization_losses
qtrainable_variables
rregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
#__inference__wrapped_model_74975437�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"
input_1���������6
�2�
>__inference_lstm_neural_network_model_2_layer_call_fn_27019035
>__inference_lstm_neural_network_model_2_layer_call_fn_27013104
>__inference_lstm_neural_network_model_2_layer_call_fn_27013132
>__inference_lstm_neural_network_model_2_layer_call_fn_27019007�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27013164
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27035701
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27031582
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27019078�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dense_11_layer_call_fn_27011741�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_11_layer_call_and_return_conditional_losses_27024864�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5B3
&__inference_signature_wrapper_74975466input_1
�2�
)__inference_lstm_2_layer_call_fn_26997258
)__inference_lstm_2_layer_call_fn_27006589
)__inference_lstm_2_layer_call_fn_27012242
)__inference_lstm_2_layer_call_fn_27011663�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_lstm_2_layer_call_and_return_conditional_losses_27023072
D__inference_lstm_2_layer_call_and_return_conditional_losses_27034162
D__inference_lstm_2_layer_call_and_return_conditional_losses_27024221
D__inference_lstm_2_layer_call_and_return_conditional_losses_27024098�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_simple_neural_network_layer_block_6_layer_call_fn_26991199
F__inference_simple_neural_network_layer_block_6_layer_call_fn_27013044�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_27006778
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_27029219�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_simple_neural_network_layer_block_7_layer_call_fn_27011439
F__inference_simple_neural_network_layer_block_7_layer_call_fn_27019046�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_27019108
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_27004547�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_simple_neural_network_layer_block_8_layer_call_fn_27007387
F__inference_simple_neural_network_layer_block_8_layer_call_fn_27003154�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_26999506
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_27000036�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_6_layer_call_fn_27018094
8__inference_batch_normalization_6_layer_call_fn_26989149�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27002456
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27023613�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_7_layer_call_fn_27001954
8__inference_batch_normalization_7_layer_call_fn_27025395�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_26997000
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27005127�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
8__inference_batch_normalization_8_layer_call_fn_27005829
8__inference_batch_normalization_8_layer_call_fn_27001860�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_26990909
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_27030001�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
#__inference__wrapped_model_74975437�#$%& '("!4�1
*�'
%�"
input_1���������6
� "3�0
.
output_1"�
output_1����������
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27002456d#$4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27023613d#$4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
8__inference_batch_normalization_6_layer_call_fn_26989149W#$4�1
*�'
!�
inputs����������
p
� "������������
8__inference_batch_normalization_6_layer_call_fn_27018094W#$4�1
*�'
!�
inputs����������
p 
� "������������
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_26997000d%&4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27005127d%&4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
8__inference_batch_normalization_7_layer_call_fn_27001954W%&4�1
*�'
!�
inputs����������
p
� "������������
8__inference_batch_normalization_7_layer_call_fn_27025395W%&4�1
*�'
!�
inputs����������
p 
� "������������
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_26990909d'("!4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_27030001d'("!4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
8__inference_batch_normalization_8_layer_call_fn_27001860W'("!4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_8_layer_call_fn_27005829W'("!4�1
*�'
!�
inputs����������
p
� "������������
F__inference_dense_11_layer_call_and_return_conditional_losses_27024864]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� 
+__inference_dense_11_layer_call_fn_27011741P0�-
&�#
!�
inputs����������
� "�����������
D__inference_lstm_2_layer_call_and_return_conditional_losses_27023072~O�L
E�B
4�1
/�,
inputs/0������������������6

 
p 

 
� "&�#
�
0����������
� �
D__inference_lstm_2_layer_call_and_return_conditional_losses_27024098n?�<
5�2
$�!
inputs���������6

 
p 

 
� "&�#
�
0����������
� �
D__inference_lstm_2_layer_call_and_return_conditional_losses_27024221n?�<
5�2
$�!
inputs���������6

 
p

 
� "&�#
�
0����������
� �
D__inference_lstm_2_layer_call_and_return_conditional_losses_27034162~O�L
E�B
4�1
/�,
inputs/0������������������6

 
p

 
� "&�#
�
0����������
� �
)__inference_lstm_2_layer_call_fn_26997258a?�<
5�2
$�!
inputs���������6

 
p 

 
� "������������
)__inference_lstm_2_layer_call_fn_27006589qO�L
E�B
4�1
/�,
inputs/0������������������6

 
p

 
� "������������
)__inference_lstm_2_layer_call_fn_27011663qO�L
E�B
4�1
/�,
inputs/0������������������6

 
p 

 
� "������������
)__inference_lstm_2_layer_call_fn_27012242a?�<
5�2
$�!
inputs���������6

 
p

 
� "������������
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27013164z#$%& '("!8�5
.�+
%�"
input_1���������6
p 
� "%�"
�
0���������
� �
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27019078z#$%& '("!8�5
.�+
%�"
input_1���������6
p
� "%�"
�
0���������
� �
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27031582y#$%& '("!7�4
-�*
$�!
inputs���������6
p
� "%�"
�
0���������
� �
Y__inference_lstm_neural_network_model_2_layer_call_and_return_conditional_losses_27035701y#$%& '("!7�4
-�*
$�!
inputs���������6
p 
� "%�"
�
0���������
� �
>__inference_lstm_neural_network_model_2_layer_call_fn_27013104l#$%& '("!7�4
-�*
$�!
inputs���������6
p 
� "�����������
>__inference_lstm_neural_network_model_2_layer_call_fn_27013132m#$%& '("!8�5
.�+
%�"
input_1���������6
p 
� "�����������
>__inference_lstm_neural_network_model_2_layer_call_fn_27019007l#$%& '("!7�4
-�*
$�!
inputs���������6
p
� "�����������
>__inference_lstm_neural_network_model_2_layer_call_fn_27019035m#$%& '("!8�5
.�+
%�"
input_1���������6
p
� "�����������
&__inference_signature_wrapper_74975466�#$%& '("!?�<
� 
5�2
0
input_1%�"
input_1���������6"3�0
.
output_1"�
output_1����������
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_27006778f#$4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
a__inference_simple_neural_network_layer_block_6_layer_call_and_return_conditional_losses_27029219f#$4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_simple_neural_network_layer_block_6_layer_call_fn_26991199Y#$4�1
*�'
!�
inputs����������
p
� "������������
F__inference_simple_neural_network_layer_block_6_layer_call_fn_27013044Y#$4�1
*�'
!�
inputs����������
p 
� "������������
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_27004547f%&4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
a__inference_simple_neural_network_layer_block_7_layer_call_and_return_conditional_losses_27019108f%&4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_simple_neural_network_layer_block_7_layer_call_fn_27011439Y%&4�1
*�'
!�
inputs����������
p 
� "������������
F__inference_simple_neural_network_layer_block_7_layer_call_fn_27019046Y%&4�1
*�'
!�
inputs����������
p
� "������������
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_26999506f '("!4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
a__inference_simple_neural_network_layer_block_8_layer_call_and_return_conditional_losses_27000036f '("!4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_simple_neural_network_layer_block_8_layer_call_fn_27003154Y '("!4�1
*�'
!�
inputs����������
p 
� "������������
F__inference_simple_neural_network_layer_block_8_layer_call_fn_27007387Y '("!4�1
*�'
!�
inputs����������
p
� "�����������