˽$
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
shapeshape�"serve*2.1.02unknown8�"
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	�*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
w
lstm_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	6�*
shared_namelstm_3/kernel
p
!lstm_3/kernel/Read/ReadVariableOpReadVariableOplstm_3/kernel*
_output_shapes
:	6�*
dtype0
�
lstm_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*(
shared_namelstm_3/recurrent_kernel
�
+lstm_3/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_3/recurrent_kernel* 
_output_shapes
:
��*
dtype0
o
lstm_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namelstm_3/bias
h
lstm_3/bias/Read/ReadVariableOpReadVariableOplstm_3/bias*
_output_shapes	
:�*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
��*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_9/gamma
�
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_9/beta
�
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes	
:�*
dtype0
|
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_13/kernel
u
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel* 
_output_shapes
:
��*
dtype0
s
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_13/bias
l
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_10/gamma
�
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_10/beta
�
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes	
:�*
dtype0
|
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_14/kernel
u
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel* 
_output_shapes
:
��*
dtype0
s
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_14/bias
l
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*-
shared_namebatch_normalization_11/gamma
�
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_11/beta
�
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_9/moving_mean
�
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_9/moving_variance
�
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_10/moving_mean
�
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_10/moving_variance
�
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes	
:�*
dtype0
�
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"batch_normalization_11/moving_mean
�
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:�*
dtype0
�
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&batch_normalization_11/moving_variance
�
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:�*
dtype0

NoOpNoOp
�8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�7
value�7B�7 B�7
�
input_layer
lstm_layers
hidden_layers
output_layer

signatures
trainable_variables
	variables
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
trainable_variables
	variables
regularization_losses
	keras_api
 
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
 
�
)non_trainable_variables

*layers
trainable_variables
	variables
+metrics
,layer_regularization_losses
regularization_losses
l
-cell
.
state_spec
/trainable_variables
0	variables
1regularization_losses
2	keras_api
s
3dense_layer
4
norm_layer
5trainable_variables
6	variables
7regularization_losses
8	keras_api
s
9dense_layer
:
norm_layer
;trainable_variables
<	variables
=regularization_losses
>	keras_api
s
?dense_layer
@
norm_layer
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
SQ
VARIABLE_VALUEdense_15/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEdense_15/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Enon_trainable_variables

Flayers
trainable_variables
	variables
Gmetrics
Hlayer_regularization_losses
regularization_losses
SQ
VARIABLE_VALUElstm_3/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_3/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm_3/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_12/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_12/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_9/gamma0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_9/beta0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_13/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_13/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_10/gamma0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_10/beta1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_14/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_14/bias1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_11/gamma1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_11/beta1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_9/moving_mean'variables/15/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_9/moving_variance'variables/16/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_10/moving_mean'variables/17/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_10/moving_variance'variables/18/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_11/moving_mean'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_11/moving_variance'variables/20/.ATTRIBUTES/VARIABLE_VALUE
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
 
~

kernel
recurrent_kernel
bias
Itrainable_variables
J	variables
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
Mnon_trainable_variables

Nlayers
/trainable_variables
0	variables
Ometrics
Player_regularization_losses
1regularization_losses
h

kernel
bias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
�
Uaxis
	gamma
beta
#moving_mean
$moving_variance
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api

0
1
2
3
*
0
1
2
3
#4
$5
 
�
Znon_trainable_variables

[layers
5trainable_variables
6	variables
\metrics
]layer_regularization_losses
7regularization_losses
h

kernel
bias
^trainable_variables
_	variables
`regularization_losses
a	keras_api
�
baxis
	gamma
beta
%moving_mean
&moving_variance
ctrainable_variables
d	variables
eregularization_losses
f	keras_api

0
1
2
3
*
0
1
2
3
%4
&5
 
�
gnon_trainable_variables

hlayers
;trainable_variables
<	variables
imetrics
jlayer_regularization_losses
=regularization_losses
h

kernel
 bias
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
�
oaxis
	!gamma
"beta
'moving_mean
(moving_variance
ptrainable_variables
q	variables
rregularization_losses
s	keras_api

0
 1
!2
"3
*
0
 1
!2
"3
'4
(5
 
�
tnon_trainable_variables

ulayers
Atrainable_variables
B	variables
vmetrics
wlayer_regularization_losses
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
xnon_trainable_variables

ylayers
Itrainable_variables
J	variables
zmetrics
{layer_regularization_losses
Kregularization_losses
 

-0
 
 

0
1

0
1
 
�
|non_trainable_variables

}layers
Qtrainable_variables
R	variables
~metrics
layer_regularization_losses
Sregularization_losses
 

0
1

0
1
#2
$3
 
�
�non_trainable_variables
�layers
Vtrainable_variables
W	variables
�metrics
 �layer_regularization_losses
Xregularization_losses

#0
$1

30
41
 
 

0
1

0
1
 
�
�non_trainable_variables
�layers
^trainable_variables
_	variables
�metrics
 �layer_regularization_losses
`regularization_losses
 

0
1

0
1
%2
&3
 
�
�non_trainable_variables
�layers
ctrainable_variables
d	variables
�metrics
 �layer_regularization_losses
eregularization_losses

%0
&1

90
:1
 
 

0
 1

0
 1
 
�
�non_trainable_variables
�layers
ktrainable_variables
l	variables
�metrics
 �layer_regularization_losses
mregularization_losses
 

!0
"1

!0
"1
'2
(3
 
�
�non_trainable_variables
�layers
ptrainable_variables
q	variables
�metrics
 �layer_regularization_losses
rregularization_losses
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
 
�
serving_default_input_1Placeholder*+
_output_shapes
:���������6*
dtype0* 
shape:���������6
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1lstm_3/kernellstm_3/recurrent_kernellstm_3/biasdense_12/kerneldense_12/bias!batch_normalization_9/moving_mean%batch_normalization_9/moving_variancebatch_normalization_9/betabatch_normalization_9/gammadense_13/kerneldense_13/bias"batch_normalization_10/moving_mean&batch_normalization_10/moving_variancebatch_normalization_10/betabatch_normalization_10/gammadense_14/kerneldense_14/bias"batch_normalization_11/moving_mean&batch_normalization_11/moving_variancebatch_normalization_11/betabatch_normalization_11/gammadense_15/kerneldense_15/bias*#
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
&__inference_signature_wrapper_46361885
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp!lstm_3/kernel/Read/ReadVariableOp+lstm_3/recurrent_kernel/Read/ReadVariableOplstm_3/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOpConst*$
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
!__inference__traced_save_46361978
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/biaslstm_3/kernellstm_3/recurrent_kernellstm_3/biasdense_12/kerneldense_12/biasbatch_normalization_9/gammabatch_normalization_9/betadense_13/kerneldense_13/biasbatch_normalization_10/gammabatch_normalization_10/betadense_14/kerneldense_14/biasbatch_normalization_11/gammabatch_normalization_11/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance"batch_normalization_10/moving_mean&batch_normalization_10/moving_variance"batch_normalization_11/moving_mean&batch_normalization_11/moving_variance*#
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
$__inference__traced_restore_46362059Ɠ!
�
�
(__inference_while_cond_12831132_23175027
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12831132___redundant_placeholder00
,while_cond_12831132___redundant_placeholder10
,while_cond_12831132___redundant_placeholder20
,while_cond_12831132___redundant_placeholder3
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
(__inference_while_body_12837151_23174665
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
�	
B__forward_lstm_3_layer_call_and_return_conditional_losses_23772775

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
 __forward_standard_lstm_237727172
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
backward_function_nameZX__inference___backward_lstm_3_layer_call_and_return_conditional_losses_23772259_2377277622
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
+__inference_dense_15_layer_call_fn_23172501

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
F__inference_dense_15_layer_call_and_return_conditional_losses_231724942
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
�.
�
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_23174297

inputs
assignmovingavg_12838744
assignmovingavg_1_12838750 
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
loc:@AssignMovingAvg/12838744*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_12838744*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12838744*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12838744*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_12838744AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/12838744*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/12838750*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_12838750*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12838750*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12838750*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_12838750AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/12838750*
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
�
�
D__forward_dense_15_layer_call_and_return_conditional_losses_23771987
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
backward_function_name\Z__inference___backward_dense_15_layer_call_and_return_conditional_losses_23771973_2377198820
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_11_layer_call_fn_23174079

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
GPU2*0J 8*]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_231740702
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
�G
�
"__inference_standard_lstm_23174740

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
(__inference_while_body_12837151_23174665*4
cond,R*
(__inference_while_cond_12837150_23171773*M
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
�
__inference_call_23172676

inputs+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource2
.batch_normalization_9_assignmovingavg_128329644
0batch_normalization_9_assignmovingavg_1_128329706
2batch_normalization_9_cast_readvariableop_resource8
4batch_normalization_9_cast_1_readvariableop_resource
identity��9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_9/AssignMovingAvg/ReadVariableOp�;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices�
"batch_normalization_9/moments/meanMeandense_12/BiasAdd:output:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_9/moments/mean�
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_9/moments/StopGradient�
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_12/BiasAdd:output:03batch_normalization_9/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_9/moments/SquaredDifference�
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices�
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_9/moments/variance�
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_9/moments/Squeeze�
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1�
+batch_normalization_9/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12832964*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_9/AssignMovingAvg/decay�
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_9_assignmovingavg_12832964*
_output_shapes	
:�*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp�
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12832964*
_output_shapes	
:�2+
)batch_normalization_9/AssignMovingAvg/sub�
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12832964*
_output_shapes	
:�2+
)batch_normalization_9/AssignMovingAvg/mul�
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_9_assignmovingavg_12832964-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12832964*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_9/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12832970*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_9/AssignMovingAvg_1/decay�
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_9_assignmovingavg_1_12832970*
_output_shapes	
:�*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12832970*
_output_shapes	
:�2-
+batch_normalization_9/AssignMovingAvg_1/sub�
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12832970*
_output_shapes	
:�2-
+batch_normalization_9/AssignMovingAvg_1/mul�
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_9_assignmovingavg_1_12832970/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12832970*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_12/BiasAdd:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub1batch_normalization_9/Cast/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/add_1r
TanhTanh)batch_normalization_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_9/AssignMovingAvg/ReadVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�a
�
$__inference__traced_restore_46362059
file_prefix$
 assignvariableop_dense_15_kernel$
 assignvariableop_1_dense_15_bias$
 assignvariableop_2_lstm_3_kernel.
*assignvariableop_3_lstm_3_recurrent_kernel"
assignvariableop_4_lstm_3_bias&
"assignvariableop_5_dense_12_kernel$
 assignvariableop_6_dense_12_bias2
.assignvariableop_7_batch_normalization_9_gamma1
-assignvariableop_8_batch_normalization_9_beta&
"assignvariableop_9_dense_13_kernel%
!assignvariableop_10_dense_13_bias4
0assignvariableop_11_batch_normalization_10_gamma3
/assignvariableop_12_batch_normalization_10_beta'
#assignvariableop_13_dense_14_kernel%
!assignvariableop_14_dense_14_bias4
0assignvariableop_15_batch_normalization_11_gamma3
/assignvariableop_16_batch_normalization_11_beta9
5assignvariableop_17_batch_normalization_9_moving_mean=
9assignvariableop_18_batch_normalization_9_moving_variance:
6assignvariableop_19_batch_normalization_10_moving_mean>
:assignvariableop_20_batch_normalization_10_moving_variance:
6assignvariableop_21_batch_normalization_11_moving_mean>
:assignvariableop_22_batch_normalization_11_moving_variance
identity_24��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUE2
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
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_lstm_3_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp*assignvariableop_3_lstm_3_recurrent_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_lstm_3_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_12_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense_12_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_9_gammaIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batch_normalization_9_betaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_13_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_13_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_10_gammaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_10_betaIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_14_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_14_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batch_normalization_11_gammaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_11_betaIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp5assignvariableop_17_batch_normalization_9_moving_meanIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp9assignvariableop_18_batch_normalization_9_moving_varianceIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp6assignvariableop_19_batch_normalization_10_moving_meanIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_normalization_10_moving_varianceIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp6assignvariableop_21_batch_normalization_11_moving_meanIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp:assignvariableop_22_batch_normalization_11_moving_varianceIdentity_22:output:0*
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
�
�
8__inference_batch_normalization_9_layer_call_fn_23174467

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
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_231744582
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
�
�
>__inference_lstm_neural_network_model_3_layer_call_fn_23173751
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
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_231737232
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
�G
�
"__inference_standard_lstm_23175102

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
(__inference_while_body_12831133_23174515*4
cond,R*
(__inference_while_cond_12831132_23175027*M
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
�5
�
!__inference__traced_save_46361978
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop,
(savev2_lstm_3_kernel_read_readvariableop6
2savev2_lstm_3_recurrent_kernel_read_readvariableop*
&savev2_lstm_3_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_1a869293d8d347ea81603b4cb84fab26/part2
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
ShardedFilename�	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop(savev2_lstm_3_kernel_read_readvariableop2savev2_lstm_3_recurrent_kernel_read_readvariableop&savev2_lstm_3_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop"/device:CPU:0*
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
�	
�
F__inference_simple_neural_network_layer_block_9_layer_call_fn_23172542

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
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_231725312
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
�.
�
(__inference_while_body_12833184_23172483
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
�	
�
G__inference_simple_neural_network_layer_block_10_layer_call_fn_23172435

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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_231724242
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
�P
�
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23172945

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource3
/batch_normalization_11_assignmovingavg_128322065
1batch_normalization_11_assignmovingavg_1_128322127
3batch_normalization_11_cast_readvariableop_resource9
5batch_normalization_11_cast_1_readvariableop_resource
identity��:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_11/AssignMovingAvg/ReadVariableOp�<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_11/Cast/ReadVariableOp�,batch_normalization_11/Cast_1/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/BiasAdd�
#batch_normalization_11/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/x�
#batch_normalization_11/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/y�
!batch_normalization_11/LogicalAnd
LogicalAnd,batch_normalization_11/LogicalAnd/x:output:0,batch_normalization_11/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_11/LogicalAnd�
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_11/moments/mean/reduction_indices�
#batch_normalization_11/moments/meanMeandense_14/BiasAdd:output:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2%
#batch_normalization_11/moments/mean�
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:	�2-
+batch_normalization_11/moments/StopGradient�
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_14/BiasAdd:output:04batch_normalization_11/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������22
0batch_normalization_11/moments/SquaredDifference�
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_11/moments/variance/reduction_indices�
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2)
'batch_normalization_11/moments/variance�
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2(
&batch_normalization_11/moments/Squeeze�
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2*
(batch_normalization_11/moments/Squeeze_1�
,batch_normalization_11/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12832206*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_11/AssignMovingAvg/decay�
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_11_assignmovingavg_12832206*
_output_shapes	
:�*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOp�
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12832206*
_output_shapes	
:�2,
*batch_normalization_11/AssignMovingAvg/sub�
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12832206*
_output_shapes	
:�2,
*batch_normalization_11/AssignMovingAvg/mul�
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_11_assignmovingavg_12832206.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12832206*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_11/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12832212*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_11/AssignMovingAvg_1/decay�
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_11_assignmovingavg_1_12832212*
_output_shapes	
:�*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12832212*
_output_shapes	
:�2.
,batch_normalization_11/AssignMovingAvg_1/sub�
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12832212*
_output_shapes	
:�2.
,batch_normalization_11/AssignMovingAvg_1/mul�
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_11_assignmovingavg_1_128322120batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12832212*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp�
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_11/Cast/ReadVariableOp�
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp�
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_11/batchnorm/add/y�
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/add�
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/Rsqrt�
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/mul�
&batch_normalization_11/batchnorm/mul_1Muldense_14/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/mul_1�
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/mul_2�
$batch_normalization_11/batchnorm/subSub2batch_normalization_11/Cast/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/sub�
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/add_1s
TanhTanh*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_11/AssignMovingAvg/ReadVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�P
�
__inference_call_23172133

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource3
/batch_normalization_11_assignmovingavg_128330705
1batch_normalization_11_assignmovingavg_1_128330767
3batch_normalization_11_cast_readvariableop_resource9
5batch_normalization_11_cast_1_readvariableop_resource
identity��:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_11/AssignMovingAvg/ReadVariableOp�<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_11/Cast/ReadVariableOp�,batch_normalization_11/Cast_1/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/BiasAdd�
#batch_normalization_11/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/x�
#batch_normalization_11/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/y�
!batch_normalization_11/LogicalAnd
LogicalAnd,batch_normalization_11/LogicalAnd/x:output:0,batch_normalization_11/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_11/LogicalAnd�
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_11/moments/mean/reduction_indices�
#batch_normalization_11/moments/meanMeandense_14/BiasAdd:output:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2%
#batch_normalization_11/moments/mean�
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:	�2-
+batch_normalization_11/moments/StopGradient�
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_14/BiasAdd:output:04batch_normalization_11/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������22
0batch_normalization_11/moments/SquaredDifference�
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_11/moments/variance/reduction_indices�
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2)
'batch_normalization_11/moments/variance�
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2(
&batch_normalization_11/moments/Squeeze�
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2*
(batch_normalization_11/moments/Squeeze_1�
,batch_normalization_11/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12833070*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_11/AssignMovingAvg/decay�
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_11_assignmovingavg_12833070*
_output_shapes	
:�*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOp�
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12833070*
_output_shapes	
:�2,
*batch_normalization_11/AssignMovingAvg/sub�
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12833070*
_output_shapes	
:�2,
*batch_normalization_11/AssignMovingAvg/mul�
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_11_assignmovingavg_12833070.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12833070*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_11/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12833076*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_11/AssignMovingAvg_1/decay�
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_11_assignmovingavg_1_12833076*
_output_shapes	
:�*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12833076*
_output_shapes	
:�2.
,batch_normalization_11/AssignMovingAvg_1/sub�
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12833076*
_output_shapes	
:�2.
,batch_normalization_11/AssignMovingAvg_1/mul�
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_11_assignmovingavg_1_128330760batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12833076*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp�
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_11/Cast/ReadVariableOp�
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp�
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_11/batchnorm/add/y�
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/add�
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/Rsqrt�
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/mul�
&batch_normalization_11/batchnorm/mul_1Muldense_14/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/mul_1�
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/mul_2�
$batch_normalization_11/batchnorm/subSub2batch_normalization_11/Cast/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/sub�
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/add_1s
TanhTanh*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_11/AssignMovingAvg/ReadVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23173943

inputs
assignmovingavg_12830850
assignmovingavg_1_12830856 
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
loc:@AssignMovingAvg/12830850*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_12830850*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12830850*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12830850*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_12830850AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/12830850*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/12830856*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_12830856*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12830856*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12830856*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_12830856AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/12830856*
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
�4
�
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23175263
input_1)
%lstm_3_statefulpartitionedcall_args_1)
%lstm_3_statefulpartitionedcall_args_2)
%lstm_3_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6+
'dense_15_statefulpartitionedcall_args_1+
'dense_15_statefulpartitionedcall_args_2
identity�� dense_15/StatefulPartitionedCall�lstm_3/StatefulPartitionedCall�<simple_neural_network_layer_block_10/StatefulPartitionedCall�<simple_neural_network_layer_block_11/StatefulPartitionedCall�;simple_neural_network_layer_block_9/StatefulPartitionedCall�
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinput_1%lstm_3_statefulpartitionedcall_args_1%lstm_3_statefulpartitionedcall_args_2%lstm_3_statefulpartitionedcall_args_3*
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_231751352 
lstm_3/StatefulPartitionedCall�
;simple_neural_network_layer_block_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6*
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
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_231736062=
;simple_neural_network_layer_block_9/StatefulPartitionedCall�
<simple_neural_network_layer_block_10/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_9/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6*
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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_231724242>
<simple_neural_network_layer_block_10/StatefulPartitionedCall�
<simple_neural_network_layer_block_11/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_10/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6*
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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_231729452>
<simple_neural_network_layer_block_11/StatefulPartitionedCall�
 dense_15/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_11/StatefulPartitionedCall:output:0'dense_15_statefulpartitionedcall_args_1'dense_15_statefulpartitionedcall_args_2*
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
F__inference_dense_15_layer_call_and_return_conditional_losses_231724942"
 dense_15/StatefulPartitionedCall�
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall=^simple_neural_network_layer_block_10/StatefulPartitionedCall=^simple_neural_network_layer_block_11/StatefulPartitionedCall<^simple_neural_network_layer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2|
<simple_neural_network_layer_block_10/StatefulPartitionedCall<simple_neural_network_layer_block_10/StatefulPartitionedCall2|
<simple_neural_network_layer_block_11/StatefulPartitionedCall<simple_neural_network_layer_block_11/StatefulPartitionedCall2z
;simple_neural_network_layer_block_9/StatefulPartitionedCall;simple_neural_network_layer_block_9/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
)__inference_lstm_3_layer_call_fn_23171758

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
D__inference_lstm_3_layer_call_and_return_conditional_losses_231717502
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
�)
�
__inference_call_23174545

inputs+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource6
2batch_normalization_9_cast_readvariableop_resource8
4batch_normalization_9_cast_1_readvariableop_resource8
4batch_normalization_9_cast_2_readvariableop_resource8
4batch_normalization_9_cast_3_readvariableop_resource
identity��)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�+batch_normalization_9/Cast_2/ReadVariableOp�+batch_normalization_9/Cast_3/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOp�
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_12/BiasAdd:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/add_1r
TanhTanh)batch_normalization_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2Z
+batch_normalization_9/Cast_2/ReadVariableOp+batch_normalization_9/Cast_2/ReadVariableOp2Z
+batch_normalization_9/Cast_3/ReadVariableOp+batch_normalization_9/Cast_3/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�G
�
"__inference_standard_lstm_23174926

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
(__inference_while_body_12836245_23174851*4
cond,R*
(__inference_while_cond_12836244_23172770*M
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
(__inference_while_body_12830247_23171832
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
�
�
(__inference_while_cond_12832572_23174094
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12832572___redundant_placeholder00
,while_cond_12832572___redundant_placeholder10
,while_cond_12832572___redundant_placeholder20
,while_cond_12832572___redundant_placeholder3
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
�
�
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_23175012

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
�4
�
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23175175

inputs)
%lstm_3_statefulpartitionedcall_args_1)
%lstm_3_statefulpartitionedcall_args_2)
%lstm_3_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6+
'dense_15_statefulpartitionedcall_args_1+
'dense_15_statefulpartitionedcall_args_2
identity�� dense_15/StatefulPartitionedCall�lstm_3/StatefulPartitionedCall�<simple_neural_network_layer_block_10/StatefulPartitionedCall�<simple_neural_network_layer_block_11/StatefulPartitionedCall�;simple_neural_network_layer_block_9/StatefulPartitionedCall�
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinputs%lstm_3_statefulpartitionedcall_args_1%lstm_3_statefulpartitionedcall_args_2%lstm_3_statefulpartitionedcall_args_3*
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_231751352 
lstm_3/StatefulPartitionedCall�
;simple_neural_network_layer_block_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6*
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
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_231736062=
;simple_neural_network_layer_block_9/StatefulPartitionedCall�
<simple_neural_network_layer_block_10/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_9/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6*
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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_231724242>
<simple_neural_network_layer_block_10/StatefulPartitionedCall�
<simple_neural_network_layer_block_11/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_10/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6*
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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_231729452>
<simple_neural_network_layer_block_11/StatefulPartitionedCall�
 dense_15/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_11/StatefulPartitionedCall:output:0'dense_15_statefulpartitionedcall_args_1'dense_15_statefulpartitionedcall_args_2*
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
F__inference_dense_15_layer_call_and_return_conditional_losses_231724942"
 dense_15/StatefulPartitionedCall�
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall=^simple_neural_network_layer_block_10/StatefulPartitionedCall=^simple_neural_network_layer_block_11/StatefulPartitionedCall<^simple_neural_network_layer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2|
<simple_neural_network_layer_block_10/StatefulPartitionedCall<simple_neural_network_layer_block_10/StatefulPartitionedCall2|
<simple_neural_network_layer_block_11/StatefulPartitionedCall<simple_neural_network_layer_block_11/StatefulPartitionedCall2z
;simple_neural_network_layer_block_9/StatefulPartitionedCall;simple_neural_network_layer_block_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�P
�
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23172380

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource3
/batch_normalization_11_assignmovingavg_128383545
1batch_normalization_11_assignmovingavg_1_128383607
3batch_normalization_11_cast_readvariableop_resource9
5batch_normalization_11_cast_1_readvariableop_resource
identity��:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_11/AssignMovingAvg/ReadVariableOp�<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_11/Cast/ReadVariableOp�,batch_normalization_11/Cast_1/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/BiasAdd�
#batch_normalization_11/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/x�
#batch_normalization_11/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/y�
!batch_normalization_11/LogicalAnd
LogicalAnd,batch_normalization_11/LogicalAnd/x:output:0,batch_normalization_11/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_11/LogicalAnd�
5batch_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_11/moments/mean/reduction_indices�
#batch_normalization_11/moments/meanMeandense_14/BiasAdd:output:0>batch_normalization_11/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2%
#batch_normalization_11/moments/mean�
+batch_normalization_11/moments/StopGradientStopGradient,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes
:	�2-
+batch_normalization_11/moments/StopGradient�
0batch_normalization_11/moments/SquaredDifferenceSquaredDifferencedense_14/BiasAdd:output:04batch_normalization_11/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������22
0batch_normalization_11/moments/SquaredDifference�
9batch_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_11/moments/variance/reduction_indices�
'batch_normalization_11/moments/varianceMean4batch_normalization_11/moments/SquaredDifference:z:0Bbatch_normalization_11/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2)
'batch_normalization_11/moments/variance�
&batch_normalization_11/moments/SqueezeSqueeze,batch_normalization_11/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2(
&batch_normalization_11/moments/Squeeze�
(batch_normalization_11/moments/Squeeze_1Squeeze0batch_normalization_11/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2*
(batch_normalization_11/moments/Squeeze_1�
,batch_normalization_11/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12838354*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_11/AssignMovingAvg/decay�
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_11_assignmovingavg_12838354*
_output_shapes	
:�*
dtype027
5batch_normalization_11/AssignMovingAvg/ReadVariableOp�
*batch_normalization_11/AssignMovingAvg/subSub=batch_normalization_11/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_11/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12838354*
_output_shapes	
:�2,
*batch_normalization_11/AssignMovingAvg/sub�
*batch_normalization_11/AssignMovingAvg/mulMul.batch_normalization_11/AssignMovingAvg/sub:z:05batch_normalization_11/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12838354*
_output_shapes	
:�2,
*batch_normalization_11/AssignMovingAvg/mul�
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_11_assignmovingavg_12838354.batch_normalization_11/AssignMovingAvg/mul:z:06^batch_normalization_11/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_11/AssignMovingAvg/12838354*
_output_shapes
 *
dtype02<
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_11/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12838360*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_11/AssignMovingAvg_1/decay�
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_11_assignmovingavg_1_12838360*
_output_shapes	
:�*
dtype029
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_11/AssignMovingAvg_1/subSub?batch_normalization_11/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_11/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12838360*
_output_shapes	
:�2.
,batch_normalization_11/AssignMovingAvg_1/sub�
,batch_normalization_11/AssignMovingAvg_1/mulMul0batch_normalization_11/AssignMovingAvg_1/sub:z:07batch_normalization_11/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12838360*
_output_shapes	
:�2.
,batch_normalization_11/AssignMovingAvg_1/mul�
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_11_assignmovingavg_1_128383600batch_normalization_11/AssignMovingAvg_1/mul:z:08^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_11/AssignMovingAvg_1/12838360*
_output_shapes
 *
dtype02>
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp�
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_11/Cast/ReadVariableOp�
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp�
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_11/batchnorm/add/y�
$batch_normalization_11/batchnorm/addAddV21batch_normalization_11/moments/Squeeze_1:output:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/add�
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/Rsqrt�
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/mul�
&batch_normalization_11/batchnorm/mul_1Muldense_14/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/mul_1�
&batch_normalization_11/batchnorm/mul_2Mul/batch_normalization_11/moments/Squeeze:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/mul_2�
$batch_normalization_11/batchnorm/subSub2batch_normalization_11/Cast/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/sub�
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/add_1s
TanhTanh*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_11/AssignMovingAvg/ReadVariableOp=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_11/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2x
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp:batch_normalization_11/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_11/AssignMovingAvg/ReadVariableOp5batch_normalization_11/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
D__inference_lstm_3_layer_call_and_return_conditional_losses_23174959
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
"__inference_standard_lstm_231749262
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
�3
�
`__forward_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23772056
inputs_0+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource7
3batch_normalization_11_cast_readvariableop_resource9
5batch_normalization_11_cast_1_readvariableop_resource9
5batch_normalization_11_cast_2_readvariableop_resource9
5batch_normalization_11_cast_3_readvariableop_resource
identity
tanh*
&batch_normalization_11_batchnorm_mul_1(
$batch_normalization_11_batchnorm_sub
dense_14_biasadd(
$batch_normalization_11_batchnorm_mul.
*batch_normalization_11_cast_readvariableop"
dense_14_matmul_readvariableop

inputs0
,batch_normalization_11_cast_3_readvariableop*
&batch_normalization_11_batchnorm_rsqrt��*batch_normalization_11/Cast/ReadVariableOp�,batch_normalization_11/Cast_1/ReadVariableOp�,batch_normalization_11/Cast_2/ReadVariableOp�,batch_normalization_11/Cast_3/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulinputs_0&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/BiasAdd�
#batch_normalization_11/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_11/LogicalAnd/x�
#batch_normalization_11/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/y�
!batch_normalization_11/LogicalAnd
LogicalAnd,batch_normalization_11/LogicalAnd/x:output:0,batch_normalization_11/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_11/LogicalAnd�
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_11/Cast/ReadVariableOp�
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp�
,batch_normalization_11/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_11_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_2/ReadVariableOp�
,batch_normalization_11/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_11_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_3/ReadVariableOp�
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_11/batchnorm/add/y�
$batch_normalization_11/batchnorm/addAddV24batch_normalization_11/Cast_1/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/add�
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/Rsqrt�
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/mul�
&batch_normalization_11/batchnorm/mul_1Muldense_14/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/mul_1�
&batch_normalization_11/batchnorm/mul_2Mul2batch_normalization_11/Cast/ReadVariableOp:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/mul_2�
$batch_normalization_11/batchnorm/subSub4batch_normalization_11/Cast_2/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/sub�
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/add_1s
TanhTanh*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp-^batch_normalization_11/Cast_2/ReadVariableOp-^batch_normalization_11/Cast_3/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"P
$batch_normalization_11_batchnorm_mul(batch_normalization_11/batchnorm/mul:z:0"T
&batch_normalization_11_batchnorm_mul_1*batch_normalization_11/batchnorm/mul_1:z:0"T
&batch_normalization_11_batchnorm_rsqrt*batch_normalization_11/batchnorm/Rsqrt:y:0"P
$batch_normalization_11_batchnorm_sub(batch_normalization_11/batchnorm/sub:z:0"d
,batch_normalization_11_cast_3_readvariableop4batch_normalization_11/Cast_3/ReadVariableOp:value:0"`
*batch_normalization_11_cast_readvariableop2batch_normalization_11/Cast/ReadVariableOp:value:0"-
dense_14_biasadddense_14/BiasAdd:output:0"H
dense_14_matmul_readvariableop&dense_14/MatMul/ReadVariableOp:value:0"
identityIdentity:output:0"
inputsinputs_0"
tanhTanh:y:0*?
_input_shapes.
,:����������::::::*�
backward_function_namexv__inference___backward_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23771998_237720572X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2\
,batch_normalization_11/Cast_2/ReadVariableOp,batch_normalization_11/Cast_2/ReadVariableOp2\
,batch_normalization_11/Cast_3/ReadVariableOp,batch_normalization_11/Cast_3/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
9__inference_batch_normalization_11_layer_call_fn_23174329

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
GPU2*0J 8*]
fXRV
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_231743202
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
�
�
D__inference_lstm_3_layer_call_and_return_conditional_losses_23175135

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
"__inference_standard_lstm_231751022
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
"__inference_standard_lstm_23173079

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
(__inference_while_body_12836706_23173004*4
cond,R*
(__inference_while_cond_12836705_23172251*M
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_23171750

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
"__inference_standard_lstm_231717172
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
"__inference_standard_lstm_23173276

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
(__inference_while_body_12829794_23173201*4
cond,R*
(__inference_while_cond_12829793_23172089*M
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
�G
�
"__inference_standard_lstm_23172860

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
(__inference_while_body_12830247_23171832*4
cond,R*
(__inference_while_cond_12830246_23172785*M
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
�	
�
F__inference_simple_neural_network_layer_block_9_layer_call_fn_23173617

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
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_231736062
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
�G
�
"__inference_standard_lstm_23171970

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
(__inference_while_body_12835800_23171880*4
cond,R*
(__inference_while_cond_12835799_23171895*M
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
�	
�
G__inference_simple_neural_network_layer_block_10_layer_call_fn_23173153

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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_231731422
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
�.
�
(__inference_while_body_12831578_23171642
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
(__inference_while_body_12831133_23174515
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
�P
�
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23172424

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource3
/batch_normalization_10_assignmovingavg_128320995
1batch_normalization_10_assignmovingavg_1_128321057
3batch_normalization_10_cast_readvariableop_resource9
5batch_normalization_10_cast_1_readvariableop_resource
identity��:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_10/AssignMovingAvg/ReadVariableOp�<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indices�
#batch_normalization_10/moments/meanMeandense_13/BiasAdd:output:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2%
#batch_normalization_10/moments/mean�
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:	�2-
+batch_normalization_10/moments/StopGradient�
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_13/BiasAdd:output:04batch_normalization_10/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������22
0batch_normalization_10/moments/SquaredDifference�
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices�
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2)
'batch_normalization_10/moments/variance�
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze�
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1�
,batch_normalization_10/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12832099*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_10/AssignMovingAvg/decay�
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_10_assignmovingavg_12832099*
_output_shapes	
:�*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp�
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12832099*
_output_shapes	
:�2,
*batch_normalization_10/AssignMovingAvg/sub�
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12832099*
_output_shapes	
:�2,
*batch_normalization_10/AssignMovingAvg/mul�
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_10_assignmovingavg_12832099.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12832099*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_10/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12832105*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_10/AssignMovingAvg_1/decay�
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_10_assignmovingavg_1_12832105*
_output_shapes	
:�*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12832105*
_output_shapes	
:�2.
,batch_normalization_10/AssignMovingAvg_1/sub�
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12832105*
_output_shapes	
:�2.
,batch_normalization_10/AssignMovingAvg_1/mul�
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_10_assignmovingavg_1_128321050batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12832105*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_13/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub2batch_normalization_10/Cast/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/add_1s
TanhTanh*batch_normalization_10/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_10/AssignMovingAvg/ReadVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
(__inference_while_body_12836706_23173004
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
�
�
(__inference_while_cond_12835799_23171895
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12835799___redundant_placeholder00
,while_cond_12835799___redundant_placeholder10
,while_cond_12835799___redundant_placeholder20
,while_cond_12835799___redundant_placeholder3
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
�4
�
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23173822
input_1)
%lstm_3_statefulpartitionedcall_args_1)
%lstm_3_statefulpartitionedcall_args_2)
%lstm_3_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6+
'dense_15_statefulpartitionedcall_args_1+
'dense_15_statefulpartitionedcall_args_2
identity�� dense_15/StatefulPartitionedCall�lstm_3/StatefulPartitionedCall�<simple_neural_network_layer_block_10/StatefulPartitionedCall�<simple_neural_network_layer_block_11/StatefulPartitionedCall�;simple_neural_network_layer_block_9/StatefulPartitionedCall�
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinput_1%lstm_3_statefulpartitionedcall_args_1%lstm_3_statefulpartitionedcall_args_2%lstm_3_statefulpartitionedcall_args_3*
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_231717502 
lstm_3/StatefulPartitionedCall�
;simple_neural_network_layer_block_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6*
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
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_231725312=
;simple_neural_network_layer_block_9/StatefulPartitionedCall�
<simple_neural_network_layer_block_10/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_9/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6*
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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_231731422>
<simple_neural_network_layer_block_10/StatefulPartitionedCall�
<simple_neural_network_layer_block_11/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_10/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6*
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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_231736912>
<simple_neural_network_layer_block_11/StatefulPartitionedCall�
 dense_15/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_11/StatefulPartitionedCall:output:0'dense_15_statefulpartitionedcall_args_1'dense_15_statefulpartitionedcall_args_2*
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
F__inference_dense_15_layer_call_and_return_conditional_losses_231724942"
 dense_15/StatefulPartitionedCall�
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall=^simple_neural_network_layer_block_10/StatefulPartitionedCall=^simple_neural_network_layer_block_11/StatefulPartitionedCall<^simple_neural_network_layer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2|
<simple_neural_network_layer_block_10/StatefulPartitionedCall<simple_neural_network_layer_block_10/StatefulPartitionedCall2|
<simple_neural_network_layer_block_11/StatefulPartitionedCall<simple_neural_network_layer_block_11/StatefulPartitionedCall2z
;simple_neural_network_layer_block_9/StatefulPartitionedCall;simple_neural_network_layer_block_9/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�.
�
(__inference_while_body_12832573_23172336
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
�*
�
__inference_call_23171594

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource7
3batch_normalization_11_cast_readvariableop_resource9
5batch_normalization_11_cast_1_readvariableop_resource9
5batch_normalization_11_cast_2_readvariableop_resource9
5batch_normalization_11_cast_3_readvariableop_resource
identity��*batch_normalization_11/Cast/ReadVariableOp�,batch_normalization_11/Cast_1/ReadVariableOp�,batch_normalization_11/Cast_2/ReadVariableOp�,batch_normalization_11/Cast_3/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/BiasAdd�
#batch_normalization_11/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_11/LogicalAnd/x�
#batch_normalization_11/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/y�
!batch_normalization_11/LogicalAnd
LogicalAnd,batch_normalization_11/LogicalAnd/x:output:0,batch_normalization_11/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_11/LogicalAnd�
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_11/Cast/ReadVariableOp�
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp�
,batch_normalization_11/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_11_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_2/ReadVariableOp�
,batch_normalization_11/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_11_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_3/ReadVariableOp�
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_11/batchnorm/add/y�
$batch_normalization_11/batchnorm/addAddV24batch_normalization_11/Cast_1/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/add�
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/Rsqrt�
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/mul�
&batch_normalization_11/batchnorm/mul_1Muldense_14/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/mul_1�
&batch_normalization_11/batchnorm/mul_2Mul2batch_normalization_11/Cast/ReadVariableOp:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/mul_2�
$batch_normalization_11/batchnorm/subSub4batch_normalization_11/Cast_2/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/sub�
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/add_1s
TanhTanh*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp-^batch_normalization_11/Cast_2/ReadVariableOp-^batch_normalization_11/Cast_3/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2\
,batch_normalization_11/Cast_2/ReadVariableOp,batch_normalization_11/Cast_2/ReadVariableOp2\
,batch_normalization_11/Cast_3/ReadVariableOp,batch_normalization_11/Cast_3/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
#__inference__wrapped_model_46361856
input_1>
:lstm_neural_network_model_3_statefulpartitionedcall_args_1>
:lstm_neural_network_model_3_statefulpartitionedcall_args_2>
:lstm_neural_network_model_3_statefulpartitionedcall_args_3>
:lstm_neural_network_model_3_statefulpartitionedcall_args_4>
:lstm_neural_network_model_3_statefulpartitionedcall_args_5>
:lstm_neural_network_model_3_statefulpartitionedcall_args_6>
:lstm_neural_network_model_3_statefulpartitionedcall_args_7>
:lstm_neural_network_model_3_statefulpartitionedcall_args_8>
:lstm_neural_network_model_3_statefulpartitionedcall_args_9?
;lstm_neural_network_model_3_statefulpartitionedcall_args_10?
;lstm_neural_network_model_3_statefulpartitionedcall_args_11?
;lstm_neural_network_model_3_statefulpartitionedcall_args_12?
;lstm_neural_network_model_3_statefulpartitionedcall_args_13?
;lstm_neural_network_model_3_statefulpartitionedcall_args_14?
;lstm_neural_network_model_3_statefulpartitionedcall_args_15?
;lstm_neural_network_model_3_statefulpartitionedcall_args_16?
;lstm_neural_network_model_3_statefulpartitionedcall_args_17?
;lstm_neural_network_model_3_statefulpartitionedcall_args_18?
;lstm_neural_network_model_3_statefulpartitionedcall_args_19?
;lstm_neural_network_model_3_statefulpartitionedcall_args_20?
;lstm_neural_network_model_3_statefulpartitionedcall_args_21?
;lstm_neural_network_model_3_statefulpartitionedcall_args_22?
;lstm_neural_network_model_3_statefulpartitionedcall_args_23
identity��3lstm_neural_network_model_3/StatefulPartitionedCall�
3lstm_neural_network_model_3/StatefulPartitionedCallStatefulPartitionedCallinput_1:lstm_neural_network_model_3_statefulpartitionedcall_args_1:lstm_neural_network_model_3_statefulpartitionedcall_args_2:lstm_neural_network_model_3_statefulpartitionedcall_args_3:lstm_neural_network_model_3_statefulpartitionedcall_args_4:lstm_neural_network_model_3_statefulpartitionedcall_args_5:lstm_neural_network_model_3_statefulpartitionedcall_args_6:lstm_neural_network_model_3_statefulpartitionedcall_args_7:lstm_neural_network_model_3_statefulpartitionedcall_args_8:lstm_neural_network_model_3_statefulpartitionedcall_args_9;lstm_neural_network_model_3_statefulpartitionedcall_args_10;lstm_neural_network_model_3_statefulpartitionedcall_args_11;lstm_neural_network_model_3_statefulpartitionedcall_args_12;lstm_neural_network_model_3_statefulpartitionedcall_args_13;lstm_neural_network_model_3_statefulpartitionedcall_args_14;lstm_neural_network_model_3_statefulpartitionedcall_args_15;lstm_neural_network_model_3_statefulpartitionedcall_args_16;lstm_neural_network_model_3_statefulpartitionedcall_args_17;lstm_neural_network_model_3_statefulpartitionedcall_args_18;lstm_neural_network_model_3_statefulpartitionedcall_args_19;lstm_neural_network_model_3_statefulpartitionedcall_args_20;lstm_neural_network_model_3_statefulpartitionedcall_args_21;lstm_neural_network_model_3_statefulpartitionedcall_args_22;lstm_neural_network_model_3_statefulpartitionedcall_args_23*#
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
+__inference_restored_function_body_2317596525
3lstm_neural_network_model_3/StatefulPartitionedCall�
IdentityIdentity<lstm_neural_network_model_3/StatefulPartitionedCall:output:04^lstm_neural_network_model_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2j
3lstm_neural_network_model_3/StatefulPartitionedCall3lstm_neural_network_model_3/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
�
D__inference_lstm_3_layer_call_and_return_conditional_losses_23172893

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
"__inference_standard_lstm_231728602
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
�*
�
__inference_call_23172033

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource7
3batch_normalization_10_cast_readvariableop_resource9
5batch_normalization_10_cast_1_readvariableop_resource9
5batch_normalization_10_cast_2_readvariableop_resource9
5batch_normalization_10_cast_3_readvariableop_resource
identity��*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�,batch_normalization_10/Cast_2/ReadVariableOp�,batch_normalization_10/Cast_3/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOp�
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_13/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/add_1s
TanhTanh*batch_normalization_10/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2\
,batch_normalization_10/Cast_2/ReadVariableOp,batch_normalization_10/Cast_2/ReadVariableOp2\
,batch_normalization_10/Cast_3/ReadVariableOp,batch_normalization_10/Cast_3/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
(__inference_while_body_12835800_23171880
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
�*
�
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23173142

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource7
3batch_normalization_10_cast_readvariableop_resource9
5batch_normalization_10_cast_1_readvariableop_resource9
5batch_normalization_10_cast_2_readvariableop_resource9
5batch_normalization_10_cast_3_readvariableop_resource
identity��*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�,batch_normalization_10/Cast_2/ReadVariableOp�,batch_normalization_10/Cast_3/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOp�
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_13/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/add_1s
TanhTanh*batch_normalization_10/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2\
,batch_normalization_10/Cast_2/ReadVariableOp,batch_normalization_10/Cast_2/ReadVariableOp2\
,batch_normalization_10/Cast_3/ReadVariableOp,batch_normalization_10/Cast_3/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
G__inference_simple_neural_network_layer_block_11_layer_call_fn_23173790

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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_231736912
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
(__inference_while_cond_12837150_23171773
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12837150___redundant_placeholder00
,while_cond_12837150___redundant_placeholder10
,while_cond_12837150___redundant_placeholder20
,while_cond_12837150___redundant_placeholder3
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
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23174989

inputs+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource6
2batch_normalization_9_cast_readvariableop_resource8
4batch_normalization_9_cast_1_readvariableop_resource8
4batch_normalization_9_cast_2_readvariableop_resource8
4batch_normalization_9_cast_3_readvariableop_resource
identity��)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�+batch_normalization_9/Cast_2/ReadVariableOp�+batch_normalization_9/Cast_3/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOp�
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_12/BiasAdd:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/add_1r
TanhTanh)batch_normalization_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2Z
+batch_normalization_9/Cast_2/ReadVariableOp+batch_normalization_9/Cast_2/ReadVariableOp2Z
+batch_normalization_9/Cast_3/ReadVariableOp+batch_normalization_9/Cast_3/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�O
�
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23173606

inputs+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource2
.batch_normalization_9_assignmovingavg_128319924
0batch_normalization_9_assignmovingavg_1_128319986
2batch_normalization_9_cast_readvariableop_resource8
4batch_normalization_9_cast_1_readvariableop_resource
identity��9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_9/AssignMovingAvg/ReadVariableOp�;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices�
"batch_normalization_9/moments/meanMeandense_12/BiasAdd:output:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_9/moments/mean�
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_9/moments/StopGradient�
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_12/BiasAdd:output:03batch_normalization_9/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_9/moments/SquaredDifference�
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices�
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_9/moments/variance�
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_9/moments/Squeeze�
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1�
+batch_normalization_9/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12831992*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_9/AssignMovingAvg/decay�
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_9_assignmovingavg_12831992*
_output_shapes	
:�*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp�
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12831992*
_output_shapes	
:�2+
)batch_normalization_9/AssignMovingAvg/sub�
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12831992*
_output_shapes	
:�2+
)batch_normalization_9/AssignMovingAvg/mul�
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_9_assignmovingavg_12831992-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12831992*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_9/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12831998*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_9/AssignMovingAvg_1/decay�
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_9_assignmovingavg_1_12831998*
_output_shapes	
:�*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12831998*
_output_shapes	
:�2-
+batch_normalization_9/AssignMovingAvg_1/sub�
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12831998*
_output_shapes	
:�2-
+batch_normalization_9/AssignMovingAvg_1/mul�
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_9_assignmovingavg_1_12831998/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12831998*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_12/BiasAdd:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub1batch_normalization_9/Cast/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/add_1r
TanhTanh)batch_normalization_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_9/AssignMovingAvg/ReadVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
>__inference_lstm_neural_network_model_3_layer_call_fn_23175203
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
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_231751752
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
�P
�
__inference_call_23174033

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource3
/batch_normalization_10_assignmovingavg_128330175
1batch_normalization_10_assignmovingavg_1_128330237
3batch_normalization_10_cast_readvariableop_resource9
5batch_normalization_10_cast_1_readvariableop_resource
identity��:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_10/AssignMovingAvg/ReadVariableOp�<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indices�
#batch_normalization_10/moments/meanMeandense_13/BiasAdd:output:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2%
#batch_normalization_10/moments/mean�
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:	�2-
+batch_normalization_10/moments/StopGradient�
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_13/BiasAdd:output:04batch_normalization_10/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������22
0batch_normalization_10/moments/SquaredDifference�
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices�
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2)
'batch_normalization_10/moments/variance�
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze�
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1�
,batch_normalization_10/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12833017*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_10/AssignMovingAvg/decay�
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_10_assignmovingavg_12833017*
_output_shapes	
:�*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp�
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12833017*
_output_shapes	
:�2,
*batch_normalization_10/AssignMovingAvg/sub�
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12833017*
_output_shapes	
:�2,
*batch_normalization_10/AssignMovingAvg/mul�
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_10_assignmovingavg_12833017.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12833017*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_10/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12833023*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_10/AssignMovingAvg_1/decay�
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_10_assignmovingavg_1_12833023*
_output_shapes	
:�*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12833023*
_output_shapes	
:�2.
,batch_normalization_10/AssignMovingAvg_1/sub�
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12833023*
_output_shapes	
:�2.
,batch_normalization_10/AssignMovingAvg_1/mul�
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_10_assignmovingavg_1_128330230batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12833023*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_13/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub2batch_normalization_10/Cast/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/add_1s
TanhTanh*batch_normalization_10/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_10/AssignMovingAvg/ReadVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�*
�
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23172531

inputs+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource6
2batch_normalization_9_cast_readvariableop_resource8
4batch_normalization_9_cast_1_readvariableop_resource8
4batch_normalization_9_cast_2_readvariableop_resource8
4batch_normalization_9_cast_3_readvariableop_resource
identity��)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�+batch_normalization_9/Cast_2/ReadVariableOp�+batch_normalization_9/Cast_3/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOp�
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_12/BiasAdd:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/add_1r
TanhTanh)batch_normalization_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2Z
+batch_normalization_9/Cast_2/ReadVariableOp+batch_normalization_9/Cast_2/ReadVariableOp2Z
+batch_normalization_9/Cast_3/ReadVariableOp+batch_normalization_9/Cast_3/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23174320

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
�P
�
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23173661

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource3
/batch_normalization_10_assignmovingavg_128379585
1batch_normalization_10_assignmovingavg_1_128379647
3batch_normalization_10_cast_readvariableop_resource9
5batch_normalization_10_cast_1_readvariableop_resource
identity��:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_10/AssignMovingAvg/ReadVariableOp�<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
5batch_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_10/moments/mean/reduction_indices�
#batch_normalization_10/moments/meanMeandense_13/BiasAdd:output:0>batch_normalization_10/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2%
#batch_normalization_10/moments/mean�
+batch_normalization_10/moments/StopGradientStopGradient,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes
:	�2-
+batch_normalization_10/moments/StopGradient�
0batch_normalization_10/moments/SquaredDifferenceSquaredDifferencedense_13/BiasAdd:output:04batch_normalization_10/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������22
0batch_normalization_10/moments/SquaredDifference�
9batch_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_10/moments/variance/reduction_indices�
'batch_normalization_10/moments/varianceMean4batch_normalization_10/moments/SquaredDifference:z:0Bbatch_normalization_10/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2)
'batch_normalization_10/moments/variance�
&batch_normalization_10/moments/SqueezeSqueeze,batch_normalization_10/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2(
&batch_normalization_10/moments/Squeeze�
(batch_normalization_10/moments/Squeeze_1Squeeze0batch_normalization_10/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2*
(batch_normalization_10/moments/Squeeze_1�
,batch_normalization_10/AssignMovingAvg/decayConst*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12837958*
_output_shapes
: *
dtype0*
valueB
 *
�#<2.
,batch_normalization_10/AssignMovingAvg/decay�
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp/batch_normalization_10_assignmovingavg_12837958*
_output_shapes	
:�*
dtype027
5batch_normalization_10/AssignMovingAvg/ReadVariableOp�
*batch_normalization_10/AssignMovingAvg/subSub=batch_normalization_10/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_10/moments/Squeeze:output:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12837958*
_output_shapes	
:�2,
*batch_normalization_10/AssignMovingAvg/sub�
*batch_normalization_10/AssignMovingAvg/mulMul.batch_normalization_10/AssignMovingAvg/sub:z:05batch_normalization_10/AssignMovingAvg/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12837958*
_output_shapes	
:�2,
*batch_normalization_10/AssignMovingAvg/mul�
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp/batch_normalization_10_assignmovingavg_12837958.batch_normalization_10/AssignMovingAvg/mul:z:06^batch_normalization_10/AssignMovingAvg/ReadVariableOp*B
_class8
64loc:@batch_normalization_10/AssignMovingAvg/12837958*
_output_shapes
 *
dtype02<
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_10/AssignMovingAvg_1/decayConst*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12837964*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.batch_normalization_10/AssignMovingAvg_1/decay�
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp1batch_normalization_10_assignmovingavg_1_12837964*
_output_shapes	
:�*
dtype029
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp�
,batch_normalization_10/AssignMovingAvg_1/subSub?batch_normalization_10/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_10/moments/Squeeze_1:output:0*
T0*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12837964*
_output_shapes	
:�2.
,batch_normalization_10/AssignMovingAvg_1/sub�
,batch_normalization_10/AssignMovingAvg_1/mulMul0batch_normalization_10/AssignMovingAvg_1/sub:z:07batch_normalization_10/AssignMovingAvg_1/decay:output:0*
T0*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12837964*
_output_shapes	
:�2.
,batch_normalization_10/AssignMovingAvg_1/mul�
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp1batch_normalization_10_assignmovingavg_1_128379640batch_normalization_10/AssignMovingAvg_1/mul:z:08^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp*D
_class:
86loc:@batch_normalization_10/AssignMovingAvg_1/12837964*
_output_shapes
 *
dtype02>
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV21batch_normalization_10/moments/Squeeze_1:output:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_13/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul/batch_normalization_10/moments/Squeeze:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub2batch_normalization_10/Cast/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/add_1s
TanhTanh*batch_normalization_10/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_10/AssignMovingAvg/ReadVariableOp=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_10/AssignMovingAvg_1/ReadVariableOp+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2x
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp:batch_normalization_10/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_10/AssignMovingAvg/ReadVariableOp5batch_normalization_10/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
>__inference_lstm_neural_network_model_3_layer_call_fn_23173779

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
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_231737232
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
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23172156

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
"__inference_standard_lstm_23174169

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
(__inference_while_body_12832573_23172336*4
cond,R*
(__inference_while_cond_12832572_23174094*M
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
�*
�
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23174260

inputs+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource7
3batch_normalization_10_cast_readvariableop_resource9
5batch_normalization_10_cast_1_readvariableop_resource9
5batch_normalization_10_cast_2_readvariableop_resource9
5batch_normalization_10_cast_3_readvariableop_resource
identity��*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�,batch_normalization_10/Cast_2/ReadVariableOp�,batch_normalization_10/Cast_3/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulinputs&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOp�
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_13/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/add_1s
TanhTanh*batch_normalization_10/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2\
,batch_normalization_10/Cast_2/ReadVariableOp,batch_normalization_10/Cast_2/ReadVariableOp2\
,batch_normalization_10/Cast_3/ReadVariableOp,batch_normalization_10/Cast_3/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�G
�
"__inference_standard_lstm_23171717

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
(__inference_while_body_12831578_23171642*4
cond,R*
(__inference_while_cond_12831577_23171520*M
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_23173112

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
"__inference_standard_lstm_231730792
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
�.
�
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23174070

inputs
assignmovingavg_12830994
assignmovingavg_1_12831000 
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
loc:@AssignMovingAvg/12830994*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_12830994*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12830994*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12830994*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_12830994AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/12830994*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/12831000*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_12831000*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12831000*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12831000*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_12831000AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/12831000*
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
�3
�
___forward_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23772230
inputs_0+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource6
2batch_normalization_9_cast_readvariableop_resource8
4batch_normalization_9_cast_1_readvariableop_resource8
4batch_normalization_9_cast_2_readvariableop_resource8
4batch_normalization_9_cast_3_readvariableop_resource
identity
tanh)
%batch_normalization_9_batchnorm_mul_1'
#batch_normalization_9_batchnorm_sub
dense_12_biasadd'
#batch_normalization_9_batchnorm_mul-
)batch_normalization_9_cast_readvariableop"
dense_12_matmul_readvariableop

inputs/
+batch_normalization_9_cast_3_readvariableop)
%batch_normalization_9_batchnorm_rsqrt��)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�+batch_normalization_9/Cast_2/ReadVariableOp�+batch_normalization_9/Cast_3/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulinputs_0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
+batch_normalization_9/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_9_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_2/ReadVariableOp�
+batch_normalization_9/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_9_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_3/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV23batch_normalization_9/Cast_1/ReadVariableOp:value:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_12/BiasAdd:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul1batch_normalization_9/Cast/ReadVariableOp:value:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub3batch_normalization_9/Cast_2/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/add_1r
TanhTanh)batch_normalization_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp,^batch_normalization_9/Cast_2/ReadVariableOp,^batch_normalization_9/Cast_3/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"N
#batch_normalization_9_batchnorm_mul'batch_normalization_9/batchnorm/mul:z:0"R
%batch_normalization_9_batchnorm_mul_1)batch_normalization_9/batchnorm/mul_1:z:0"R
%batch_normalization_9_batchnorm_rsqrt)batch_normalization_9/batchnorm/Rsqrt:y:0"N
#batch_normalization_9_batchnorm_sub'batch_normalization_9/batchnorm/sub:z:0"b
+batch_normalization_9_cast_3_readvariableop3batch_normalization_9/Cast_3/ReadVariableOp:value:0"^
)batch_normalization_9_cast_readvariableop1batch_normalization_9/Cast/ReadVariableOp:value:0"-
dense_12_biasadddense_12/BiasAdd:output:0"H
dense_12_matmul_readvariableop&dense_12/MatMul/ReadVariableOp:value:0"
identityIdentity:output:0"
inputsinputs_0"
tanhTanh:y:0*?
_input_shapes.
,:����������::::::*�
backward_function_namewu__inference___backward_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23772172_237722312V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2Z
+batch_normalization_9/Cast_2/ReadVariableOp+batch_normalization_9/Cast_2/ReadVariableOp2Z
+batch_normalization_9/Cast_3/ReadVariableOp+batch_normalization_9/Cast_3/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�.
�
(__inference_while_body_12836245_23174851
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
�M
�
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23174606

inputs)
%lstm_3_statefulpartitionedcall_args_3)
%lstm_3_statefulpartitionedcall_args_4)
%lstm_3_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity��dense_15/BiasAdd/ReadVariableOp�dense_15/MatMul/ReadVariableOp�lstm_3/StatefulPartitionedCall�<simple_neural_network_layer_block_10/StatefulPartitionedCall�<simple_neural_network_layer_block_11/StatefulPartitionedCall�;simple_neural_network_layer_block_9/StatefulPartitionedCallR
lstm_3/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_3/Shape�
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack�
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1�
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2�
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicek
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros/mul/y�
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros/Less/y�
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessq
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros/packed/1�
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const�
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:����������2
lstm_3/zeroso
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros_1/mul/y�
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros_1/Less/y�
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lessu
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros_1/packed/1�
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const�
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
lstm_3/zeros_1�
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinputslstm_3/zeros:output:0lstm_3/zeros_1:output:0%lstm_3_statefulpartitionedcall_args_3%lstm_3_statefulpartitionedcall_args_4%lstm_3_statefulpartitionedcall_args_5*
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
"__inference_standard_lstm_231726322 
lstm_3/StatefulPartitionedCall�
;simple_neural_network_layer_block_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6*
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
__inference_call_231745452=
;simple_neural_network_layer_block_9/StatefulPartitionedCall�
<simple_neural_network_layer_block_10/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_9/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6*
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
__inference_call_231720332>
<simple_neural_network_layer_block_10/StatefulPartitionedCall�
<simple_neural_network_layer_block_11/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_10/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6*
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
__inference_call_231715942>
<simple_neural_network_layer_block_11/StatefulPartitionedCall�
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_15/MatMul/ReadVariableOp�
dense_15/MatMulMatMulEsimple_neural_network_layer_block_11/StatefulPartitionedCall:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_15/MatMul�
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp�
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_15/BiasAdds
dense_15/TanhTanhdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_15/Tanh�
IdentityIdentitydense_15/Tanh:y:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp^lstm_3/StatefulPartitionedCall=^simple_neural_network_layer_block_10/StatefulPartitionedCall=^simple_neural_network_layer_block_11/StatefulPartitionedCall<^simple_neural_network_layer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2|
<simple_neural_network_layer_block_10/StatefulPartitionedCall<simple_neural_network_layer_block_10/StatefulPartitionedCall2|
<simple_neural_network_layer_block_11/StatefulPartitionedCall<simple_neural_network_layer_block_11/StatefulPartitionedCall2z
;simple_neural_network_layer_block_9/StatefulPartitionedCall;simple_neural_network_layer_block_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
>__inference_lstm_neural_network_model_3_layer_call_fn_23175231

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
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_231751752
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
�G
�
"__inference_standard_lstm_23172632

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
(__inference_while_body_12833184_23172483*4
cond,R*
(__inference_while_cond_12833183_23172557*M
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_23173309

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
"__inference_standard_lstm_231732762
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
�	
�
G__inference_simple_neural_network_layer_block_11_layer_call_fn_23172956

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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_231729452
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
�*
�
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23173691

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource7
3batch_normalization_11_cast_readvariableop_resource9
5batch_normalization_11_cast_1_readvariableop_resource9
5batch_normalization_11_cast_2_readvariableop_resource9
5batch_normalization_11_cast_3_readvariableop_resource
identity��*batch_normalization_11/Cast/ReadVariableOp�,batch_normalization_11/Cast_1/ReadVariableOp�,batch_normalization_11/Cast_2/ReadVariableOp�,batch_normalization_11/Cast_3/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/BiasAdd�
#batch_normalization_11/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_11/LogicalAnd/x�
#batch_normalization_11/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/y�
!batch_normalization_11/LogicalAnd
LogicalAnd,batch_normalization_11/LogicalAnd/x:output:0,batch_normalization_11/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_11/LogicalAnd�
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_11/Cast/ReadVariableOp�
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp�
,batch_normalization_11/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_11_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_2/ReadVariableOp�
,batch_normalization_11/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_11_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_3/ReadVariableOp�
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_11/batchnorm/add/y�
$batch_normalization_11/batchnorm/addAddV24batch_normalization_11/Cast_1/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/add�
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/Rsqrt�
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/mul�
&batch_normalization_11/batchnorm/mul_1Muldense_14/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/mul_1�
&batch_normalization_11/batchnorm/mul_2Mul2batch_normalization_11/Cast/ReadVariableOp:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/mul_2�
$batch_normalization_11/batchnorm/subSub4batch_normalization_11/Cast_2/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/sub�
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/add_1s
TanhTanh*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp-^batch_normalization_11/Cast_2/ReadVariableOp-^batch_normalization_11/Cast_3/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2\
,batch_normalization_11/Cast_2/ReadVariableOp,batch_normalization_11/Cast_2/ReadVariableOp2\
,batch_normalization_11/Cast_3/ReadVariableOp,batch_normalization_11/Cast_3/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_lstm_3_layer_call_fn_23173317
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_231733092
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
�
�
(__inference_while_cond_12831577_23171520
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12831577___redundant_placeholder00
,while_cond_12831577___redundant_placeholder10
,while_cond_12831577___redundant_placeholder20
,while_cond_12831577___redundant_placeholder3
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
�
�
9__inference_batch_normalization_10_layer_call_fn_23173952

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
GPU2*0J 8*]
fXRV
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_231739432
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
�
�
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_23174352

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
�
�
)__inference_lstm_3_layer_call_fn_23172901
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_231728932
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
�
�
(__inference_while_cond_12833183_23172557
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12833183___redundant_placeholder00
,while_cond_12833183___redundant_placeholder10
,while_cond_12833183___redundant_placeholder20
,while_cond_12833183___redundant_placeholder3
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
�O
�
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23171564

inputs+
'dense_12_matmul_readvariableop_resource,
(dense_12_biasadd_readvariableop_resource2
.batch_normalization_9_assignmovingavg_128375624
0batch_normalization_9_assignmovingavg_1_128375686
2batch_normalization_9_cast_readvariableop_resource8
4batch_normalization_9_cast_1_readvariableop_resource
identity��9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_9/AssignMovingAvg/ReadVariableOp�;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�)batch_normalization_9/Cast/ReadVariableOp�+batch_normalization_9/Cast_1/ReadVariableOp�dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_12/BiasAdd�
"batch_normalization_9/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/x�
"batch_normalization_9/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_9/LogicalAnd/y�
 batch_normalization_9/LogicalAnd
LogicalAnd+batch_normalization_9/LogicalAnd/x:output:0+batch_normalization_9/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_9/LogicalAnd�
4batch_normalization_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_9/moments/mean/reduction_indices�
"batch_normalization_9/moments/meanMeandense_12/BiasAdd:output:0=batch_normalization_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2$
"batch_normalization_9/moments/mean�
*batch_normalization_9/moments/StopGradientStopGradient+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes
:	�2,
*batch_normalization_9/moments/StopGradient�
/batch_normalization_9/moments/SquaredDifferenceSquaredDifferencedense_12/BiasAdd:output:03batch_normalization_9/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������21
/batch_normalization_9/moments/SquaredDifference�
8batch_normalization_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_9/moments/variance/reduction_indices�
&batch_normalization_9/moments/varianceMean3batch_normalization_9/moments/SquaredDifference:z:0Abatch_normalization_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(2(
&batch_normalization_9/moments/variance�
%batch_normalization_9/moments/SqueezeSqueeze+batch_normalization_9/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2'
%batch_normalization_9/moments/Squeeze�
'batch_normalization_9/moments/Squeeze_1Squeeze/batch_normalization_9/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 2)
'batch_normalization_9/moments/Squeeze_1�
+batch_normalization_9/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12837562*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_9/AssignMovingAvg/decay�
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_9_assignmovingavg_12837562*
_output_shapes	
:�*
dtype026
4batch_normalization_9/AssignMovingAvg/ReadVariableOp�
)batch_normalization_9/AssignMovingAvg/subSub<batch_normalization_9/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_9/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12837562*
_output_shapes	
:�2+
)batch_normalization_9/AssignMovingAvg/sub�
)batch_normalization_9/AssignMovingAvg/mulMul-batch_normalization_9/AssignMovingAvg/sub:z:04batch_normalization_9/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12837562*
_output_shapes	
:�2+
)batch_normalization_9/AssignMovingAvg/mul�
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_9_assignmovingavg_12837562-batch_normalization_9/AssignMovingAvg/mul:z:05^batch_normalization_9/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_9/AssignMovingAvg/12837562*
_output_shapes
 *
dtype02;
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_9/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12837568*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_9/AssignMovingAvg_1/decay�
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_9_assignmovingavg_1_12837568*
_output_shapes	
:�*
dtype028
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_9/AssignMovingAvg_1/subSub>batch_normalization_9/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_9/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12837568*
_output_shapes	
:�2-
+batch_normalization_9/AssignMovingAvg_1/sub�
+batch_normalization_9/AssignMovingAvg_1/mulMul/batch_normalization_9/AssignMovingAvg_1/sub:z:06batch_normalization_9/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12837568*
_output_shapes	
:�2-
+batch_normalization_9/AssignMovingAvg_1/mul�
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_9_assignmovingavg_1_12837568/batch_normalization_9/AssignMovingAvg_1/mul:z:07^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_9/AssignMovingAvg_1/12837568*
_output_shapes
 *
dtype02=
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp�
)batch_normalization_9/Cast/ReadVariableOpReadVariableOp2batch_normalization_9_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02+
)batch_normalization_9/Cast/ReadVariableOp�
+batch_normalization_9/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_9_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+batch_normalization_9/Cast_1/ReadVariableOp�
%batch_normalization_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_9/batchnorm/add/y�
#batch_normalization_9/batchnorm/addAddV20batch_normalization_9/moments/Squeeze_1:output:0.batch_normalization_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/add�
%batch_normalization_9/batchnorm/RsqrtRsqrt'batch_normalization_9/batchnorm/add:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/Rsqrt�
#batch_normalization_9/batchnorm/mulMul)batch_normalization_9/batchnorm/Rsqrt:y:03batch_normalization_9/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/mul�
%batch_normalization_9/batchnorm/mul_1Muldense_12/BiasAdd:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/mul_1�
%batch_normalization_9/batchnorm/mul_2Mul.batch_normalization_9/moments/Squeeze:output:0'batch_normalization_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2'
%batch_normalization_9/batchnorm/mul_2�
#batch_normalization_9/batchnorm/subSub1batch_normalization_9/Cast/ReadVariableOp:value:0)batch_normalization_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2%
#batch_normalization_9/batchnorm/sub�
%batch_normalization_9/batchnorm/add_1AddV2)batch_normalization_9/batchnorm/mul_1:z:0'batch_normalization_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2'
%batch_normalization_9/batchnorm/add_1r
TanhTanh)batch_normalization_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_9/AssignMovingAvg/ReadVariableOp<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_9/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_9/Cast/ReadVariableOp,^batch_normalization_9/Cast_1/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2v
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp9batch_normalization_9/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_9/AssignMovingAvg/ReadVariableOp4batch_normalization_9/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_9/Cast/ReadVariableOp)batch_normalization_9/Cast/ReadVariableOp2Z
+batch_normalization_9/Cast_1/ReadVariableOp+batch_normalization_9/Cast_1/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_12830246_23172785
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12830246___redundant_placeholder00
,while_cond_12830246___redundant_placeholder10
,while_cond_12830246___redundant_placeholder20
,while_cond_12830246___redundant_placeholder3
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
�3
�
`__forward_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23772143
inputs_0+
'dense_13_matmul_readvariableop_resource,
(dense_13_biasadd_readvariableop_resource7
3batch_normalization_10_cast_readvariableop_resource9
5batch_normalization_10_cast_1_readvariableop_resource9
5batch_normalization_10_cast_2_readvariableop_resource9
5batch_normalization_10_cast_3_readvariableop_resource
identity
tanh*
&batch_normalization_10_batchnorm_mul_1(
$batch_normalization_10_batchnorm_sub
dense_13_biasadd(
$batch_normalization_10_batchnorm_mul.
*batch_normalization_10_cast_readvariableop"
dense_13_matmul_readvariableop

inputs0
,batch_normalization_10_cast_3_readvariableop*
&batch_normalization_10_batchnorm_rsqrt��*batch_normalization_10/Cast/ReadVariableOp�,batch_normalization_10/Cast_1/ReadVariableOp�,batch_normalization_10/Cast_2/ReadVariableOp�,batch_normalization_10/Cast_3/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_13/MatMul/ReadVariableOp�
dense_13/MatMulMatMulinputs_0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/MatMul�
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_13/BiasAdd/ReadVariableOp�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_13/BiasAdd�
#batch_normalization_10/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_10/LogicalAnd/x�
#batch_normalization_10/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_10/LogicalAnd/y�
!batch_normalization_10/LogicalAnd
LogicalAnd,batch_normalization_10/LogicalAnd/x:output:0,batch_normalization_10/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_10/LogicalAnd�
*batch_normalization_10/Cast/ReadVariableOpReadVariableOp3batch_normalization_10_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_10/Cast/ReadVariableOp�
,batch_normalization_10/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_10_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_1/ReadVariableOp�
,batch_normalization_10/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_10_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_2/ReadVariableOp�
,batch_normalization_10/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_10_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_10/Cast_3/ReadVariableOp�
&batch_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_10/batchnorm/add/y�
$batch_normalization_10/batchnorm/addAddV24batch_normalization_10/Cast_1/ReadVariableOp:value:0/batch_normalization_10/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/add�
&batch_normalization_10/batchnorm/RsqrtRsqrt(batch_normalization_10/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/Rsqrt�
$batch_normalization_10/batchnorm/mulMul*batch_normalization_10/batchnorm/Rsqrt:y:04batch_normalization_10/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/mul�
&batch_normalization_10/batchnorm/mul_1Muldense_13/BiasAdd:output:0(batch_normalization_10/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/mul_1�
&batch_normalization_10/batchnorm/mul_2Mul2batch_normalization_10/Cast/ReadVariableOp:value:0(batch_normalization_10/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_10/batchnorm/mul_2�
$batch_normalization_10/batchnorm/subSub4batch_normalization_10/Cast_2/ReadVariableOp:value:0*batch_normalization_10/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_10/batchnorm/sub�
&batch_normalization_10/batchnorm/add_1AddV2*batch_normalization_10/batchnorm/mul_1:z:0(batch_normalization_10/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_10/batchnorm/add_1s
TanhTanh*batch_normalization_10/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0+^batch_normalization_10/Cast/ReadVariableOp-^batch_normalization_10/Cast_1/ReadVariableOp-^batch_normalization_10/Cast_2/ReadVariableOp-^batch_normalization_10/Cast_3/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"P
$batch_normalization_10_batchnorm_mul(batch_normalization_10/batchnorm/mul:z:0"T
&batch_normalization_10_batchnorm_mul_1*batch_normalization_10/batchnorm/mul_1:z:0"T
&batch_normalization_10_batchnorm_rsqrt*batch_normalization_10/batchnorm/Rsqrt:y:0"P
$batch_normalization_10_batchnorm_sub(batch_normalization_10/batchnorm/sub:z:0"d
,batch_normalization_10_cast_3_readvariableop4batch_normalization_10/Cast_3/ReadVariableOp:value:0"`
*batch_normalization_10_cast_readvariableop2batch_normalization_10/Cast/ReadVariableOp:value:0"-
dense_13_biasadddense_13/BiasAdd:output:0"H
dense_13_matmul_readvariableop&dense_13/MatMul/ReadVariableOp:value:0"
identityIdentity:output:0"
inputsinputs_0"
tanhTanh:y:0*?
_input_shapes.
,:����������::::::*�
backward_function_namexv__inference___backward_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23772085_237721442X
*batch_normalization_10/Cast/ReadVariableOp*batch_normalization_10/Cast/ReadVariableOp2\
,batch_normalization_10/Cast_1/ReadVariableOp,batch_normalization_10/Cast_1/ReadVariableOp2\
,batch_normalization_10/Cast_2/ReadVariableOp,batch_normalization_10/Cast_2/ReadVariableOp2\
,batch_normalization_10/Cast_3/ReadVariableOp,batch_normalization_10/Cast_3/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
F__inference_dense_15_layer_call_and_return_conditional_losses_23172494

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
D__inference_lstm_3_layer_call_and_return_conditional_losses_23174773

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
"__inference_standard_lstm_231747402
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
�.
�
(__inference_while_body_12829794_23173201
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
�
�
&__inference_signature_wrapper_46361885
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
#__inference__wrapped_model_463618562
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
�W
�
2__inference_while_body_12831578_23171642_rewritten
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
�4
�
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23173723

inputs)
%lstm_3_statefulpartitionedcall_args_1)
%lstm_3_statefulpartitionedcall_args_2)
%lstm_3_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6+
'dense_15_statefulpartitionedcall_args_1+
'dense_15_statefulpartitionedcall_args_2
identity�� dense_15/StatefulPartitionedCall�lstm_3/StatefulPartitionedCall�<simple_neural_network_layer_block_10/StatefulPartitionedCall�<simple_neural_network_layer_block_11/StatefulPartitionedCall�;simple_neural_network_layer_block_9/StatefulPartitionedCall�
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinputs%lstm_3_statefulpartitionedcall_args_1%lstm_3_statefulpartitionedcall_args_2%lstm_3_statefulpartitionedcall_args_3*
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_231717502 
lstm_3/StatefulPartitionedCall�
;simple_neural_network_layer_block_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6*
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
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_231725312=
;simple_neural_network_layer_block_9/StatefulPartitionedCall�
<simple_neural_network_layer_block_10/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_9/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6*
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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_231731422>
<simple_neural_network_layer_block_10/StatefulPartitionedCall�
<simple_neural_network_layer_block_11/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_10/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6*
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
GPU2*0J 8*k
ffRd
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_231736912>
<simple_neural_network_layer_block_11/StatefulPartitionedCall�
 dense_15/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_11/StatefulPartitionedCall:output:0'dense_15_statefulpartitionedcall_args_1'dense_15_statefulpartitionedcall_args_2*
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
F__inference_dense_15_layer_call_and_return_conditional_losses_231724942"
 dense_15/StatefulPartitionedCall�
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall=^simple_neural_network_layer_block_10/StatefulPartitionedCall=^simple_neural_network_layer_block_11/StatefulPartitionedCall<^simple_neural_network_layer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2|
<simple_neural_network_layer_block_10/StatefulPartitionedCall<simple_neural_network_layer_block_10/StatefulPartitionedCall2|
<simple_neural_network_layer_block_11/StatefulPartitionedCall<simple_neural_network_layer_block_11/StatefulPartitionedCall2z
;simple_neural_network_layer_block_9/StatefulPartitionedCall;simple_neural_network_layer_block_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
F__inference_dense_15_layer_call_and_return_conditional_losses_23171784

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
�
�
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23171475

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
(__inference_while_cond_12836244_23172770
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12836244___redundant_placeholder00
,while_cond_12836244___redundant_placeholder10
,while_cond_12836244___redundant_placeholder20
,while_cond_12836244___redundant_placeholder3
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
)__inference_lstm_3_layer_call_fn_23175143

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
D__inference_lstm_3_layer_call_and_return_conditional_losses_231751352
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
�
�
(__inference_while_cond_12836705_23172251
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12836705___redundant_placeholder00
,while_cond_12836705___redundant_placeholder10
,while_cond_12836705___redundant_placeholder20
,while_cond_12836705___redundant_placeholder3
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
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23171505

inputs+
'dense_14_matmul_readvariableop_resource,
(dense_14_biasadd_readvariableop_resource7
3batch_normalization_11_cast_readvariableop_resource9
5batch_normalization_11_cast_1_readvariableop_resource9
5batch_normalization_11_cast_2_readvariableop_resource9
5batch_normalization_11_cast_3_readvariableop_resource
identity��*batch_normalization_11/Cast/ReadVariableOp�,batch_normalization_11/Cast_1/ReadVariableOp�,batch_normalization_11/Cast_2/ReadVariableOp�,batch_normalization_11/Cast_3/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp�
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_14/MatMul/ReadVariableOp�
dense_14/MatMulMatMulinputs&dense_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/MatMul�
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_14/BiasAdd/ReadVariableOp�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_14/BiasAdd�
#batch_normalization_11/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_11/LogicalAnd/x�
#batch_normalization_11/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_11/LogicalAnd/y�
!batch_normalization_11/LogicalAnd
LogicalAnd,batch_normalization_11/LogicalAnd/x:output:0,batch_normalization_11/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_11/LogicalAnd�
*batch_normalization_11/Cast/ReadVariableOpReadVariableOp3batch_normalization_11_cast_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*batch_normalization_11/Cast/ReadVariableOp�
,batch_normalization_11/Cast_1/ReadVariableOpReadVariableOp5batch_normalization_11_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_1/ReadVariableOp�
,batch_normalization_11/Cast_2/ReadVariableOpReadVariableOp5batch_normalization_11_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_2/ReadVariableOp�
,batch_normalization_11/Cast_3/ReadVariableOpReadVariableOp5batch_normalization_11_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,batch_normalization_11/Cast_3/ReadVariableOp�
&batch_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2(
&batch_normalization_11/batchnorm/add/y�
$batch_normalization_11/batchnorm/addAddV24batch_normalization_11/Cast_1/ReadVariableOp:value:0/batch_normalization_11/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/add�
&batch_normalization_11/batchnorm/RsqrtRsqrt(batch_normalization_11/batchnorm/add:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/Rsqrt�
$batch_normalization_11/batchnorm/mulMul*batch_normalization_11/batchnorm/Rsqrt:y:04batch_normalization_11/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/mul�
&batch_normalization_11/batchnorm/mul_1Muldense_14/BiasAdd:output:0(batch_normalization_11/batchnorm/mul:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/mul_1�
&batch_normalization_11/batchnorm/mul_2Mul2batch_normalization_11/Cast/ReadVariableOp:value:0(batch_normalization_11/batchnorm/mul:z:0*
T0*
_output_shapes	
:�2(
&batch_normalization_11/batchnorm/mul_2�
$batch_normalization_11/batchnorm/subSub4batch_normalization_11/Cast_2/ReadVariableOp:value:0*batch_normalization_11/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�2&
$batch_normalization_11/batchnorm/sub�
&batch_normalization_11/batchnorm/add_1AddV2*batch_normalization_11/batchnorm/mul_1:z:0(batch_normalization_11/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������2(
&batch_normalization_11/batchnorm/add_1s
TanhTanh*batch_normalization_11/batchnorm/add_1:z:0*
T0*(
_output_shapes
:����������2
Tanh�
IdentityIdentityTanh:y:0+^batch_normalization_11/Cast/ReadVariableOp-^batch_normalization_11/Cast_1/ReadVariableOp-^batch_normalization_11/Cast_2/ReadVariableOp-^batch_normalization_11/Cast_3/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:����������::::::2X
*batch_normalization_11/Cast/ReadVariableOp*batch_normalization_11/Cast/ReadVariableOp2\
,batch_normalization_11/Cast_1/ReadVariableOp,batch_normalization_11/Cast_1/ReadVariableOp2\
,batch_normalization_11/Cast_2/ReadVariableOp,batch_normalization_11/Cast_2/ReadVariableOp2\
,batch_normalization_11/Cast_3/ReadVariableOp,batch_normalization_11/Cast_3/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
(__inference_while_cond_12829793_23172089
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12829793___redundant_placeholder00
,while_cond_12829793___redundant_placeholder10
,while_cond_12829793___redundant_placeholder20
,while_cond_12829793___redundant_placeholder3
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
�
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23173989

inputs
assignmovingavg_12838826
assignmovingavg_1_12838832 
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
loc:@AssignMovingAvg/12838826*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_12838826*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12838826*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12838826*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_12838826AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/12838826*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/12838832*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_12838832*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12838832*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12838832*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_12838832AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/12838832*
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
�M
�
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23174230

inputs)
%lstm_3_statefulpartitionedcall_args_3)
%lstm_3_statefulpartitionedcall_args_4)
%lstm_3_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5F
Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5G
Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity��dense_15/BiasAdd/ReadVariableOp�dense_15/MatMul/ReadVariableOp�lstm_3/StatefulPartitionedCall�<simple_neural_network_layer_block_10/StatefulPartitionedCall�<simple_neural_network_layer_block_11/StatefulPartitionedCall�;simple_neural_network_layer_block_9/StatefulPartitionedCallR
lstm_3/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_3/Shape�
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack�
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1�
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2�
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicek
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros/mul/y�
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros/Less/y�
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessq
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros/packed/1�
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const�
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*(
_output_shapes
:����������2
lstm_3/zeroso
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros_1/mul/y�
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros_1/Less/y�
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lessu
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :�2
lstm_3/zeros_1/packed/1�
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const�
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*(
_output_shapes
:����������2
lstm_3/zeros_1�
lstm_3/StatefulPartitionedCallStatefulPartitionedCallinputslstm_3/zeros:output:0lstm_3/zeros_1:output:0%lstm_3_statefulpartitionedcall_args_3%lstm_3_statefulpartitionedcall_args_4%lstm_3_statefulpartitionedcall_args_5*
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
"__inference_standard_lstm_231741692 
lstm_3/StatefulPartitionedCall�
;simple_neural_network_layer_block_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_1Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_2Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_3Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_4Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_5Bsimple_neural_network_layer_block_9_statefulpartitionedcall_args_6*
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
__inference_call_231726762=
;simple_neural_network_layer_block_9/StatefulPartitionedCall�
<simple_neural_network_layer_block_10/StatefulPartitionedCallStatefulPartitionedCallDsimple_neural_network_layer_block_9/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_10_statefulpartitionedcall_args_6*
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
__inference_call_231740332>
<simple_neural_network_layer_block_10/StatefulPartitionedCall�
<simple_neural_network_layer_block_11/StatefulPartitionedCallStatefulPartitionedCallEsimple_neural_network_layer_block_10/StatefulPartitionedCall:output:0Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_1Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_2Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_3Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_4Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_5Csimple_neural_network_layer_block_11_statefulpartitionedcall_args_6*
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
__inference_call_231721332>
<simple_neural_network_layer_block_11/StatefulPartitionedCall�
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_15/MatMul/ReadVariableOp�
dense_15/MatMulMatMulEsimple_neural_network_layer_block_11/StatefulPartitionedCall:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_15/MatMul�
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp�
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_15/BiasAdds
dense_15/TanhTanhdense_15/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_15/Tanh�
IdentityIdentitydense_15/Tanh:y:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp^lstm_3/StatefulPartitionedCall=^simple_neural_network_layer_block_10/StatefulPartitionedCall=^simple_neural_network_layer_block_11/StatefulPartitionedCall<^simple_neural_network_layer_block_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesu
s:���������6:::::::::::::::::::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2|
<simple_neural_network_layer_block_10/StatefulPartitionedCall<simple_neural_network_layer_block_10/StatefulPartitionedCall2|
<simple_neural_network_layer_block_11/StatefulPartitionedCall<simple_neural_network_layer_block_11/StatefulPartitionedCall2z
;simple_neural_network_layer_block_9/StatefulPartitionedCall;simple_neural_network_layer_block_9/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�.
�
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_23174458

inputs
assignmovingavg_12830706
assignmovingavg_1_12830712 
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
loc:@AssignMovingAvg/12830706*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_12830706*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12830706*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12830706*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_12830706AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/12830706*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/12830712*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_12830712*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12830712*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12830712*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_12830712AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/12830712*
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
�
�
8__inference_batch_normalization_9_layer_call_fn_23174361

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
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_231743522
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
�
�
D__inference_lstm_3_layer_call_and_return_conditional_losses_23172003
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
"__inference_standard_lstm_231719702
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
�
�
+__inference_restored_function_body_23175965

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
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_231738222
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
�
�
 __forward_standard_lstm_23772717

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
2__inference_while_body_12831578_23171642_rewritten*>
cond6R4
2__inference_while_cond_12831577_23171520_rewritten*m
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
backward_function_name86__inference___backward_standard_lstm_23772271_2377271820
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
�
�
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23173340

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
�
�
9__inference_batch_normalization_10_layer_call_fn_23173349

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
GPU2*0J 8*]
fXRV
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_231733402
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
�
2__inference_while_cond_12831577_23171520_rewritten
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice0
,while_cond_12831577___redundant_placeholder00
,while_cond_12831577___redundant_placeholder10
,while_cond_12831577___redundant_placeholder20
,while_cond_12831577___redundant_placeholder3
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
�.
�
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23172288

inputs
assignmovingavg_12838908
assignmovingavg_1_12838914 
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
loc:@AssignMovingAvg/12838908*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_12838908*
_output_shapes	
:�*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12838908*
_output_shapes	
:�2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/12838908*
_output_shapes	
:�2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_12838908AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/12838908*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/12838914*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_12838914*
_output_shapes	
:�*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12838914*
_output_shapes	
:�2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/12838914*
_output_shapes	
:�2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_12838914AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/12838914*
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�
input_layer
lstm_layers
hidden_layers
output_layer

signatures
trainable_variables
	variables
regularization_losses
		keras_api
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�
_tf_keras_model�{"class_name": "LSTMNeuralNetworkModel", "name": "lstm_neural_network_model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "LSTMNeuralNetworkModel"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 5, 54], "config": {"batch_input_shape": [null, 5, 54], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
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
trainable_variables
	variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 6, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
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
15
16"
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
#15
$16
%17
&18
'19
(20
21
22"
trackable_list_wrapper
 "
trackable_list_wrapper
�
)non_trainable_variables

*layers
trainable_variables
	variables
+metrics
,layer_regularization_losses
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
/trainable_variables
0	variables
1regularization_losses
2	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LSTM", "name": "lstm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 512, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 54], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
�
3dense_layer
4
norm_layer
5trainable_variables
6	variables
7regularization_losses
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "SimpleNeuralNetworkLayerBlock", "name": "simple_neural_network_layer_block_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "simple_neural_network_layer_block_9", "trainable": true, "dtype": "float32", "units": 512, "activation_func": "tanh", "kernel_initializer": "glorot_uniform"}}
�
9dense_layer
:
norm_layer
;trainable_variables
<	variables
=regularization_losses
>	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "SimpleNeuralNetworkLayerBlock", "name": "simple_neural_network_layer_block_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "simple_neural_network_layer_block_10", "trainable": true, "dtype": "float32", "units": 512, "activation_func": "tanh", "kernel_initializer": "glorot_uniform"}}
�
?dense_layer
@
norm_layer
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "SimpleNeuralNetworkLayerBlock", "name": "simple_neural_network_layer_block_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "simple_neural_network_layer_block_11", "trainable": true, "dtype": "float32", "units": 512, "activation_func": "tanh", "kernel_initializer": "glorot_uniform"}}
": 	�2dense_15/kernel
:2dense_15/bias
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
Enon_trainable_variables

Flayers
trainable_variables
	variables
Gmetrics
Hlayer_regularization_losses
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	6�2lstm_3/kernel
+:)
��2lstm_3/recurrent_kernel
:�2lstm_3/bias
#:!
��2dense_12/kernel
:�2dense_12/bias
*:(�2batch_normalization_9/gamma
):'�2batch_normalization_9/beta
#:!
��2dense_13/kernel
:�2dense_13/bias
+:)�2batch_normalization_10/gamma
*:(�2batch_normalization_10/beta
#:!
��2dense_14/kernel
:�2dense_14/bias
+:)�2batch_normalization_11/gamma
*:(�2batch_normalization_11/beta
2:0� (2!batch_normalization_9/moving_mean
6:4� (2%batch_normalization_9/moving_variance
3:1� (2"batch_normalization_10/moving_mean
7:5� (2&batch_normalization_10/moving_variance
3:1� (2"batch_normalization_11/moving_mean
7:5� (2&batch_normalization_11/moving_variance
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
 "
trackable_list_wrapper
�

kernel
recurrent_kernel
bias
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LSTMCell", "name": "lstm_cell_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell_3", "trainable": true, "dtype": "float32", "units": 512, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
Mnon_trainable_variables

Nlayers
/trainable_variables
0	variables
Ometrics
Player_regularization_losses
1regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

kernel
bias
Qtrainable_variables
R	variables
Sregularization_losses
T	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_12", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
�
Uaxis
	gamma
beta
#moving_mean
$moving_variance
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_9", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}}
<
0
1
2
3"
trackable_list_wrapper
J
0
1
2
3
#4
$5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Znon_trainable_variables

[layers
5trainable_variables
6	variables
\metrics
]layer_regularization_losses
7regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

kernel
bias
^trainable_variables
_	variables
`regularization_losses
a	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_13", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
�
baxis
	gamma
beta
%moving_mean
&moving_variance
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_10", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}}
<
0
1
2
3"
trackable_list_wrapper
J
0
1
2
3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
;trainable_variables
<	variables
imetrics
jlayer_regularization_losses
=regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

kernel
 bias
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_14", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
�
oaxis
	!gamma
"beta
'moving_mean
(moving_variance
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_11", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 512}}}}
<
0
 1
!2
"3"
trackable_list_wrapper
J
0
 1
!2
"3
'4
(5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
tnon_trainable_variables

ulayers
Atrainable_variables
B	variables
vmetrics
wlayer_regularization_losses
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
xnon_trainable_variables

ylayers
Itrainable_variables
J	variables
zmetrics
{layer_regularization_losses
Kregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
-0"
trackable_list_wrapper
 "
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
|non_trainable_variables

}layers
Qtrainable_variables
R	variables
~metrics
layer_regularization_losses
Sregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
#2
$3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
Vtrainable_variables
W	variables
�metrics
 �layer_regularization_losses
Xregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
�non_trainable_variables
�layers
^trainable_variables
_	variables
�metrics
 �layer_regularization_losses
`regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
%2
&3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
ctrainable_variables
d	variables
�metrics
 �layer_regularization_losses
eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
�non_trainable_variables
�layers
ktrainable_variables
l	variables
�metrics
 �layer_regularization_losses
mregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
<
!0
"1
'2
(3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
ptrainable_variables
q	variables
�metrics
 �layer_regularization_losses
rregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
�2�
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23175263
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23174230
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23173822
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23174606�
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
#__inference__wrapped_model_46361856�
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
>__inference_lstm_neural_network_model_3_layer_call_fn_23173751
>__inference_lstm_neural_network_model_3_layer_call_fn_23175231
>__inference_lstm_neural_network_model_3_layer_call_fn_23175203
>__inference_lstm_neural_network_model_3_layer_call_fn_23173779�
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
F__inference_dense_15_layer_call_and_return_conditional_losses_23171784�
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
+__inference_dense_15_layer_call_fn_23172501�
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
&__inference_signature_wrapper_46361885input_1
�2�
D__inference_lstm_3_layer_call_and_return_conditional_losses_23173112
D__inference_lstm_3_layer_call_and_return_conditional_losses_23172003
D__inference_lstm_3_layer_call_and_return_conditional_losses_23174959
D__inference_lstm_3_layer_call_and_return_conditional_losses_23174773�
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
)__inference_lstm_3_layer_call_fn_23172901
)__inference_lstm_3_layer_call_fn_23171758
)__inference_lstm_3_layer_call_fn_23175143
)__inference_lstm_3_layer_call_fn_23173317�
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
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23171564
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23174989�
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
F__inference_simple_neural_network_layer_block_9_layer_call_fn_23172542
F__inference_simple_neural_network_layer_block_9_layer_call_fn_23173617�
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
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23173661
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23174260�
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
G__inference_simple_neural_network_layer_block_10_layer_call_fn_23172435
G__inference_simple_neural_network_layer_block_10_layer_call_fn_23173153�
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
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23172380
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23171505�
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
G__inference_simple_neural_network_layer_block_11_layer_call_fn_23172956
G__inference_simple_neural_network_layer_block_11_layer_call_fn_23173790�
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
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_23174297
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_23175012�
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
8__inference_batch_normalization_9_layer_call_fn_23174467
8__inference_batch_normalization_9_layer_call_fn_23174361�
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
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23171475
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23173989�
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
9__inference_batch_normalization_10_layer_call_fn_23173349
9__inference_batch_normalization_10_layer_call_fn_23173952�
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
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23172288
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23172156�
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
9__inference_batch_normalization_11_layer_call_fn_23174079
9__inference_batch_normalization_11_layer_call_fn_23174329�
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
#__inference__wrapped_model_46361856�#$%& '("!4�1
*�'
%�"
input_1���������6
� "3�0
.
output_1"�
output_1����������
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23171475d%&4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_10_layer_call_and_return_conditional_losses_23173989d%&4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_10_layer_call_fn_23173349W%&4�1
*�'
!�
inputs����������
p 
� "������������
9__inference_batch_normalization_10_layer_call_fn_23173952W%&4�1
*�'
!�
inputs����������
p
� "������������
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23172156d'("!4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
T__inference_batch_normalization_11_layer_call_and_return_conditional_losses_23172288d'("!4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
9__inference_batch_normalization_11_layer_call_fn_23174079W'("!4�1
*�'
!�
inputs����������
p
� "������������
9__inference_batch_normalization_11_layer_call_fn_23174329W'("!4�1
*�'
!�
inputs����������
p 
� "������������
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_23174297d#$4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
S__inference_batch_normalization_9_layer_call_and_return_conditional_losses_23175012d#$4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
8__inference_batch_normalization_9_layer_call_fn_23174361W#$4�1
*�'
!�
inputs����������
p 
� "������������
8__inference_batch_normalization_9_layer_call_fn_23174467W#$4�1
*�'
!�
inputs����������
p
� "������������
F__inference_dense_15_layer_call_and_return_conditional_losses_23171784]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� 
+__inference_dense_15_layer_call_fn_23172501P0�-
&�#
!�
inputs����������
� "�����������
D__inference_lstm_3_layer_call_and_return_conditional_losses_23172003~O�L
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_23173112n?�<
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_23174773n?�<
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
D__inference_lstm_3_layer_call_and_return_conditional_losses_23174959~O�L
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
)__inference_lstm_3_layer_call_fn_23171758a?�<
5�2
$�!
inputs���������6

 
p 

 
� "������������
)__inference_lstm_3_layer_call_fn_23172901qO�L
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
)__inference_lstm_3_layer_call_fn_23173317qO�L
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
)__inference_lstm_3_layer_call_fn_23175143a?�<
5�2
$�!
inputs���������6

 
p

 
� "������������
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23173822z#$%& '("!8�5
.�+
%�"
input_1���������6
p 
� "%�"
�
0���������
� �
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23174230y#$%& '("!7�4
-�*
$�!
inputs���������6
p
� "%�"
�
0���������
� �
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23174606y#$%& '("!7�4
-�*
$�!
inputs���������6
p 
� "%�"
�
0���������
� �
Y__inference_lstm_neural_network_model_3_layer_call_and_return_conditional_losses_23175263z#$%& '("!8�5
.�+
%�"
input_1���������6
p
� "%�"
�
0���������
� �
>__inference_lstm_neural_network_model_3_layer_call_fn_23173751m#$%& '("!8�5
.�+
%�"
input_1���������6
p 
� "�����������
>__inference_lstm_neural_network_model_3_layer_call_fn_23173779l#$%& '("!7�4
-�*
$�!
inputs���������6
p 
� "�����������
>__inference_lstm_neural_network_model_3_layer_call_fn_23175203m#$%& '("!8�5
.�+
%�"
input_1���������6
p
� "�����������
>__inference_lstm_neural_network_model_3_layer_call_fn_23175231l#$%& '("!7�4
-�*
$�!
inputs���������6
p
� "�����������
&__inference_signature_wrapper_46361885�#$%& '("!?�<
� 
5�2
0
input_1%�"
input_1���������6"3�0
.
output_1"�
output_1����������
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23173661f%&4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
b__inference_simple_neural_network_layer_block_10_layer_call_and_return_conditional_losses_23174260f%&4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
G__inference_simple_neural_network_layer_block_10_layer_call_fn_23172435Y%&4�1
*�'
!�
inputs����������
p
� "������������
G__inference_simple_neural_network_layer_block_10_layer_call_fn_23173153Y%&4�1
*�'
!�
inputs����������
p 
� "������������
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23171505f '("!4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
b__inference_simple_neural_network_layer_block_11_layer_call_and_return_conditional_losses_23172380f '("!4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
G__inference_simple_neural_network_layer_block_11_layer_call_fn_23172956Y '("!4�1
*�'
!�
inputs����������
p
� "������������
G__inference_simple_neural_network_layer_block_11_layer_call_fn_23173790Y '("!4�1
*�'
!�
inputs����������
p 
� "������������
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23171564f#$4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
a__inference_simple_neural_network_layer_block_9_layer_call_and_return_conditional_losses_23174989f#$4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
F__inference_simple_neural_network_layer_block_9_layer_call_fn_23172542Y#$4�1
*�'
!�
inputs����������
p 
� "������������
F__inference_simple_neural_network_layer_block_9_layer_call_fn_23173617Y#$4�1
*�'
!�
inputs����������
p
� "�����������