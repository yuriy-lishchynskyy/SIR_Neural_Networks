ѩ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x?* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	x?*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:?*
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
??*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:?*
dtype0
|
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_17/kernel
u
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel* 
_output_shapes
:
??*
dtype0
s
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_17/bias
l
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes	
:?*
dtype0
|
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_18/kernel
u
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel* 
_output_shapes
:
??*
dtype0
s
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_18/bias
l
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes	
:?*
dtype0
{
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_19/kernel
t
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes
:	?*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Nadam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x?*(
shared_nameNadam/dense_15/kernel/m
?
+Nadam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_15/kernel/m*
_output_shapes
:	x?*
dtype0
?
Nadam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_15/bias/m
|
)Nadam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_15/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_16/kernel/m
?
+Nadam/dense_16/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_16/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_16/bias/m
|
)Nadam/dense_16/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_16/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_17/kernel/m
?
+Nadam/dense_17/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_17/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_17/bias/m
|
)Nadam/dense_17/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_17/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_18/kernel/m
?
+Nadam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_18/kernel/m* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_18/bias/m
|
)Nadam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_18/bias/m*
_output_shapes	
:?*
dtype0
?
Nadam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameNadam/dense_19/kernel/m
?
+Nadam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_19/kernel/m*
_output_shapes
:	?*
dtype0
?
Nadam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_19/bias/m
{
)Nadam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_19/bias/m*
_output_shapes
:*
dtype0
?
Nadam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x?*(
shared_nameNadam/dense_15/kernel/v
?
+Nadam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_15/kernel/v*
_output_shapes
:	x?*
dtype0
?
Nadam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_15/bias/v
|
)Nadam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_15/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_16/kernel/v
?
+Nadam/dense_16/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_16/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_16/bias/v
|
)Nadam/dense_16/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_16/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_17/kernel/v
?
+Nadam/dense_17/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_17/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_17/bias/v
|
)Nadam/dense_17/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_17/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameNadam/dense_18/kernel/v
?
+Nadam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_18/kernel/v* 
_output_shapes
:
??*
dtype0
?
Nadam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameNadam/dense_18/bias/v
|
)Nadam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_18/bias/v*
_output_shapes	
:?*
dtype0
?
Nadam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameNadam/dense_19/kernel/v
?
+Nadam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_19/kernel/v*
_output_shapes
:	?*
dtype0
?
Nadam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameNadam/dense_19/bias/v
{
)Nadam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_19/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?7
value?7B?7 B?7
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
?
*iter

+beta_1

,beta_2
	-decay
.learning_rate
/momentum_cachemYmZm[m\m]m^m_m`$ma%mbvcvdvevfvgvhvivj$vk%vl
F
0
1
2
3
4
5
6
7
$8
%9
 
F
0
1
2
3
4
5
6
7
$8
%9
?
0non_trainable_variables
1metrics
2layer_metrics
3layer_regularization_losses
trainable_variables
regularization_losses
		variables

4layers
 
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
5non_trainable_variables
6metrics
7layer_metrics
8layer_regularization_losses
trainable_variables
regularization_losses
	variables

9layers
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
:non_trainable_variables
;metrics
<layer_metrics
=layer_regularization_losses
trainable_variables
regularization_losses
	variables

>layers
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
?non_trainable_variables
@metrics
Alayer_metrics
Blayer_regularization_losses
trainable_variables
regularization_losses
	variables

Clayers
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Dnon_trainable_variables
Emetrics
Flayer_metrics
Glayer_regularization_losses
 trainable_variables
!regularization_losses
"	variables

Hlayers
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
?
Inon_trainable_variables
Jmetrics
Klayer_metrics
Llayer_regularization_losses
&trainable_variables
'regularization_losses
(	variables

Mlayers
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1
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
 
 
 
 
 
4
	Ptotal
	Qcount
R	variables
S	keras_api
D
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

R	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

W	variables
}
VARIABLE_VALUENadam/dense_15/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_15/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_16/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_16/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_17/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_17/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_18/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_18/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_19/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_19/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_15/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_15/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_16/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_16/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_17/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_17/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_18/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_18/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/dense_19/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/dense_19/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_4Placeholder*'
_output_shapes
:?????????x*
dtype0*
shape:?????????x
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_836670
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Nadam/dense_15/kernel/m/Read/ReadVariableOp)Nadam/dense_15/bias/m/Read/ReadVariableOp+Nadam/dense_16/kernel/m/Read/ReadVariableOp)Nadam/dense_16/bias/m/Read/ReadVariableOp+Nadam/dense_17/kernel/m/Read/ReadVariableOp)Nadam/dense_17/bias/m/Read/ReadVariableOp+Nadam/dense_18/kernel/m/Read/ReadVariableOp)Nadam/dense_18/bias/m/Read/ReadVariableOp+Nadam/dense_19/kernel/m/Read/ReadVariableOp)Nadam/dense_19/bias/m/Read/ReadVariableOp+Nadam/dense_15/kernel/v/Read/ReadVariableOp)Nadam/dense_15/bias/v/Read/ReadVariableOp+Nadam/dense_16/kernel/v/Read/ReadVariableOp)Nadam/dense_16/bias/v/Read/ReadVariableOp+Nadam/dense_17/kernel/v/Read/ReadVariableOp)Nadam/dense_17/bias/v/Read/ReadVariableOp+Nadam/dense_18/kernel/v/Read/ReadVariableOp)Nadam/dense_18/bias/v/Read/ReadVariableOp+Nadam/dense_19/kernel/v/Read/ReadVariableOp)Nadam/dense_19/bias/v/Read/ReadVariableOpConst*5
Tin.
,2*	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_837041
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense_19/kerneldense_19/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1Nadam/dense_15/kernel/mNadam/dense_15/bias/mNadam/dense_16/kernel/mNadam/dense_16/bias/mNadam/dense_17/kernel/mNadam/dense_17/bias/mNadam/dense_18/kernel/mNadam/dense_18/bias/mNadam/dense_19/kernel/mNadam/dense_19/bias/mNadam/dense_15/kernel/vNadam/dense_15/bias/vNadam/dense_16/kernel/vNadam/dense_16/bias/vNadam/dense_17/kernel/vNadam/dense_17/bias/vNadam/dense_18/kernel/vNadam/dense_18/bias/vNadam/dense_19/kernel/vNadam/dense_19/bias/v*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_837171??
?
?
-__inference_sequential_3_layer_call_fn_836798

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8366122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
~
)__inference_dense_19_layer_call_fn_836898

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8364802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_18_layer_call_and_return_conditional_losses_836869

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_15_layer_call_and_return_conditional_losses_836809

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_836670
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_8363572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????x
!
_user_specified_name	input_4
?T
?
__inference__traced_save_837041
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_nadam_dense_15_kernel_m_read_readvariableop4
0savev2_nadam_dense_15_bias_m_read_readvariableop6
2savev2_nadam_dense_16_kernel_m_read_readvariableop4
0savev2_nadam_dense_16_bias_m_read_readvariableop6
2savev2_nadam_dense_17_kernel_m_read_readvariableop4
0savev2_nadam_dense_17_bias_m_read_readvariableop6
2savev2_nadam_dense_18_kernel_m_read_readvariableop4
0savev2_nadam_dense_18_bias_m_read_readvariableop6
2savev2_nadam_dense_19_kernel_m_read_readvariableop4
0savev2_nadam_dense_19_bias_m_read_readvariableop6
2savev2_nadam_dense_15_kernel_v_read_readvariableop4
0savev2_nadam_dense_15_bias_v_read_readvariableop6
2savev2_nadam_dense_16_kernel_v_read_readvariableop4
0savev2_nadam_dense_16_bias_v_read_readvariableop6
2savev2_nadam_dense_17_kernel_v_read_readvariableop4
0savev2_nadam_dense_17_bias_v_read_readvariableop6
2savev2_nadam_dense_18_kernel_v_read_readvariableop4
0savev2_nadam_dense_18_bias_v_read_readvariableop6
2savev2_nadam_dense_19_kernel_v_read_readvariableop4
0savev2_nadam_dense_19_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*?
value?B?)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_nadam_dense_15_kernel_m_read_readvariableop0savev2_nadam_dense_15_bias_m_read_readvariableop2savev2_nadam_dense_16_kernel_m_read_readvariableop0savev2_nadam_dense_16_bias_m_read_readvariableop2savev2_nadam_dense_17_kernel_m_read_readvariableop0savev2_nadam_dense_17_bias_m_read_readvariableop2savev2_nadam_dense_18_kernel_m_read_readvariableop0savev2_nadam_dense_18_bias_m_read_readvariableop2savev2_nadam_dense_19_kernel_m_read_readvariableop0savev2_nadam_dense_19_bias_m_read_readvariableop2savev2_nadam_dense_15_kernel_v_read_readvariableop0savev2_nadam_dense_15_bias_v_read_readvariableop2savev2_nadam_dense_16_kernel_v_read_readvariableop0savev2_nadam_dense_16_bias_v_read_readvariableop2savev2_nadam_dense_17_kernel_v_read_readvariableop0savev2_nadam_dense_17_bias_v_read_readvariableop2savev2_nadam_dense_18_kernel_v_read_readvariableop0savev2_nadam_dense_18_bias_v_read_readvariableop2savev2_nadam_dense_19_kernel_v_read_readvariableop0savev2_nadam_dense_19_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	x?:?:
??:?:
??:?:
??:?:	?:: : : : : : : : : : :	x?:?:
??:?:
??:?:
??:?:	?::	x?:?:
??:?:
??:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	x?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%	!

_output_shapes
:	?: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	x?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	x?:! 

_output_shapes	
:?:&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:&#"
 
_output_shapes
:
??:!$

_output_shapes	
:?:&%"
 
_output_shapes
:
??:!&

_output_shapes	
:?:%'!

_output_shapes
:	?: (

_output_shapes
::)

_output_shapes
: 
?0
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836709

inputs+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource
identity??dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulinputs&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_15/Relu?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_16/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/BiasAddt
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_17/Relu?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMuldense_17/Relu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_18/Relu?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdd|
dense_19/SigmoidSigmoiddense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Sigmoid?
IdentityIdentitydense_19/Sigmoid:y:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_836635
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8366122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????x
!
_user_specified_name	input_4
?	
?
D__inference_dense_19_layer_call_and_return_conditional_losses_836889

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836497
input_4
dense_15_836383
dense_15_836385
dense_16_836410
dense_16_836412
dense_17_836437
dense_17_836439
dense_18_836464
dense_18_836466
dense_19_836491
dense_19_836493
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_15_836383dense_15_836385*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_8363722"
 dense_15/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_836410dense_16_836412*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_8363992"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_836437dense_17_836439*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_8364262"
 dense_17/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_836464dense_18_836466*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8364532"
 dense_18/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_836491dense_19_836493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8364802"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:P L
'
_output_shapes
:?????????x
!
_user_specified_name	input_4
?0
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836748

inputs+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource
identity??dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?dense_18/BiasAdd/ReadVariableOp?dense_18/MatMul/ReadVariableOp?dense_19/BiasAdd/ReadVariableOp?dense_19/MatMul/ReadVariableOp?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulinputs&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_15/Relu?
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_16/MatMul/ReadVariableOp?
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/MatMul?
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_16/BiasAdd/ReadVariableOp?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_16/Relu?
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_17/MatMul/ReadVariableOp?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/MatMul?
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_17/BiasAdd/ReadVariableOp?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_17/BiasAddt
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_17/Relu?
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_18/MatMul/ReadVariableOp?
dense_18/MatMulMatMuldense_17/Relu:activations:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/MatMul?
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_18/BiasAdd/ReadVariableOp?
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_18/BiasAddt
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_18/Relu?
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_19/MatMul/ReadVariableOp?
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/MatMul?
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_19/BiasAdd/ReadVariableOp?
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_19/BiasAdd|
dense_19/SigmoidSigmoiddense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_19/Sigmoid?
IdentityIdentitydense_19/Sigmoid:y:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_836829

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_16_layer_call_and_return_conditional_losses_836399

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836612

inputs
dense_15_836586
dense_15_836588
dense_16_836591
dense_16_836593
dense_17_836596
dense_17_836598
dense_18_836601
dense_18_836603
dense_19_836606
dense_19_836608
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_836586dense_15_836588*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_8363722"
 dense_15/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_836591dense_16_836593*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_8363992"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_836596dense_17_836598*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_8364262"
 dense_17/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_836601dense_18_836603*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8364532"
 dense_18/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_836606dense_19_836608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8364802"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836558

inputs
dense_15_836532
dense_15_836534
dense_16_836537
dense_16_836539
dense_17_836542
dense_17_836544
dense_18_836547
dense_18_836549
dense_19_836552
dense_19_836554
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_836532dense_15_836534*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_8363722"
 dense_15/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_836537dense_16_836539*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_8363992"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_836542dense_17_836544*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_8364262"
 dense_17/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_836547dense_18_836549*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8364532"
 dense_18/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_836552dense_19_836554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8364802"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_836849

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_15_layer_call_and_return_conditional_losses_836372

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836526
input_4
dense_15_836500
dense_15_836502
dense_16_836505
dense_16_836507
dense_17_836510
dense_17_836512
dense_18_836515
dense_18_836517
dense_19_836520
dense_19_836522
identity?? dense_15/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall? dense_18/StatefulPartitionedCall? dense_19/StatefulPartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_15_836500dense_15_836502*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_8363722"
 dense_15/StatefulPartitionedCall?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_836505dense_16_836507*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_8363992"
 dense_16/StatefulPartitionedCall?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_836510dense_17_836512*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_8364262"
 dense_17/StatefulPartitionedCall?
 dense_18/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0dense_18_836515dense_18_836517*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8364532"
 dense_18/StatefulPartitionedCall?
 dense_19/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0dense_19_836520dense_19_836522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_8364802"
 dense_19/StatefulPartitionedCall?
IdentityIdentity)dense_19/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall:P L
'
_output_shapes
:?????????x
!
_user_specified_name	input_4
?	
?
D__inference_dense_18_layer_call_and_return_conditional_losses_836453

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_15_layer_call_fn_836818

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_8363722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_837171
file_prefix$
 assignvariableop_dense_15_kernel$
 assignvariableop_1_dense_15_bias&
"assignvariableop_2_dense_16_kernel$
 assignvariableop_3_dense_16_bias&
"assignvariableop_4_dense_17_kernel$
 assignvariableop_5_dense_17_bias&
"assignvariableop_6_dense_18_kernel$
 assignvariableop_7_dense_18_bias&
"assignvariableop_8_dense_19_kernel$
 assignvariableop_9_dense_19_bias"
assignvariableop_10_nadam_iter$
 assignvariableop_11_nadam_beta_1$
 assignvariableop_12_nadam_beta_2#
assignvariableop_13_nadam_decay+
'assignvariableop_14_nadam_learning_rate,
(assignvariableop_15_nadam_momentum_cache
assignvariableop_16_total
assignvariableop_17_count
assignvariableop_18_total_1
assignvariableop_19_count_1/
+assignvariableop_20_nadam_dense_15_kernel_m-
)assignvariableop_21_nadam_dense_15_bias_m/
+assignvariableop_22_nadam_dense_16_kernel_m-
)assignvariableop_23_nadam_dense_16_bias_m/
+assignvariableop_24_nadam_dense_17_kernel_m-
)assignvariableop_25_nadam_dense_17_bias_m/
+assignvariableop_26_nadam_dense_18_kernel_m-
)assignvariableop_27_nadam_dense_18_bias_m/
+assignvariableop_28_nadam_dense_19_kernel_m-
)assignvariableop_29_nadam_dense_19_bias_m/
+assignvariableop_30_nadam_dense_15_kernel_v-
)assignvariableop_31_nadam_dense_15_bias_v/
+assignvariableop_32_nadam_dense_16_kernel_v-
)assignvariableop_33_nadam_dense_16_bias_v/
+assignvariableop_34_nadam_dense_17_kernel_v-
)assignvariableop_35_nadam_dense_17_bias_v/
+assignvariableop_36_nadam_dense_18_kernel_v-
)assignvariableop_37_nadam_dense_18_bias_v/
+assignvariableop_38_nadam_dense_19_kernel_v-
)assignvariableop_39_nadam_dense_19_bias_v
identity_41??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*?
value?B?)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_18_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_18_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_19_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_19_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_nadam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_nadam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_nadam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_nadam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_nadam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp(assignvariableop_15_nadam_momentum_cacheIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_nadam_dense_15_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_nadam_dense_15_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_nadam_dense_16_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_nadam_dense_16_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_nadam_dense_17_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_nadam_dense_17_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_nadam_dense_18_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_nadam_dense_18_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_nadam_dense_19_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_nadam_dense_19_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_nadam_dense_15_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_nadam_dense_15_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_nadam_dense_16_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_nadam_dense_16_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_nadam_dense_17_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_nadam_dense_17_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_nadam_dense_18_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_nadam_dense_18_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp+assignvariableop_38_nadam_dense_19_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_nadam_dense_19_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_399
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_40?
Identity_41IdentityIdentity_40:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_41"#
identity_41Identity_41:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
-__inference_sequential_3_layer_call_fn_836773

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8365582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????x
 
_user_specified_nameinputs
?
~
)__inference_dense_18_layer_call_fn_836878

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_8364532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?=
?
!__inference__wrapped_model_836357
input_48
4sequential_3_dense_15_matmul_readvariableop_resource9
5sequential_3_dense_15_biasadd_readvariableop_resource8
4sequential_3_dense_16_matmul_readvariableop_resource9
5sequential_3_dense_16_biasadd_readvariableop_resource8
4sequential_3_dense_17_matmul_readvariableop_resource9
5sequential_3_dense_17_biasadd_readvariableop_resource8
4sequential_3_dense_18_matmul_readvariableop_resource9
5sequential_3_dense_18_biasadd_readvariableop_resource8
4sequential_3_dense_19_matmul_readvariableop_resource9
5sequential_3_dense_19_biasadd_readvariableop_resource
identity??,sequential_3/dense_15/BiasAdd/ReadVariableOp?+sequential_3/dense_15/MatMul/ReadVariableOp?,sequential_3/dense_16/BiasAdd/ReadVariableOp?+sequential_3/dense_16/MatMul/ReadVariableOp?,sequential_3/dense_17/BiasAdd/ReadVariableOp?+sequential_3/dense_17/MatMul/ReadVariableOp?,sequential_3/dense_18/BiasAdd/ReadVariableOp?+sequential_3/dense_18/MatMul/ReadVariableOp?,sequential_3/dense_19/BiasAdd/ReadVariableOp?+sequential_3/dense_19/MatMul/ReadVariableOp?
+sequential_3/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_15_matmul_readvariableop_resource*
_output_shapes
:	x?*
dtype02-
+sequential_3/dense_15/MatMul/ReadVariableOp?
sequential_3/dense_15/MatMulMatMulinput_43sequential_3/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_15/MatMul?
,sequential_3/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_3/dense_15/BiasAdd/ReadVariableOp?
sequential_3/dense_15/BiasAddBiasAdd&sequential_3/dense_15/MatMul:product:04sequential_3/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_15/BiasAdd?
sequential_3/dense_15/ReluRelu&sequential_3/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_15/Relu?
+sequential_3/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_3/dense_16/MatMul/ReadVariableOp?
sequential_3/dense_16/MatMulMatMul(sequential_3/dense_15/Relu:activations:03sequential_3/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_16/MatMul?
,sequential_3/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_3/dense_16/BiasAdd/ReadVariableOp?
sequential_3/dense_16/BiasAddBiasAdd&sequential_3/dense_16/MatMul:product:04sequential_3/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_16/BiasAdd?
sequential_3/dense_16/ReluRelu&sequential_3/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_16/Relu?
+sequential_3/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_3/dense_17/MatMul/ReadVariableOp?
sequential_3/dense_17/MatMulMatMul(sequential_3/dense_16/Relu:activations:03sequential_3/dense_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_17/MatMul?
,sequential_3/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_3/dense_17/BiasAdd/ReadVariableOp?
sequential_3/dense_17/BiasAddBiasAdd&sequential_3/dense_17/MatMul:product:04sequential_3/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_17/BiasAdd?
sequential_3/dense_17/ReluRelu&sequential_3/dense_17/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_17/Relu?
+sequential_3/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_18_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02-
+sequential_3/dense_18/MatMul/ReadVariableOp?
sequential_3/dense_18/MatMulMatMul(sequential_3/dense_17/Relu:activations:03sequential_3/dense_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_18/MatMul?
,sequential_3/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,sequential_3/dense_18/BiasAdd/ReadVariableOp?
sequential_3/dense_18/BiasAddBiasAdd&sequential_3/dense_18/MatMul:product:04sequential_3/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_18/BiasAdd?
sequential_3/dense_18/ReluRelu&sequential_3/dense_18/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_18/Relu?
+sequential_3/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_3_dense_19_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02-
+sequential_3/dense_19/MatMul/ReadVariableOp?
sequential_3/dense_19/MatMulMatMul(sequential_3/dense_18/Relu:activations:03sequential_3/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_19/MatMul?
,sequential_3/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_dense_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_3/dense_19/BiasAdd/ReadVariableOp?
sequential_3/dense_19/BiasAddBiasAdd&sequential_3/dense_19/MatMul:product:04sequential_3/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_19/BiasAdd?
sequential_3/dense_19/SigmoidSigmoid&sequential_3/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_3/dense_19/Sigmoid?
IdentityIdentity!sequential_3/dense_19/Sigmoid:y:0-^sequential_3/dense_15/BiasAdd/ReadVariableOp,^sequential_3/dense_15/MatMul/ReadVariableOp-^sequential_3/dense_16/BiasAdd/ReadVariableOp,^sequential_3/dense_16/MatMul/ReadVariableOp-^sequential_3/dense_17/BiasAdd/ReadVariableOp,^sequential_3/dense_17/MatMul/ReadVariableOp-^sequential_3/dense_18/BiasAdd/ReadVariableOp,^sequential_3/dense_18/MatMul/ReadVariableOp-^sequential_3/dense_19/BiasAdd/ReadVariableOp,^sequential_3/dense_19/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::2\
,sequential_3/dense_15/BiasAdd/ReadVariableOp,sequential_3/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_15/MatMul/ReadVariableOp+sequential_3/dense_15/MatMul/ReadVariableOp2\
,sequential_3/dense_16/BiasAdd/ReadVariableOp,sequential_3/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_16/MatMul/ReadVariableOp+sequential_3/dense_16/MatMul/ReadVariableOp2\
,sequential_3/dense_17/BiasAdd/ReadVariableOp,sequential_3/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_17/MatMul/ReadVariableOp+sequential_3/dense_17/MatMul/ReadVariableOp2\
,sequential_3/dense_18/BiasAdd/ReadVariableOp,sequential_3/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_18/MatMul/ReadVariableOp+sequential_3/dense_18/MatMul/ReadVariableOp2\
,sequential_3/dense_19/BiasAdd/ReadVariableOp,sequential_3/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_3/dense_19/MatMul/ReadVariableOp+sequential_3/dense_19/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????x
!
_user_specified_name	input_4
?
~
)__inference_dense_17_layer_call_fn_836858

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_8364262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_dense_16_layer_call_fn_836838

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_8363992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_836426

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_19_layer_call_and_return_conditional_losses_836480

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_836581
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_8365582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????x::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????x
!
_user_specified_name	input_4"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_40
serving_default_input_4:0?????????x<
dense_190
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?0
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
m_default_save_signature
n__call__
*o&call_and_return_all_conditional_losses"?-
_tf_keras_sequential?,{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 120]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 120]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mean_absolute_percentage_error", "dtype": "float32", "fn": "mean_absolute_percentage_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
p__call__
*q&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 120]}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
v__call__
*w&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 2, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
*iter

+beta_1

,beta_2
	-decay
.learning_rate
/momentum_cachemYmZm[m\m]m^m_m`$ma%mbvcvdvevfvgvhvivj$vk%vl"
	optimizer
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
?
0non_trainable_variables
1metrics
2layer_metrics
3layer_regularization_losses
trainable_variables
regularization_losses
		variables

4layers
n__call__
m_default_save_signature
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
,
zserving_default"
signature_map
": 	x?2dense_15/kernel
:?2dense_15/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
5non_trainable_variables
6metrics
7layer_metrics
8layer_regularization_losses
trainable_variables
regularization_losses
	variables

9layers
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_16/kernel
:?2dense_16/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
:non_trainable_variables
;metrics
<layer_metrics
=layer_regularization_losses
trainable_variables
regularization_losses
	variables

>layers
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_17/kernel
:?2dense_17/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?non_trainable_variables
@metrics
Alayer_metrics
Blayer_regularization_losses
trainable_variables
regularization_losses
	variables

Clayers
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_18/kernel
:?2dense_18/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Dnon_trainable_variables
Emetrics
Flayer_metrics
Glayer_regularization_losses
 trainable_variables
!regularization_losses
"	variables

Hlayers
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_19/kernel
:2dense_19/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
Inon_trainable_variables
Jmetrics
Klayer_metrics
Llayer_regularization_losses
&trainable_variables
'regularization_losses
(	variables

Mlayers
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
?
	Ptotal
	Qcount
R	variables
S	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mean_absolute_percentage_error", "dtype": "float32", "config": {"name": "mean_absolute_percentage_error", "dtype": "float32", "fn": "mean_absolute_percentage_error"}}
:  (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
(:&	x?2Nadam/dense_15/kernel/m
": ?2Nadam/dense_15/bias/m
):'
??2Nadam/dense_16/kernel/m
": ?2Nadam/dense_16/bias/m
):'
??2Nadam/dense_17/kernel/m
": ?2Nadam/dense_17/bias/m
):'
??2Nadam/dense_18/kernel/m
": ?2Nadam/dense_18/bias/m
(:&	?2Nadam/dense_19/kernel/m
!:2Nadam/dense_19/bias/m
(:&	x?2Nadam/dense_15/kernel/v
": ?2Nadam/dense_15/bias/v
):'
??2Nadam/dense_16/kernel/v
": ?2Nadam/dense_16/bias/v
):'
??2Nadam/dense_17/kernel/v
": ?2Nadam/dense_17/bias/v
):'
??2Nadam/dense_18/kernel/v
": ?2Nadam/dense_18/bias/v
(:&	?2Nadam/dense_19/kernel/v
!:2Nadam/dense_19/bias/v
?2?
!__inference__wrapped_model_836357?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_4?????????x
?2?
-__inference_sequential_3_layer_call_fn_836635
-__inference_sequential_3_layer_call_fn_836581
-__inference_sequential_3_layer_call_fn_836773
-__inference_sequential_3_layer_call_fn_836798?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836497
H__inference_sequential_3_layer_call_and_return_conditional_losses_836709
H__inference_sequential_3_layer_call_and_return_conditional_losses_836526
H__inference_sequential_3_layer_call_and_return_conditional_losses_836748?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_15_layer_call_fn_836818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_15_layer_call_and_return_conditional_losses_836809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_16_layer_call_fn_836838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_16_layer_call_and_return_conditional_losses_836829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_17_layer_call_fn_836858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_17_layer_call_and_return_conditional_losses_836849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_18_layer_call_fn_836878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_18_layer_call_and_return_conditional_losses_836869?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_19_layer_call_fn_836898?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_19_layer_call_and_return_conditional_losses_836889?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_836670input_4"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_836357s
$%0?-
&?#
!?
input_4?????????x
? "3?0
.
dense_19"?
dense_19??????????
D__inference_dense_15_layer_call_and_return_conditional_losses_836809]/?,
%?"
 ?
inputs?????????x
? "&?#
?
0??????????
? }
)__inference_dense_15_layer_call_fn_836818P/?,
%?"
 ?
inputs?????????x
? "????????????
D__inference_dense_16_layer_call_and_return_conditional_losses_836829^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_16_layer_call_fn_836838Q0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_17_layer_call_and_return_conditional_losses_836849^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_17_layer_call_fn_836858Q0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_18_layer_call_and_return_conditional_losses_836869^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
)__inference_dense_18_layer_call_fn_836878Q0?-
&?#
!?
inputs??????????
? "????????????
D__inference_dense_19_layer_call_and_return_conditional_losses_836889]$%0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_dense_19_layer_call_fn_836898P$%0?-
&?#
!?
inputs??????????
? "???????????
H__inference_sequential_3_layer_call_and_return_conditional_losses_836497m
$%8?5
.?+
!?
input_4?????????x
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836526m
$%8?5
.?+
!?
input_4?????????x
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836709l
$%7?4
-?*
 ?
inputs?????????x
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_836748l
$%7?4
-?*
 ?
inputs?????????x
p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_3_layer_call_fn_836581`
$%8?5
.?+
!?
input_4?????????x
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_836635`
$%8?5
.?+
!?
input_4?????????x
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_836773_
$%7?4
-?*
 ?
inputs?????????x
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_836798_
$%7?4
-?*
 ?
inputs?????????x
p 

 
? "???????????
$__inference_signature_wrapper_836670~
$%;?8
? 
1?.
,
input_4!?
input_4?????????x"3?0
.
dense_19"?
dense_19?????????