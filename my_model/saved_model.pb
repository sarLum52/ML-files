¢;
Ô©
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018¦â7
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
d*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
z
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
d*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
z
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
h
StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
StateVar
a
StateVar/Read/ReadVariableOpReadVariableOpStateVar*
_output_shapes
:*
dtype0	
l

StateVar_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
StateVar_1
e
StateVar_1/Read/ReadVariableOpReadVariableOp
StateVar_1*
_output_shapes
:*
dtype0	
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
d*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
Õ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bø
°
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
ª
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*

!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses* 
È
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias
 /_jit_compiled_convolution_op*

0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
È
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op*

?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
È
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op*

N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses* 
¥
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator* 
È
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias
 c_jit_compiled_convolution_op*

d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 
¥
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p_random_generator* 

q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses* 
¦
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias*
­
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
\
-0
.1
<2
=3
K4
L5
a6
b7
}8
~9
10
11*
\
-0
.1
<2
=3
K4
L5
a6
b7
}8
~9
10
11*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
½
	iter
beta_1
beta_2

decay
learning_rate-m¼.m½<m¾=m¿KmÀLmÁamÂbmÃ}mÄ~mÅ	mÆ	mÇ-vÈ.vÉ<vÊ=vËKvÌLvÍavÎbvÏ}vÐ~vÑ	vÒ	vÓ*

serving_default* 
®
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
 _random_generator*
®
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses
§_random_generator*
* 
* 
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
:
­trace_0
®trace_1
¯trace_2
°trace_3* 
:
±trace_0
²trace_1
³trace_2
´trace_3* 
* 
* 
* 

µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

ºtrace_0* 

»trace_0* 

-0
.1*

-0
.1*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

Átrace_0* 

Âtrace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

Ètrace_0* 

Étrace_0* 

<0
=1*

<0
=1*
* 

Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

Ïtrace_0* 

Ðtrace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

Ötrace_0* 

×trace_0* 

K0
L1*

K0
L1*
* 

Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

Ýtrace_0* 

Þtrace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

ätrace_0* 

åtrace_0* 
* 
* 
* 

ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

ëtrace_0
ìtrace_1* 

ítrace_0
îtrace_1* 
* 

a0
b1*

a0
b1*
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

ôtrace_0* 

õtrace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

ûtrace_0* 

ütrace_0* 
* 
* 
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

}0
~1*

}0
~1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

trace_0* 

trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

¢trace_0
£trace_1* 

¤trace_0
¥trace_1* 

¦
_generator*
* 
* 
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 

¬trace_0
­trace_1* 

®trace_0
¯trace_1* 

°
_generator*
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
±	variables
²	keras_api

³total

´count*
M
µ	variables
¶	keras_api

·total

¸count
¹
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 

º
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

»
_state_var*

³0
´1*

±	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

·0
¸1*

µ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
vp
VARIABLE_VALUE
StateVar_1Rlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEStateVarRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

 serving_default_sequential_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ  

StatefulPartitionedCallStatefulPartitionedCall serving_default_sequential_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_23889
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
´
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpStateVar_1/Read/ReadVariableOpStateVar/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*<
Tin5
321			*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_26925
	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount
StateVar_1StateVarAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*;
Tin4
220*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_27076ñå5
¿g
Ë
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_26376
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counter¥
 loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsV
Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderX
Tloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0³
®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0·
²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0u
qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0S
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityU
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1U
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2U
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice±
¬loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2µ
°loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1s
oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Jloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/addAddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderUloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdereloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
: 
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1PackNloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add:z:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:­
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
Tloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSlice®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0cloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
:  
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ï
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdergloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:¢
`loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ñ
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1PackPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0iloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¯
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSlice²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÑ
_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0]loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Î
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDimshloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes
:Æ
kloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemTloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderZloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ì
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: Î
Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: ¡
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1Identity loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ð
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2IdentityPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: û
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identity{loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "«
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityXloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"ä
oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shapeqloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0"
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_algloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0"²
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0"è
°loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0"à
¬loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Æ
^
B__inference_flatten_layer_call_and_return_conditional_losses_25881

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 2  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
ú

,__inference_sequential_1_layer_call_fn_23951

inputs
unknown:	
	unknown_0:	#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@$
	unknown_7:@
	unknown_8:	
	unknown_9:
d

unknown_10:	

unknown_11:	

unknown_12:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_23698o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

©
J__inference_random_contrast_layer_call_and_return_conditional_losses_23227

inputsI
;loop_body_stateful_uniform_full_int_rngreadandskip_resource:	
identity¢2loop_body/stateful_uniform_full_int/RngReadAndSkip¢=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
Rank/packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:W
	Max/inputPackstrided_slice:output:0*
N*
T0*
_output_shapes
:O
MaxMaxMax/input:output:0range:output:0*
T0*
_output_shapes
: h
&loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : 
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_sliceStridedSliceloop_body/Shape:output:0&loop_body/strided_slice/stack:output:0(loop_body/strided_slice/stack_1:output:0(loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :}
loop_body/GreaterGreater loop_body/strided_slice:output:0loop_body/Greater/y:output:0*
T0*
_output_shapes
: V
loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B :  
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:  s
)loop_body/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:s
)loop_body/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¹
(loop_body/stateful_uniform_full_int/ProdProd2loop_body/stateful_uniform_full_int/shape:output:02loop_body/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: l
*loop_body/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*loop_body/stateful_uniform_full_int/Cast_1Cast1loop_body/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2loop_body/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip;loop_body_stateful_uniform_full_int_rngreadandskip_resource3loop_body/stateful_uniform_full_int/Cast/x:output:0.loop_body/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
7loop_body/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9loop_body/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/stateful_uniform_full_int/strided_sliceStridedSlice:loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0@loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask£
+loop_body/stateful_uniform_full_int/BitcastBitcast:loop_body/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9loop_body/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3loop_body/stateful_uniform_full_int/strided_slice_1StridedSlice:loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0Bloop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:§
-loop_body/stateful_uniform_full_int/Bitcast_1Bitcast<loop_body/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'loop_body/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :Ã
#loop_body/stateful_uniform_full_intStatelessRandomUniformFullIntV22loop_body/stateful_uniform_full_int/shape:output:06loop_body/stateful_uniform_full_int/Bitcast_1:output:04loop_body/stateful_uniform_full_int/Bitcast:output:00loop_body/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	^
loop_body/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 
loop_body/stackPack,loop_body/stateful_uniform_full_int:output:0loop_body/zeros_like:output:0*
N*
T0	*
_output_shapes

:p
loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
loop_body/strided_slice_1StridedSliceloop_body/stack:output:0(loop_body/strided_slice_1/stack:output:0*loop_body/strided_slice_1/stack_1:output:0*loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskk
(loop_body/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB k
&loop_body/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?k
&loop_body/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?¥
?loop_body/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"loop_body/strided_slice_1:output:0* 
_output_shapes
::
?loop_body/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
;loop_body/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV21loop_body/stateless_random_uniform/shape:output:0Eloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: °
&loop_body/stateless_random_uniform/subSub/loop_body/stateless_random_uniform/max:output:0/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: À
&loop_body/stateless_random_uniform/mulMulDloop_body/stateless_random_uniform/StatelessRandomUniformV2:output:0*loop_body/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ©
"loop_body/stateless_random_uniformAddV2*loop_body/stateless_random_uniform/mul:z:0/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
loop_body/adjust_contrastAdjustContrastv2loop_body/GatherV2:output:0&loop_body/stateless_random_uniform:z:0*$
_output_shapes
:  
"loop_body/adjust_contrast/IdentityIdentity"loop_body/adjust_contrast:output:0*
T0*$
_output_shapes
:  f
!loop_body/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C²
loop_body/clip_by_value/MinimumMinimum+loop_body/adjust_contrast/Identity:output:0*loop_body/clip_by_value/Minimum/y:output:0*
T0*$
_output_shapes
:  ^
loop_body/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
loop_body/clip_by_valueMaximum#loop_body/clip_by_value/Minimum:z:0"loop_body/clip_by_value/y:output:0*
T0*$
_output_shapes
:  \
pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
pfor/ReshapeReshapeMax:output:0pfor/Reshape/shape:output:0*
T0*
_output_shapes
:R
pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : R
pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :|

pfor/rangeRangepfor/range/start:output:0Max:output:0pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Kloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Mloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Mloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Tloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack:output:0Vloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1:output:0Vloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Sloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2TensorListReserve\loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Ploop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Jloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileWhileSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counter:output:0Yloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterations:output:0Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const:output:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2:handle:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0;loop_body_stateful_uniform_full_int_rngreadandskip_resource3loop_body/stateful_uniform_full_int/Cast/x:output:0.loop_body/stateful_uniform_full_int/Cast_1:y:03^loop_body/stateful_uniform_full_int/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *T
bodyLRJ
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_22612*T
condLRJ
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_22611*#
output_shapes
: : : : : : : : 
?loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ©
Xloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ´
Jloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:output:3aloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0
Floop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Bloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
=loop_body/stateful_uniform_full_int/strided_slice/pfor/concatConcatV2Oloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0:output:0@loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Kloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1ConcatV2Qloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Mloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Dloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2ConcatV2Qloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0Mloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:å
Cloop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSliceStridedSliceSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Floop_body/stateful_uniform_full_int/strided_slice/pfor/concat:output:0Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1:output:0Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Dloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Floop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Floop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Mloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack:output:0Oloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1:output:0Oloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÅ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2TensorListReserveUloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shape:output:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌx
6loop_body/stateful_uniform_full_int/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Iloop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Cloop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
6loop_body/stateful_uniform_full_int/Bitcast/pfor/whileStatelessWhileLloop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counter:output:0Rloop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterations:output:0?loop_body/stateful_uniform_full_int/Bitcast/pfor/Const:output:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2:handle:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0Lloop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *M
bodyERC
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_22677*M
condERC
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_22676*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ{
8loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¢
Qloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Cloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2TensorListConcatV2?loop_body/stateful_uniform_full_int/Bitcast/pfor/while:output:3Zloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shape:output:0Aloop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Hloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concatConcatV2Qloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Mloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Floop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
Aloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1ConcatV2Sloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Oloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Floop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
Aloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2ConcatV2Sloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0Oloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:í
Eloop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSliceStridedSliceSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Hloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat:output:0Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1:output:0Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Floop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Hloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Hloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Oloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack:output:0Qloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1:output:0Qloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿË
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2TensorListReserveWloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌz
8loop_body/stateful_uniform_full_int/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Kloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Eloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ®
8loop_body/stateful_uniform_full_int/Bitcast_1/pfor/whileStatelessWhileNloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counter:output:0Tloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterations:output:0Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const:output:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2:handle:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0Nloop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *O
bodyGRE
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_22744*O
condGRE
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_22743*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ}
:loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¤
Sloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
Eloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV2Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while:output:3\loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
<loop_body/stateful_uniform_full_int/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ü
6loop_body/stateful_uniform_full_int/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Eloop_body/stateful_uniform_full_int/pfor/strided_slice/stack:output:0Gloop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1:output:0Gloop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Dloop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ­
6loop_body/stateful_uniform_full_int/pfor/TensorArrayV2TensorListReserveMloop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shape:output:0?loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐp
.loop_body/stateful_uniform_full_int/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform_full_int/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ}
;loop_body/stateful_uniform_full_int/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ë
.loop_body/stateful_uniform_full_int/pfor/whileStatelessWhileDloop_body/stateful_uniform_full_int/pfor/while/loop_counter:output:0Jloop_body/stateful_uniform_full_int/pfor/while/maximum_iterations:output:07loop_body/stateful_uniform_full_int/pfor/Const:output:0?loop_body/stateful_uniform_full_int/pfor/TensorArrayV2:handle:0?loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2:tensor:0Lloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2:tensor:02loop_body/stateful_uniform_full_int/shape:output:00loop_body/stateful_uniform_full_int/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *E
body=R;
9loop_body_stateful_uniform_full_int_pfor_while_body_22801*E
cond=R;
9loop_body_stateful_uniform_full_int_pfor_while_cond_22800*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: s
0loop_body/stateful_uniform_full_int/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
Iloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿø
;loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2TensorListConcatV27loop_body/stateful_uniform_full_int/pfor/while:output:3Rloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shape:output:09loop_body/stateful_uniform_full_int/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0d
loop_body/stack/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/stack/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:f
$loop_body/stack/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :¹
loop_body/stack/pfor/ones_likeFill=loop_body/stack/pfor/ones_like/Shape/shape_as_tensor:output:0-loop_body/stack/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:u
"loop_body/stack/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¢
loop_body/stack/pfor/ReshapeReshape'loop_body/stack/pfor/ones_like:output:0+loop_body/stack/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:w
$loop_body/stack/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
loop_body/stack/pfor/Reshape_1Reshapepfor/Reshape:output:0-loop_body/stack/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:b
 loop_body/stack/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
loop_body/stack/pfor/concatConcatV2'loop_body/stack/pfor/Reshape_1:output:0%loop_body/stack/pfor/Reshape:output:0)loop_body/stack/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:e
#loop_body/stack/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : £
loop_body/stack/pfor/ExpandDims
ExpandDimsloop_body/zeros_like:output:0,loop_body/stack/pfor/ExpandDims/dim:output:0*
T0	*
_output_shapes

:£
loop_body/stack/pfor/TileTile(loop_body/stack/pfor/ExpandDims:output:0$loop_body/stack/pfor/concat:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
loop_body/stack/pfor/stackPackDloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2:tensor:0"loop_body/stack/pfor/Tile:output:0*
N*
T0	*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

axisx
.loop_body/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: l
*loop_body/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
%loop_body/strided_slice_1/pfor/concatConcatV27loop_body/strided_slice_1/pfor/concat/values_0:output:0(loop_body/strided_slice_1/stack:output:03loop_body/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:z
0loop_body/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: n
,loop_body/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
'loop_body/strided_slice_1/pfor/concat_1ConcatV29loop_body/strided_slice_1/pfor/concat_1/values_0:output:0*loop_body/strided_slice_1/stack_1:output:05loop_body/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:z
0loop_body/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:n
,loop_body/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
'loop_body/strided_slice_1/pfor/concat_2ConcatV29loop_body/strided_slice_1/pfor/concat_2/values_0:output:0*loop_body/strided_slice_1/stack_2:output:05loop_body/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:í
+loop_body/strided_slice_1/pfor/StridedSliceStridedSlice#loop_body/stack/pfor/stack:output:0.loop_body/strided_slice_1/pfor/concat:output:00loop_body/strided_slice_1/pfor/concat_1:output:00loop_body/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask¢
Xloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¤
Zloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¤
Zloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_sliceStridedSlicepfor/Reshape:output:0aloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack:output:0cloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1:output:0cloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask«
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2TensorListReserveiloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shape:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ­
bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Tloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1TensorListReservekloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shape:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Jloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¨
]loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
Jloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/whileStatelessWhile`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counter:output:0floop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterations:output:0Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2:handle:0]loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1:handle:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:04loop_body/strided_slice_1/pfor/StridedSlice:output:0*
T
	2	*
_lower_using_switch_merge(*
_num_original_outputs*3
_output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *a
bodyYRW
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_22901*a
condYRW
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_22900*2
output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ
Lloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¶
eloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   è
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2TensorListConcatV2Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:3nloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shape:output:0Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Lloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ¸
gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
Yloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1TensorListConcatV2Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:4ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shape:output:0Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Tloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlicepfor/Reshape:output:0]loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask§
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReserveeloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Floop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¤
Yloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Sloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Æ

Floop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhile\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0bloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const:output:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2:tensor:0bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1:tensor:01loop_body/stateless_random_uniform/shape:output:0Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *]
bodyURS
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_22970*]
condURS
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_22969*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 
Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ´
aloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÔ
Sloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while:output:3jloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0r
0loop_body/stateless_random_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :t
2loop_body/stateless_random_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : s
1loop_body/stateless_random_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ò
/loop_body/stateless_random_uniform/mul/pfor/addAddV2;loop_body/stateless_random_uniform/mul/pfor/Rank_1:output:0:loop_body/stateless_random_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: Ï
3loop_body/stateless_random_uniform/mul/pfor/MaximumMaximum3loop_body/stateless_random_uniform/mul/pfor/add:z:09loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: ½
1loop_body/stateless_random_uniform/mul/pfor/ShapeShape\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:Ë
/loop_body/stateless_random_uniform/mul/pfor/subSub7loop_body/stateless_random_uniform/mul/pfor/Maximum:z:09loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
9loop_body/stateless_random_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ü
3loop_body/stateless_random_uniform/mul/pfor/ReshapeReshape3loop_body/stateless_random_uniform/mul/pfor/sub:z:0Bloop_body/stateless_random_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
6loop_body/stateless_random_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ú
0loop_body/stateless_random_uniform/mul/pfor/TileTile?loop_body/stateless_random_uniform/mul/pfor/Tile/input:output:0<loop_body/stateless_random_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
?loop_body/stateless_random_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9loop_body/stateless_random_uniform/mul/pfor/strided_sliceStridedSlice:loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Hloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Cloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Cloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
;loop_body/stateless_random_uniform/mul/pfor/strided_slice_1StridedSlice:loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack:output:0Lloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1:output:0Lloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masky
7loop_body/stateless_random_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
2loop_body/stateless_random_uniform/mul/pfor/concatConcatV2Bloop_body/stateless_random_uniform/mul/pfor/strided_slice:output:09loop_body/stateless_random_uniform/mul/pfor/Tile:output:0Dloop_body/stateless_random_uniform/mul/pfor/strided_slice_1:output:0@loop_body/stateless_random_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
5loop_body/stateless_random_uniform/mul/pfor/Reshape_1Reshape\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0;loop_body/stateless_random_uniform/mul/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
/loop_body/stateless_random_uniform/mul/pfor/MulMul>loop_body/stateless_random_uniform/mul/pfor/Reshape_1:output:0*loop_body/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,loop_body/stateless_random_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :p
.loop_body/stateless_random_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : o
-loop_body/stateless_random_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Æ
+loop_body/stateless_random_uniform/pfor/addAddV27loop_body/stateless_random_uniform/pfor/Rank_1:output:06loop_body/stateless_random_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: Ã
/loop_body/stateless_random_uniform/pfor/MaximumMaximum/loop_body/stateless_random_uniform/pfor/add:z:05loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
-loop_body/stateless_random_uniform/pfor/ShapeShape3loop_body/stateless_random_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:¿
+loop_body/stateless_random_uniform/pfor/subSub3loop_body/stateless_random_uniform/pfor/Maximum:z:05loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
5loop_body/stateless_random_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ð
/loop_body/stateless_random_uniform/pfor/ReshapeReshape/loop_body/stateless_random_uniform/pfor/sub:z:0>loop_body/stateless_random_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:|
2loop_body/stateless_random_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Î
,loop_body/stateless_random_uniform/pfor/TileTile;loop_body/stateless_random_uniform/pfor/Tile/input:output:08loop_body/stateless_random_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: 
;loop_body/stateless_random_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=loop_body/stateless_random_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=loop_body/stateless_random_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5loop_body/stateless_random_uniform/pfor/strided_sliceStridedSlice6loop_body/stateless_random_uniform/pfor/Shape:output:0Dloop_body/stateless_random_uniform/pfor/strided_slice/stack:output:0Floop_body/stateless_random_uniform/pfor/strided_slice/stack_1:output:0Floop_body/stateless_random_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
=loop_body/stateless_random_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7loop_body/stateless_random_uniform/pfor/strided_slice_1StridedSlice6loop_body/stateless_random_uniform/pfor/Shape:output:0Floop_body/stateless_random_uniform/pfor/strided_slice_1/stack:output:0Hloop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1:output:0Hloop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
3loop_body/stateless_random_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
.loop_body/stateless_random_uniform/pfor/concatConcatV2>loop_body/stateless_random_uniform/pfor/strided_slice:output:05loop_body/stateless_random_uniform/pfor/Tile:output:0@loop_body/stateless_random_uniform/pfor/strided_slice_1:output:0<loop_body/stateless_random_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ø
1loop_body/stateless_random_uniform/pfor/Reshape_1Reshape3loop_body/stateless_random_uniform/mul/pfor/Mul:z:07loop_body/stateless_random_uniform/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
-loop_body/stateless_random_uniform/pfor/AddV2AddV2:loop_body/stateless_random_uniform/pfor/Reshape_1:output:0/loop_body/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/addAddV2%loop_body/SelectV2/pfor/Rank:output:0&loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :`
loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : a
loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: 
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: 
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/SelectV2/pfor/TileTile+loop_body/SelectV2/pfor/Tile/input:output:0(loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: u
+loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%loop_body/SelectV2/pfor/strided_sliceStridedSlice&loop_body/SelectV2/pfor/Shape:output:04loop_body/SelectV2/pfor/strided_slice/stack:output:06loop_body/SelectV2/pfor/strided_slice/stack_1:output:06loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
'loop_body/SelectV2/pfor/strided_slice_1StridedSlice&loop_body/SelectV2/pfor/Shape:output:06loop_body/SelectV2/pfor/strided_slice_1/stack:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : î
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  |
2loop_body/adjust_contrast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4loop_body/adjust_contrast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/adjust_contrast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ô
,loop_body/adjust_contrast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0;loop_body/adjust_contrast/pfor/strided_slice/stack:output:0=loop_body/adjust_contrast/pfor/strided_slice/stack_1:output:0=loop_body/adjust_contrast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
:loop_body/adjust_contrast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
,loop_body/adjust_contrast/pfor/TensorArrayV2TensorListReserveCloop_body/adjust_contrast/pfor/TensorArrayV2/element_shape:output:05loop_body/adjust_contrast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
$loop_body/adjust_contrast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
7loop_body/adjust_contrast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿs
1loop_body/adjust_contrast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ï
$loop_body/adjust_contrast/pfor/whileStatelessWhile:loop_body/adjust_contrast/pfor/while/loop_counter:output:0@loop_body/adjust_contrast/pfor/while/maximum_iterations:output:0-loop_body/adjust_contrast/pfor/Const:output:05loop_body/adjust_contrast/pfor/TensorArrayV2:handle:05loop_body/adjust_contrast/pfor/strided_slice:output:0)loop_body/GatherV2/pfor/GatherV2:output:01loop_body/stateless_random_uniform/pfor/AddV2:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *;
body3R1
/loop_body_adjust_contrast_pfor_while_body_23119*;
cond3R1
/loop_body_adjust_contrast_pfor_while_cond_23118*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿi
&loop_body/adjust_contrast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
?loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ           Ú
1loop_body/adjust_contrast/pfor/TensorListConcatV2TensorListConcatV2-loop_body/adjust_contrast/pfor/while:output:3Hloop_body/adjust_contrast/pfor/TensorListConcatV2/element_shape:output:0/loop_body/adjust_contrast/pfor/Const_1:output:0*@
_output_shapes.
,:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0´
0loop_body/adjust_contrast/Identity/pfor/IdentityIdentity:loop_body/adjust_contrast/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  k
)loop_body/clip_by_value/Minimum/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/clip_by_value/Minimum/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/clip_by_value/Minimum/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/clip_by_value/Minimum/pfor/addAddV24loop_body/clip_by_value/Minimum/pfor/Rank_1:output:03loop_body/clip_by_value/Minimum/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/clip_by_value/Minimum/pfor/MaximumMaximum,loop_body/clip_by_value/Minimum/pfor/add:z:02loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/clip_by_value/Minimum/pfor/ShapeShape9loop_body/adjust_contrast/Identity/pfor/Identity:output:0*
T0*
_output_shapes
:¶
(loop_body/clip_by_value/Minimum/pfor/subSub0loop_body/clip_by_value/Minimum/pfor/Maximum:z:02loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/clip_by_value/Minimum/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/clip_by_value/Minimum/pfor/ReshapeReshape,loop_body/clip_by_value/Minimum/pfor/sub:z:0;loop_body/clip_by_value/Minimum/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/clip_by_value/Minimum/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/clip_by_value/Minimum/pfor/TileTile8loop_body/clip_by_value/Minimum/pfor/Tile/input:output:05loop_body/clip_by_value/Minimum/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/clip_by_value/Minimum/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/clip_by_value/Minimum/pfor/strided_sliceStridedSlice3loop_body/clip_by_value/Minimum/pfor/Shape:output:0Aloop_body/clip_by_value/Minimum/pfor/strided_slice/stack:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/clip_by_value/Minimum/pfor/strided_slice_1StridedSlice3loop_body/clip_by_value/Minimum/pfor/Shape:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack:output:0Eloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1:output:0Eloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/clip_by_value/Minimum/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/clip_by_value/Minimum/pfor/concatConcatV2;loop_body/clip_by_value/Minimum/pfor/strided_slice:output:02loop_body/clip_by_value/Minimum/pfor/Tile:output:0=loop_body/clip_by_value/Minimum/pfor/strided_slice_1:output:09loop_body/clip_by_value/Minimum/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:æ
.loop_body/clip_by_value/Minimum/pfor/Reshape_1Reshape9loop_body/adjust_contrast/Identity/pfor/Identity:output:04loop_body/clip_by_value/Minimum/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Ø
,loop_body/clip_by_value/Minimum/pfor/MinimumMinimum7loop_body/clip_by_value/Minimum/pfor/Reshape_1:output:0*loop_body/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  c
!loop_body/clip_by_value/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :e
#loop_body/clip_by_value/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : d
"loop_body/clip_by_value/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :¥
 loop_body/clip_by_value/pfor/addAddV2,loop_body/clip_by_value/pfor/Rank_1:output:0+loop_body/clip_by_value/pfor/add/y:output:0*
T0*
_output_shapes
: ¢
$loop_body/clip_by_value/pfor/MaximumMaximum$loop_body/clip_by_value/pfor/add:z:0*loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: 
"loop_body/clip_by_value/pfor/ShapeShape0loop_body/clip_by_value/Minimum/pfor/Minimum:z:0*
T0*
_output_shapes
:
 loop_body/clip_by_value/pfor/subSub(loop_body/clip_by_value/pfor/Maximum:z:0*loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: t
*loop_body/clip_by_value/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¯
$loop_body/clip_by_value/pfor/ReshapeReshape$loop_body/clip_by_value/pfor/sub:z:03loop_body/clip_by_value/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'loop_body/clip_by_value/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:­
!loop_body/clip_by_value/pfor/TileTile0loop_body/clip_by_value/pfor/Tile/input:output:0-loop_body/clip_by_value/pfor/Reshape:output:0*
T0*
_output_shapes
: z
0loop_body/clip_by_value/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2loop_body/clip_by_value/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2loop_body/clip_by_value/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
*loop_body/clip_by_value/pfor/strided_sliceStridedSlice+loop_body/clip_by_value/pfor/Shape:output:09loop_body/clip_by_value/pfor/strided_slice/stack:output:0;loop_body/clip_by_value/pfor/strided_slice/stack_1:output:0;loop_body/clip_by_value/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2loop_body/clip_by_value/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/clip_by_value/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4loop_body/clip_by_value/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
,loop_body/clip_by_value/pfor/strided_slice_1StridedSlice+loop_body/clip_by_value/pfor/Shape:output:0;loop_body/clip_by_value/pfor/strided_slice_1/stack:output:0=loop_body/clip_by_value/pfor/strided_slice_1/stack_1:output:0=loop_body/clip_by_value/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
(loop_body/clip_by_value/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¨
#loop_body/clip_by_value/pfor/concatConcatV23loop_body/clip_by_value/pfor/strided_slice:output:0*loop_body/clip_by_value/pfor/Tile:output:05loop_body/clip_by_value/pfor/strided_slice_1:output:01loop_body/clip_by_value/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Í
&loop_body/clip_by_value/pfor/Reshape_1Reshape0loop_body/clip_by_value/Minimum/pfor/Minimum:z:0,loop_body/clip_by_value/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Â
&loop_body/clip_by_value/pfor/Maximum_1Maximum/loop_body/clip_by_value/pfor/Reshape_1:output:0"loop_body/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
IdentityIdentity*loop_body/clip_by_value/pfor/Maximum_1:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  »
NoOpNoOp3^loop_body/stateful_uniform_full_int/RngReadAndSkip>^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : 2h
2loop_body/stateful_uniform_full_int/RngReadAndSkip2loop_body/stateful_uniform_full_int/RngReadAndSkip2~
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¸:
¸

Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_22744
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsH
Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderJ
Floop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1
{loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0	E
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityG
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1G
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2G
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3}
yloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice	
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :õ
<loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/addAddV2Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderGloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stackPackDloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderWloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Ploop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1Pack@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add:z:0Yloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
Floop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0Uloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack:output:0Wloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1:output:0Wloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÍ
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/BitcastBitcastOloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Gloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims
ExpandDimsIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Bitcast:output:0Ploop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
]loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderLloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ù
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1AddV2Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :³
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2AddV2~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counterIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ²
Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityBloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: ÷
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: ´
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2IdentityBloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: ß
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3Identitymloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityJloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3:output:0"ø
yloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice{loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0" 
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedsliceloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
µ

csequential_random_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_24089Ã
¾sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counterÉ
Äsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsh
dsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderj
fsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1Ã
¾sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_sliceÚ
Õsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_24089___redundant_placeholder0Ú
Õsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_24089___redundant_placeholder1Ú
Õsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_24089___redundant_placeholder2e
asequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity
­
]sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/LessLessdsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder¾sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: ñ
asequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityasequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "Ï
asequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityjsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
õ
`
B__inference_dropout_layer_call_and_return_conditional_losses_23406

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª
ú
E__inference_sequential_layer_call_and_return_conditional_losses_23249

inputs#
random_contrast_23242:	
random_zoom_23245:	
identity¢'random_contrast/StatefulPartitionedCall¢#random_zoom/StatefulPartitionedCall÷
'random_contrast/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_contrast_23242*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_23227
#random_zoom/StatefulPartitionedCallStatefulPartitionedCall0random_contrast/StatefulPartitionedCall:output:0random_zoom_23245*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_22517
IdentityIdentity,random_zoom/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
NoOpNoOp(^random_contrast/StatefulPartitionedCall$^random_zoom/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  : : 2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall2J
#random_zoom/StatefulPartitionedCall#random_zoom/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¶
K
/__inference_max_pooling2d_1_layer_call_fn_25751

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23302
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_25858

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
°

a
B__inference_dropout_layer_call_and_return_conditional_losses_23581

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æë
Æ
E__inference_sequential_layer_call_and_return_conditional_losses_25683

inputsY
Krandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	B
4random_zoom_stateful_uniform_rngreadandskip_resource:	
identity¢Brandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip¢Mrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while¢+random_zoom/stateful_uniform/RngReadAndSkipK
random_contrast/ShapeShapeinputs*
T0*
_output_shapes
:m
#random_contrast/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%random_contrast/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%random_contrast/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
random_contrast/strided_sliceStridedSlicerandom_contrast/Shape:output:0,random_contrast/strided_slice/stack:output:0.random_contrast/strided_slice/stack_1:output:0.random_contrast/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
random_contrast/Rank/packedPack&random_contrast/strided_slice:output:0*
N*
T0*
_output_shapes
:V
random_contrast/RankConst*
_output_shapes
: *
dtype0*
value	B :]
random_contrast/range/startConst*
_output_shapes
: *
dtype0*
value	B : ]
random_contrast/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¥
random_contrast/rangeRange$random_contrast/range/start:output:0random_contrast/Rank:output:0$random_contrast/range/delta:output:0*
_output_shapes
:w
random_contrast/Max/inputPack&random_contrast/strided_slice:output:0*
N*
T0*
_output_shapes
:
random_contrast/MaxMax"random_contrast/Max/input:output:0random_contrast/range:output:0*
T0*
_output_shapes
: x
6random_contrast/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ½
0random_contrast/loop_body/PlaceholderWithDefaultPlaceholderWithDefault?random_contrast/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: U
random_contrast/loop_body/ShapeShapeinputs*
T0*
_output_shapes
:w
-random_contrast/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/random_contrast/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/random_contrast/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'random_contrast/loop_body/strided_sliceStridedSlice(random_contrast/loop_body/Shape:output:06random_contrast/loop_body/strided_slice/stack:output:08random_contrast/loop_body/strided_slice/stack_1:output:08random_contrast/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#random_contrast/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :­
!random_contrast/loop_body/GreaterGreater0random_contrast/loop_body/strided_slice:output:0,random_contrast/loop_body/Greater/y:output:0*
T0*
_output_shapes
: f
$random_contrast/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : à
"random_contrast/loop_body/SelectV2SelectV2%random_contrast/loop_body/Greater:z:09random_contrast/loop_body/PlaceholderWithDefault:output:0-random_contrast/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: i
'random_contrast/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
"random_contrast/loop_body/GatherV2GatherV2inputs+random_contrast/loop_body/SelectV2:output:00random_contrast/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:  
9random_contrast/loop_body/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
9random_contrast/loop_body/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: é
8random_contrast/loop_body/stateful_uniform_full_int/ProdProdBrandom_contrast/loop_body/stateful_uniform_full_int/shape:output:0Brandom_contrast/loop_body/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: |
:random_contrast/loop_body/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :µ
:random_contrast/loop_body/stateful_uniform_full_int/Cast_1CastArandom_contrast/loop_body/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Â
Brandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipKrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resourceCrandom_contrast/loop_body/stateful_uniform_full_int/Cast/x:output:0>random_contrast/loop_body/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Grandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Irandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Irandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
Arandom_contrast/loop_body/stateful_uniform_full_int/strided_sliceStridedSliceJrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0Prandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÃ
;random_contrast/loop_body/stateful_uniform_full_int/BitcastBitcastJrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Irandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Krandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Krandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
Crandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1StridedSliceJrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ç
=random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1BitcastLrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0y
7random_contrast/loop_body/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
3random_contrast/loop_body/stateful_uniform_full_intStatelessRandomUniformFullIntV2Brandom_contrast/loop_body/stateful_uniform_full_int/shape:output:0Frandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1:output:0Drandom_contrast/loop_body/stateful_uniform_full_int/Bitcast:output:0@random_contrast/loop_body/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	n
$random_contrast/loop_body/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R Æ
random_contrast/loop_body/stackPack<random_contrast/loop_body/stateful_uniform_full_int:output:0-random_contrast/loop_body/zeros_like:output:0*
N*
T0	*
_output_shapes

:
/random_contrast/loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1random_contrast/loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1random_contrast/loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)random_contrast/loop_body/strided_slice_1StridedSlice(random_contrast/loop_body/stack:output:08random_contrast/loop_body/strided_slice_1/stack:output:0:random_contrast/loop_body/strided_slice_1/stack_1:output:0:random_contrast/loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask{
8random_contrast/loop_body/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB {
6random_contrast/loop_body/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?{
6random_contrast/loop_body/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?Å
Orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter2random_contrast/loop_body/strided_slice_1:output:0* 
_output_shapes
::
Orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Î
Krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Arandom_contrast/loop_body/stateless_random_uniform/shape:output:0Urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Yrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Xrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: à
6random_contrast/loop_body/stateless_random_uniform/subSub?random_contrast/loop_body/stateless_random_uniform/max:output:0?random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ð
6random_contrast/loop_body/stateless_random_uniform/mulMulTrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2:output:0:random_contrast/loop_body/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: Ù
2random_contrast/loop_body/stateless_random_uniformAddV2:random_contrast/loop_body/stateless_random_uniform/mul:z:0?random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: È
)random_contrast/loop_body/adjust_contrastAdjustContrastv2+random_contrast/loop_body/GatherV2:output:06random_contrast/loop_body/stateless_random_uniform:z:0*$
_output_shapes
:  ¡
2random_contrast/loop_body/adjust_contrast/IdentityIdentity2random_contrast/loop_body/adjust_contrast:output:0*
T0*$
_output_shapes
:  v
1random_contrast/loop_body/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Câ
/random_contrast/loop_body/clip_by_value/MinimumMinimum;random_contrast/loop_body/adjust_contrast/Identity:output:0:random_contrast/loop_body/clip_by_value/Minimum/y:output:0*
T0*$
_output_shapes
:  n
)random_contrast/loop_body/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ê
'random_contrast/loop_body/clip_by_valueMaximum3random_contrast/loop_body/clip_by_value/Minimum:z:02random_contrast/loop_body/clip_by_value/y:output:0*
T0*$
_output_shapes
:  l
"random_contrast/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
random_contrast/pfor/ReshapeReshaperandom_contrast/Max:output:0+random_contrast/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:b
 random_contrast/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 random_contrast/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¼
random_contrast/pfor/rangeRange)random_contrast/pfor/range/start:output:0random_contrast/Max:output:0)random_contrast/pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
[random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: §
]random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:§
]random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Urandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0drandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack:output:0frandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1:output:0frandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
crandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Urandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2TensorListReservelrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
Mrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : «
`random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Zrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ü	
Mrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileWhilecrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counter:output:0irandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterations:output:0Vrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const:output:0^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2:handle:0^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0Krandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resourceCrandom_contrast/loop_body/stateful_uniform_full_int/Cast/x:output:0>random_contrast/loop_body/stateful_uniform_full_int/Cast_1:y:0C^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *d
body\RZ
Xrandom_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_24970*d
cond\RZ
Xrandom_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_24969*#
output_shapes
: : : : : : : : 
Orandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¹
hrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
Zrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2Vrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:output:3qrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0 
Vrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Mrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concatConcatV2_random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0:output:0Prandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack:output:0[random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Orandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1ConcatV2arandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0]random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Orandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2ConcatV2arandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0]random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:µ
Srandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSliceStridedSlicecrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Vrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat:output:0Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1:output:0Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Trandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0]random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack:output:0_random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1:output:0_random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask§
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2TensorListReserveerandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shape:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Frandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¤
Yrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¬
Frandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/whileStatelessWhile\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counter:output:0brandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterations:output:0Orandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2:handle:0Wrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0\random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *]
bodyURS
Qrandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_25035*]
condURS
Qrandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_25034*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ
Hrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ²
arandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ø
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2TensorListConcatV2Orandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while:output:3jrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shape:output:0Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Orandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concatConcatV2arandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0:output:0Rrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0]random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
Zrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Vrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Qrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1ConcatV2crandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0:output:0Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0_random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¤
Zrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Vrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Qrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2ConcatV2crandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0:output:0Trandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0_random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:½
Urandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSliceStridedSlicecrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Xrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat:output:0Zrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1:output:0Zrandom_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask 
Vrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¢
Xrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0_random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack:output:0arandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1:output:0arandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask©
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿû
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2TensorListReservegrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0Yrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Hrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¦
[random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Urandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
Hrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/whileStatelessWhile^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counter:output:0drandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterations:output:0Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const:output:0Yrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2:handle:0Yrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0^random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *_
bodyWRU
Srandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_25102*_
condWRU
Srandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_25101*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ
Jrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ´
crandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
Urandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV2Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while:output:3lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Lrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0Urandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÝ
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2TensorListReserve]random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shape:output:0Orandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
>random_contrast/loop_body/stateful_uniform_full_int/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Qrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Krandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 

>random_contrast/loop_body/stateful_uniform_full_int/pfor/whileStatelessWhileTrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/loop_counter:output:0Zrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/maximum_iterations:output:0Grandom_contrast/loop_body/stateful_uniform_full_int/pfor/Const:output:0Orandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2:handle:0Orandom_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2:tensor:0\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2:tensor:0Brandom_contrast/loop_body/stateful_uniform_full_int/shape:output:0@random_contrast/loop_body/stateful_uniform_full_int/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *U
bodyMRK
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_body_25159*U
condMRK
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_25158*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 
@random_contrast/loop_body/stateful_uniform_full_int/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ª
Yrandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¸
Krandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2TensorListConcatV2Grandom_contrast/loop_body/stateful_uniform_full_int/pfor/while:output:3brandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shape:output:0Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0t
*random_contrast/loop_body/stack/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
Drandom_contrast/loop_body/stack/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:v
4random_contrast/loop_body/stack/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :é
.random_contrast/loop_body/stack/pfor/ones_likeFillMrandom_contrast/loop_body/stack/pfor/ones_like/Shape/shape_as_tensor:output:0=random_contrast/loop_body/stack/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:
2random_contrast/loop_body/stack/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÒ
,random_contrast/loop_body/stack/pfor/ReshapeReshape7random_contrast/loop_body/stack/pfor/ones_like:output:0;random_contrast/loop_body/stack/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
4random_contrast/loop_body/stack/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÄ
.random_contrast/loop_body/stack/pfor/Reshape_1Reshape%random_contrast/pfor/Reshape:output:0=random_contrast/loop_body/stack/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:r
0random_contrast/loop_body/stack/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+random_contrast/loop_body/stack/pfor/concatConcatV27random_contrast/loop_body/stack/pfor/Reshape_1:output:05random_contrast/loop_body/stack/pfor/Reshape:output:09random_contrast/loop_body/stack/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:u
3random_contrast/loop_body/stack/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ó
/random_contrast/loop_body/stack/pfor/ExpandDims
ExpandDims-random_contrast/loop_body/zeros_like:output:0<random_contrast/loop_body/stack/pfor/ExpandDims/dim:output:0*
T0	*
_output_shapes

:Ó
)random_contrast/loop_body/stack/pfor/TileTile8random_contrast/loop_body/stack/pfor/ExpandDims:output:04random_contrast/loop_body/stack/pfor/concat:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*random_contrast/loop_body/stack/pfor/stackPackTrandom_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2:tensor:02random_contrast/loop_body/stack/pfor/Tile:output:0*
N*
T0	*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

axis
>random_contrast/loop_body/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:random_contrast/loop_body/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5random_contrast/loop_body/strided_slice_1/pfor/concatConcatV2Grandom_contrast/loop_body/strided_slice_1/pfor/concat/values_0:output:08random_contrast/loop_body/strided_slice_1/stack:output:0Crandom_contrast/loop_body/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@random_contrast/loop_body/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<random_contrast/loop_body/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7random_contrast/loop_body/strided_slice_1/pfor/concat_1ConcatV2Irandom_contrast/loop_body/strided_slice_1/pfor/concat_1/values_0:output:0:random_contrast/loop_body/strided_slice_1/stack_1:output:0Erandom_contrast/loop_body/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@random_contrast/loop_body/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<random_contrast/loop_body/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7random_contrast/loop_body/strided_slice_1/pfor/concat_2ConcatV2Irandom_contrast/loop_body/strided_slice_1/pfor/concat_2/values_0:output:0:random_contrast/loop_body/strided_slice_1/stack_2:output:0Erandom_contrast/loop_body/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:½
;random_contrast/loop_body/strided_slice_1/pfor/StridedSliceStridedSlice3random_contrast/loop_body/stack/pfor/stack:output:0>random_contrast/loop_body/strided_slice_1/pfor/concat:output:0@random_contrast/loop_body/strided_slice_1/pfor/concat_1:output:0@random_contrast/loop_body/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask²
hrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ´
jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:´
jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¼
brandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0qrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack:output:0srandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1:output:0srandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask»
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ±
brandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2TensorListReserveyrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shape:output:0krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ½
rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿµ
drandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1TensorListReserve{random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shape:output:0krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Zrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¸
mrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ©
grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¨

Zrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/whileStatelessWhileprandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counter:output:0vrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterations:output:0crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const:output:0krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2:handle:0mrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1:handle:0krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0Drandom_contrast/loop_body/strided_slice_1/pfor/StridedSlice:output:0*
T
	2	*
_lower_using_switch_merge(*
_num_original_outputs*3
_output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *q
bodyiRg
erandom_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_25259*q
condiRg
erandom_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_25258*2
output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 Æ
urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¨
grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2TensorListConcatV2crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:3~random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shape:output:0erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 È
wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ­
irandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1TensorListConcatV2crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:4random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shape:output:0erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0®
drandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: °
frandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:°
frandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0mrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask·
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¥
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReserveurandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Vrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ´
irandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¥
crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
Vrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhilelrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const:output:0grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2:tensor:0rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1:tensor:0Arandom_contrast/loop_body/stateless_random_uniform/shape:output:0Xrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *m
bodyeRc
arandom_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_25328*m
condeRc
arandom_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_25327*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 
Xrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 Ä
qrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while:output:3zrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
@random_contrast/loop_body/stateless_random_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
Brandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
Arandom_contrast/loop_body/stateless_random_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
?random_contrast/loop_body/stateless_random_uniform/mul/pfor/addAddV2Krandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank_1:output:0Jrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: ÿ
Crandom_contrast/loop_body/stateless_random_uniform/mul/pfor/MaximumMaximumCrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/add:z:0Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: Ý
Arandom_contrast/loop_body/stateless_random_uniform/mul/pfor/ShapeShapelrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:û
?random_contrast/loop_body/stateless_random_uniform/mul/pfor/subSubGrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Maximum:z:0Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Crandom_contrast/loop_body/stateless_random_uniform/mul/pfor/ReshapeReshapeCrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/sub:z:0Rrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Frandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
@random_contrast/loop_body/stateless_random_uniform/mul/pfor/TileTileOrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile/input:output:0Lrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Orandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Qrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Qrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_sliceStridedSliceJrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Xrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack:output:0Zrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1:output:0Zrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Qrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Srandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Srandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ÿ
Krandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1StridedSliceJrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Zrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack:output:0\random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1:output:0\random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask
Grandom_contrast/loop_body/stateless_random_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ã
Brandom_contrast/loop_body/stateless_random_uniform/mul/pfor/concatConcatV2Rrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice:output:0Irandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile:output:0Trandom_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1:output:0Prandom_contrast/loop_body/stateless_random_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¹
Erandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape_1Reshapelrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0Krandom_contrast/loop_body/stateless_random_uniform/mul/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?random_contrast/loop_body/stateless_random_uniform/mul/pfor/MulMulNrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape_1:output:0:random_contrast/loop_body/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
<random_contrast/loop_body/stateless_random_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
>random_contrast/loop_body/stateless_random_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
=random_contrast/loop_body/stateless_random_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ö
;random_contrast/loop_body/stateless_random_uniform/pfor/addAddV2Grandom_contrast/loop_body/stateless_random_uniform/pfor/Rank_1:output:0Frandom_contrast/loop_body/stateless_random_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: ó
?random_contrast/loop_body/stateless_random_uniform/pfor/MaximumMaximum?random_contrast/loop_body/stateless_random_uniform/pfor/add:z:0Erandom_contrast/loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: °
=random_contrast/loop_body/stateless_random_uniform/pfor/ShapeShapeCrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:ï
;random_contrast/loop_body/stateless_random_uniform/pfor/subSubCrandom_contrast/loop_body/stateless_random_uniform/pfor/Maximum:z:0Erandom_contrast/loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
Erandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?random_contrast/loop_body/stateless_random_uniform/pfor/ReshapeReshape?random_contrast/loop_body/stateless_random_uniform/pfor/sub:z:0Nrandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Brandom_contrast/loop_body/stateless_random_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:þ
<random_contrast/loop_body/stateless_random_uniform/pfor/TileTileKrandom_contrast/loop_body/stateless_random_uniform/pfor/Tile/input:output:0Hrandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Krandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Mrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Mrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
Erandom_contrast/loop_body/stateless_random_uniform/pfor/strided_sliceStridedSliceFrandom_contrast/loop_body/stateless_random_uniform/pfor/Shape:output:0Trandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack:output:0Vrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_1:output:0Vrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Mrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Orandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Orandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
Grandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1StridedSliceFrandom_contrast/loop_body/stateless_random_uniform/pfor/Shape:output:0Vrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack:output:0Xrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1:output:0Xrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask
Crandom_contrast/loop_body/stateless_random_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
>random_contrast/loop_body/stateless_random_uniform/pfor/concatConcatV2Nrandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice:output:0Erandom_contrast/loop_body/stateless_random_uniform/pfor/Tile:output:0Prandom_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1:output:0Lrandom_contrast/loop_body/stateless_random_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Arandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape_1ReshapeCrandom_contrast/loop_body/stateless_random_uniform/mul/pfor/Mul:z:0Grandom_contrast/loop_body/stateless_random_uniform/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=random_contrast/loop_body/stateless_random_uniform/pfor/AddV2AddV2Jrandom_contrast/loop_body/stateless_random_uniform/pfor/Reshape_1:output:0?random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,random_contrast/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : o
-random_contrast/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ä
+random_contrast/loop_body/SelectV2/pfor/addAddV25random_contrast/loop_body/SelectV2/pfor/Rank:output:06random_contrast/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: p
.random_contrast/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :p
.random_contrast/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : q
/random_contrast/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ê
-random_contrast/loop_body/SelectV2/pfor/add_1AddV27random_contrast/loop_body/SelectV2/pfor/Rank_2:output:08random_contrast/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Å
/random_contrast/loop_body/SelectV2/pfor/MaximumMaximum7random_contrast/loop_body/SelectV2/pfor/Rank_1:output:0/random_contrast/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Å
1random_contrast/loop_body/SelectV2/pfor/Maximum_1Maximum1random_contrast/loop_body/SelectV2/pfor/add_1:z:03random_contrast/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: 
-random_contrast/loop_body/SelectV2/pfor/ShapeShape#random_contrast/pfor/range:output:0*
T0*
_output_shapes
:Ã
+random_contrast/loop_body/SelectV2/pfor/subSub5random_contrast/loop_body/SelectV2/pfor/Maximum_1:z:07random_contrast/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: 
5random_contrast/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ð
/random_contrast/loop_body/SelectV2/pfor/ReshapeReshape/random_contrast/loop_body/SelectV2/pfor/sub:z:0>random_contrast/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:|
2random_contrast/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Î
,random_contrast/loop_body/SelectV2/pfor/TileTile;random_contrast/loop_body/SelectV2/pfor/Tile/input:output:08random_contrast/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
;random_contrast/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_contrast/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_contrast/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_contrast/loop_body/SelectV2/pfor/strided_sliceStridedSlice6random_contrast/loop_body/SelectV2/pfor/Shape:output:0Drandom_contrast/loop_body/SelectV2/pfor/strided_slice/stack:output:0Frandom_contrast/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Frandom_contrast/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
=random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7random_contrast/loop_body/SelectV2/pfor/strided_slice_1StridedSlice6random_contrast/loop_body/SelectV2/pfor/Shape:output:0Frandom_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Hrandom_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Hrandom_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
3random_contrast/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
.random_contrast/loop_body/SelectV2/pfor/concatConcatV2>random_contrast/loop_body/SelectV2/pfor/strided_slice:output:05random_contrast/loop_body/SelectV2/pfor/Tile:output:0@random_contrast/loop_body/SelectV2/pfor/strided_slice_1:output:0<random_contrast/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:È
1random_contrast/loop_body/SelectV2/pfor/Reshape_1Reshape#random_contrast/pfor/range:output:07random_contrast/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
0random_contrast/loop_body/SelectV2/pfor/SelectV2SelectV2%random_contrast/loop_body/Greater:z:0:random_contrast/loop_body/SelectV2/pfor/Reshape_1:output:0-random_contrast/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5random_contrast/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
0random_contrast/loop_body/GatherV2/pfor/GatherV2GatherV2inputs9random_contrast/loop_body/SelectV2/pfor/SelectV2:output:0>random_contrast/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Brandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Drandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Drandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
<random_contrast/loop_body/adjust_contrast/pfor/strided_sliceStridedSlice%random_contrast/pfor/Reshape:output:0Krandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack:output:0Mrandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_1:output:0Mrandom_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Jrandom_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¿
<random_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2TensorListReserveSrandom_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2/element_shape:output:0Erandom_contrast/loop_body/adjust_contrast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒv
4random_contrast/loop_body/adjust_contrast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Grandom_contrast/loop_body/adjust_contrast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Arandom_contrast/loop_body/adjust_contrast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ï
4random_contrast/loop_body/adjust_contrast/pfor/whileStatelessWhileJrandom_contrast/loop_body/adjust_contrast/pfor/while/loop_counter:output:0Prandom_contrast/loop_body/adjust_contrast/pfor/while/maximum_iterations:output:0=random_contrast/loop_body/adjust_contrast/pfor/Const:output:0Erandom_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2:handle:0Erandom_contrast/loop_body/adjust_contrast/pfor/strided_slice:output:09random_contrast/loop_body/GatherV2/pfor/GatherV2:output:0Arandom_contrast/loop_body/stateless_random_uniform/pfor/AddV2:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *K
bodyCRA
?random_contrast_loop_body_adjust_contrast_pfor_while_body_25477*K
condCRA
?random_contrast_loop_body_adjust_contrast_pfor_while_cond_25476*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿy
6random_contrast/loop_body/adjust_contrast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¨
Orandom_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ           
Arandom_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2TensorListConcatV2=random_contrast/loop_body/adjust_contrast/pfor/while:output:3Xrandom_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shape:output:0?random_contrast/loop_body/adjust_contrast/pfor/Const_1:output:0*@
_output_shapes.
,:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0Ô
@random_contrast/loop_body/adjust_contrast/Identity/pfor/IdentityIdentityJrandom_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  {
9random_contrast/loop_body/clip_by_value/Minimum/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :}
;random_contrast/loop_body/clip_by_value/Minimum/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : |
:random_contrast/loop_body/clip_by_value/Minimum/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :í
8random_contrast/loop_body/clip_by_value/Minimum/pfor/addAddV2Drandom_contrast/loop_body/clip_by_value/Minimum/pfor/Rank_1:output:0Crandom_contrast/loop_body/clip_by_value/Minimum/pfor/add/y:output:0*
T0*
_output_shapes
: ê
<random_contrast/loop_body/clip_by_value/Minimum/pfor/MaximumMaximum<random_contrast/loop_body/clip_by_value/Minimum/pfor/add:z:0Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: ³
:random_contrast/loop_body/clip_by_value/Minimum/pfor/ShapeShapeIrandom_contrast/loop_body/adjust_contrast/Identity/pfor/Identity:output:0*
T0*
_output_shapes
:æ
8random_contrast/loop_body/clip_by_value/Minimum/pfor/subSub@random_contrast/loop_body/clip_by_value/Minimum/pfor/Maximum:z:0Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: 
Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_contrast/loop_body/clip_by_value/Minimum/pfor/ReshapeReshape<random_contrast/loop_body/clip_by_value/Minimum/pfor/sub:z:0Krandom_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?random_contrast/loop_body/clip_by_value/Minimum/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_contrast/loop_body/clip_by_value/Minimum/pfor/TileTileHrandom_contrast/loop_body/clip_by_value/Minimum/pfor/Tile/input:output:0Erandom_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Hrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_sliceStridedSliceCrandom_contrast/loop_body/clip_by_value/Minimum/pfor/Shape:output:0Qrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack:output:0Srandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1:output:0Srandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
Drandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1StridedSliceCrandom_contrast/loop_body/clip_by_value/Minimum/pfor/Shape:output:0Srandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack:output:0Urandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1:output:0Urandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_contrast/loop_body/clip_by_value/Minimum/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
;random_contrast/loop_body/clip_by_value/Minimum/pfor/concatConcatV2Krandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice:output:0Brandom_contrast/loop_body/clip_by_value/Minimum/pfor/Tile:output:0Mrandom_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1:output:0Irandom_contrast/loop_body/clip_by_value/Minimum/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
>random_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape_1ReshapeIrandom_contrast/loop_body/adjust_contrast/Identity/pfor/Identity:output:0Drandom_contrast/loop_body/clip_by_value/Minimum/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
<random_contrast/loop_body/clip_by_value/Minimum/pfor/MinimumMinimumGrandom_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape_1:output:0:random_contrast/loop_body/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  s
1random_contrast/loop_body/clip_by_value/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :u
3random_contrast/loop_body/clip_by_value/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : t
2random_contrast/loop_body/clip_by_value/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Õ
0random_contrast/loop_body/clip_by_value/pfor/addAddV2<random_contrast/loop_body/clip_by_value/pfor/Rank_1:output:0;random_contrast/loop_body/clip_by_value/pfor/add/y:output:0*
T0*
_output_shapes
: Ò
4random_contrast/loop_body/clip_by_value/pfor/MaximumMaximum4random_contrast/loop_body/clip_by_value/pfor/add:z:0:random_contrast/loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: ¢
2random_contrast/loop_body/clip_by_value/pfor/ShapeShape@random_contrast/loop_body/clip_by_value/Minimum/pfor/Minimum:z:0*
T0*
_output_shapes
:Î
0random_contrast/loop_body/clip_by_value/pfor/subSub8random_contrast/loop_body/clip_by_value/pfor/Maximum:z:0:random_contrast/loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: 
:random_contrast/loop_body/clip_by_value/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ß
4random_contrast/loop_body/clip_by_value/pfor/ReshapeReshape4random_contrast/loop_body/clip_by_value/pfor/sub:z:0Crandom_contrast/loop_body/clip_by_value/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
7random_contrast/loop_body/clip_by_value/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ý
1random_contrast/loop_body/clip_by_value/pfor/TileTile@random_contrast/loop_body/clip_by_value/pfor/Tile/input:output:0=random_contrast/loop_body/clip_by_value/pfor/Reshape:output:0*
T0*
_output_shapes
: 
@random_contrast/loop_body/clip_by_value/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Brandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Brandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
:random_contrast/loop_body/clip_by_value/pfor/strided_sliceStridedSlice;random_contrast/loop_body/clip_by_value/pfor/Shape:output:0Irandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack:output:0Krandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_1:output:0Krandom_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Brandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Drandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Drandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
<random_contrast/loop_body/clip_by_value/pfor/strided_slice_1StridedSlice;random_contrast/loop_body/clip_by_value/pfor/Shape:output:0Krandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack:output:0Mrandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_1:output:0Mrandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskz
8random_contrast/loop_body/clip_by_value/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ø
3random_contrast/loop_body/clip_by_value/pfor/concatConcatV2Crandom_contrast/loop_body/clip_by_value/pfor/strided_slice:output:0:random_contrast/loop_body/clip_by_value/pfor/Tile:output:0Erandom_contrast/loop_body/clip_by_value/pfor/strided_slice_1:output:0Arandom_contrast/loop_body/clip_by_value/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ý
6random_contrast/loop_body/clip_by_value/pfor/Reshape_1Reshape@random_contrast/loop_body/clip_by_value/Minimum/pfor/Minimum:z:0<random_contrast/loop_body/clip_by_value/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ò
6random_contrast/loop_body/clip_by_value/pfor/Maximum_1Maximum?random_contrast/loop_body/clip_by_value/pfor/Reshape_1:output:02random_contrast/loop_body/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  {
random_zoom/ShapeShape:random_contrast/loop_body/clip_by_value/pfor/Maximum_1:z:0*
T0*
_output_shapes
:i
random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom/strided_sliceStridedSlicerandom_zoom/Shape:output:0(random_zoom/strided_slice/stack:output:0*random_zoom/strided_slice/stack_1:output:0*random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
!random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿv
#random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿm
#random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom/strided_slice_1StridedSlicerandom_zoom/Shape:output:0*random_zoom/strided_slice_1/stack:output:0,random_zoom/strided_slice_1/stack_1:output:0,random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
random_zoom/CastCast$random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: t
!random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿv
#random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿm
#random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom/strided_slice_2StridedSlicerandom_zoom/Shape:output:0*random_zoom/strided_slice_2/stack:output:0,random_zoom/strided_slice_2/stack_1:output:0,random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
random_zoom/Cast_1Cast$random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: f
$random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :«
"random_zoom/stateful_uniform/shapePack"random_zoom/strided_slice:output:0-random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:e
 random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?e
 random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?l
"random_zoom/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¤
!random_zoom/stateful_uniform/ProdProd+random_zoom/stateful_uniform/shape:output:0+random_zoom/stateful_uniform/Const:output:0*
T0*
_output_shapes
: e
#random_zoom/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
#random_zoom/stateful_uniform/Cast_1Cast*random_zoom/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: æ
+random_zoom/stateful_uniform/RngReadAndSkipRngReadAndSkip4random_zoom_stateful_uniform_rngreadandskip_resource,random_zoom/stateful_uniform/Cast/x:output:0'random_zoom/stateful_uniform/Cast_1:y:0*
_output_shapes
:z
0random_zoom/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2random_zoom/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2random_zoom/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:è
*random_zoom/stateful_uniform/strided_sliceStridedSlice3random_zoom/stateful_uniform/RngReadAndSkip:value:09random_zoom/stateful_uniform/strided_slice/stack:output:0;random_zoom/stateful_uniform/strided_slice/stack_1:output:0;random_zoom/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
$random_zoom/stateful_uniform/BitcastBitcast3random_zoom/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0|
2random_zoom/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4random_zoom/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4random_zoom/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
,random_zoom/stateful_uniform/strided_slice_1StridedSlice3random_zoom/stateful_uniform/RngReadAndSkip:value:0;random_zoom/stateful_uniform/strided_slice_1/stack:output:0=random_zoom/stateful_uniform/strided_slice_1/stack_1:output:0=random_zoom/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
&random_zoom/stateful_uniform/Bitcast_1Bitcast5random_zoom/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0{
9random_zoom/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ë
5random_zoom/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2+random_zoom/stateful_uniform/shape:output:0/random_zoom/stateful_uniform/Bitcast_1:output:0-random_zoom/stateful_uniform/Bitcast:output:0Brandom_zoom/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 random_zoom/stateful_uniform/subSub)random_zoom/stateful_uniform/max:output:0)random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: ¿
 random_zoom/stateful_uniform/mulMul>random_zoom/stateful_uniform/StatelessRandomUniformV2:output:0$random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
random_zoom/stateful_uniformAddV2$random_zoom/stateful_uniform/mul:z:0)random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¿
random_zoom/concatConcatV2 random_zoom/stateful_uniform:z:0 random_zoom/stateful_uniform:z:0 random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
random_zoom/zoom_matrix/ShapeShaperandom_zoom/concat:output:0*
T0*
_output_shapes
:u
+random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%random_zoom/zoom_matrix/strided_sliceStridedSlice&random_zoom/zoom_matrix/Shape:output:04random_zoom/zoom_matrix/strided_slice/stack:output:06random_zoom/zoom_matrix/strided_slice/stack_1:output:06random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
random_zoom/zoom_matrix/subSubrandom_zoom/Cast_1:y:0&random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: f
!random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
random_zoom/zoom_matrix/truedivRealDivrandom_zoom/zoom_matrix/sub:z:0*random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 
-random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
'random_zoom/zoom_matrix/strided_slice_1StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_1/stack:output:08random_zoom/zoom_matrix/strided_slice_1/stack_1:output:08random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskd
random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
random_zoom/zoom_matrix/sub_1Sub(random_zoom/zoom_matrix/sub_1/x:output:00random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
random_zoom/zoom_matrix/mulMul#random_zoom/zoom_matrix/truediv:z:0!random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
random_zoom/zoom_matrix/sub_2Subrandom_zoom/Cast:y:0(random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: h
#random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
!random_zoom/zoom_matrix/truediv_1RealDiv!random_zoom/zoom_matrix/sub_2:z:0,random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 
-random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
'random_zoom/zoom_matrix/strided_slice_2StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_2/stack:output:08random_zoom/zoom_matrix/strided_slice_2/stack_1:output:08random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskd
random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?²
random_zoom/zoom_matrix/sub_3Sub(random_zoom/zoom_matrix/sub_3/x:output:00random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
random_zoom/zoom_matrix/mul_1Mul%random_zoom/zoom_matrix/truediv_1:z:0!random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
'random_zoom/zoom_matrix/strided_slice_3StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_3/stack:output:08random_zoom/zoom_matrix/strided_slice_3/stack_1:output:08random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskh
&random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :»
$random_zoom/zoom_matrix/zeros/packedPack.random_zoom/zoom_matrix/strided_slice:output:0/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:h
#random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
random_zoom/zoom_matrix/zerosFill-random_zoom/zoom_matrix/zeros/packed:output:0,random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¿
&random_zoom/zoom_matrix/zeros_1/packedPack.random_zoom/zoom_matrix/strided_slice:output:01random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:j
%random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    º
random_zoom/zoom_matrix/zeros_1Fill/random_zoom/zoom_matrix/zeros_1/packed:output:0.random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
'random_zoom/zoom_matrix/strided_slice_4StridedSlicerandom_zoom/concat:output:06random_zoom/zoom_matrix/strided_slice_4/stack:output:08random_zoom/zoom_matrix/strided_slice_4/stack_1:output:08random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskj
(random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¿
&random_zoom/zoom_matrix/zeros_2/packedPack.random_zoom/zoom_matrix/strided_slice:output:01random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:j
%random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    º
random_zoom/zoom_matrix/zeros_2Fill/random_zoom/zoom_matrix/zeros_2/packed:output:0.random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :·
random_zoom/zoom_matrix/concatConcatV20random_zoom/zoom_matrix/strided_slice_3:output:0&random_zoom/zoom_matrix/zeros:output:0random_zoom/zoom_matrix/mul:z:0(random_zoom/zoom_matrix/zeros_1:output:00random_zoom/zoom_matrix/strided_slice_4:output:0!random_zoom/zoom_matrix/mul_1:z:0(random_zoom/zoom_matrix/zeros_2:output:0,random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
random_zoom/transform/ShapeShape:random_contrast/loop_body/clip_by_value/pfor/Maximum_1:z:0*
T0*
_output_shapes
:s
)random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
#random_zoom/transform/strided_sliceStridedSlice$random_zoom/transform/Shape:output:02random_zoom/transform/strided_slice/stack:output:04random_zoom/transform/strided_slice/stack_1:output:04random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:e
 random_zoom/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
0random_zoom/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3:random_contrast/loop_body/clip_by_value/pfor/Maximum_1:z:0'random_zoom/zoom_matrix/concat:output:0,random_zoom/transform/strided_slice:output:0)random_zoom/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentityErandom_zoom/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
NoOpNoOpC^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipN^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while,^random_zoom/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  : : 2
Brandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipBrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip2
Mrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileMrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while2Z
+random_zoom/stateful_uniform/RngReadAndSkip+random_zoom/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_25776

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ(( : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( 
 
_user_specified_nameinputs

f
J__inference_random_contrast_layer_call_and_return_conditional_losses_22395

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ð
Õ
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_22743
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsH
Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderJ
Floop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_cond_22743___redundant_placeholder0	E
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity
¬
=loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/LessLessDloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: ±
Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityAloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityJloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:


,__inference_sequential_1_layer_call_fn_23762
sequential_input
unknown:	
	unknown_0:	#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@$
	unknown_7:@
	unknown_8:	
	unknown_9:
d

unknown_10:	

unknown_11:	

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_23698o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
*
_user_specified_namesequential_input

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_25786

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
{
+__inference_random_zoom_layer_call_fn_26645

inputs
unknown:	
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_22517y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Á

%__inference_dense_layer_call_fn_25890

inputs
unknown:
d
	unknown_0:	
identity¢StatefulPartitionedCallÖ
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
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23452p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ë
ÿ
Qrandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_25034
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter¥
 random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsV
Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderX
Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice¶
±random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_cond_25034___redundant_placeholder0	S
Orandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity
å
Krandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/LessLessRrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice*
T0*
_output_shapes
: Í
Orandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentityOrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "«
Orandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityXrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

f
J__inference_random_contrast_layer_call_and_return_conditional_losses_25936

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
£

ô
@__inference_dense_layer_call_and_return_conditional_losses_25901

inputs2
matmul_readvariableop_resource:
d.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ÛF
Ì
Xrandom_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_24970­
¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter³
®random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterations]
Yrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_
[random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1ª
¥random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0ª
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0:	
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0Z
Vrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity\
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1\
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2\
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3¨
£random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¨
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_x
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1¢\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipÂ
\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkiprandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0*
_output_shapes
:
\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ü
Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsdrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0erandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:â
rrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem[random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1Yrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderarandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Srandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :´
Qrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/addAddV2Yrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Urandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Srandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1AddV2¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ±
Vrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityWrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1:z:0S^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1Identity®random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsS^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ±
Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2IdentityUrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add:z:0S^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ß
Xrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3Identityrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0S^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ó
Rrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOpNoOp]^random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "¹
Vrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0"½
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1arandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1:output:0"½
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2arandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2:output:0"½
Xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3arandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3:output:0"
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0"
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_xrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0"Î
£random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¥random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0"º
random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resourcerandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2¼
\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip\random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ñ9
Ì
G__inference_sequential_1_layer_call_and_return_conditional_losses_23475

inputs&
conv2d_23359:
conv2d_23361:(
conv2d_1_23377: 
conv2d_1_23379: (
conv2d_2_23395: @
conv2d_2_23397:@)
conv2d_3_23420:@
conv2d_3_23422:	
dense_23453:
d
dense_23455:	 
dense_1_23469:	
dense_1_23471:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallÅ
sequential/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_22404à
rescaling/PartitionedCallPartitionedCall#sequential/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_23345
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_23359conv2d_23361*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23358ê
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23290
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_23377conv2d_1_23379*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23376ð
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23302
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_23395conv2d_2_23397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23394ð
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_23314ß
dropout/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23406
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_3_23420conv2d_3_23422*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23419ñ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23326ä
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_23431Ò
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23439ü
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_23453dense_23455*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23452
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_23469dense_1_23471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_23468w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

þ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23419

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
×

E__inference_sequential_layer_call_and_return_conditional_losses_23281
random_contrast_input#
random_contrast_23274:	
random_zoom_23277:	
identity¢'random_contrast/StatefulPartitionedCall¢#random_zoom/StatefulPartitionedCall
'random_contrast/StatefulPartitionedCallStatefulPartitionedCallrandom_contrast_inputrandom_contrast_23274*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_23227
#random_zoom/StatefulPartitionedCallStatefulPartitionedCall0random_contrast/StatefulPartitionedCall:output:0random_zoom_23277*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_22517
IdentityIdentity,random_zoom/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
NoOpNoOp(^random_contrast/StatefulPartitionedCall$^random_zoom/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  : : 2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall2J
#random_zoom/StatefulPartitionedCall#random_zoom/StatefulPartitionedCall:h d
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
/
_user_specified_namerandom_contrast_input

þ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_25833

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ
K
/__inference_random_contrast_layer_call_fn_25925

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_22395j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¶
K
/__inference_max_pooling2d_2_layer_call_fn_25781

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_23314
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
îU
â
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_body_24597
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counter
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterationsO
Ksequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholderQ
Msequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_strided_slice_0
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_sequential_random_contrast_loop_body_gatherv2_pfor_gatherv2_0
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0L
Hsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identityN
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identity_1N
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identity_2N
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identity_3
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_strided_slice
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_sequential_random_contrast_loop_body_gatherv2_pfor_gatherv2
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_pfor_addv2
Esequential/random_contrast/loop_body/adjust_contrast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Csequential/random_contrast/loop_body/adjust_contrast/pfor/while/addAddV2Ksequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholderNsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Usequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¶
Ssequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stackPackKsequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholder^sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Wsequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¶
Usequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1PackGsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add:z:0`sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¦
Usequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
Msequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_sliceStridedSlicesequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_sequential_random_contrast_loop_body_gatherv2_pfor_gatherv2_0\sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack:output:0^sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1:output:0^sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*$
_output_shapes
:  *
ellipsis_mask*
shrink_axis_mask
Gsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Esequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_1AddV2Ksequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholderPsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Wsequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : º
Usequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stackPackKsequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholder`sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Ysequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¼
Wsequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1PackIsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_1:z:0bsequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¨
Wsequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      û
Osequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1StridedSlicesequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0^sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack:output:0`sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1:output:0`sequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
ellipsis_mask*
shrink_axis_mask¼
Psequential/random_contrast/loop_body/adjust_contrast/pfor/while/AdjustContrastv2AdjustContrastv2Vsequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice:output:0Xsequential/random_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1:output:0*$
_output_shapes
:  
Nsequential/random_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¿
Jsequential/random_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims
ExpandDimsYsequential/random_contrast/loop_body/adjust_contrast/pfor/while/AdjustContrastv2:output:0Wsequential/random_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims/dim:output:0*
T0*(
_output_shapes
:  ª
dsequential/random_contrast/loop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemMsequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1Ksequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholderSsequential/random_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Gsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
Esequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_2AddV2Ksequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholderPsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Gsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :Ð
Esequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_3AddV2sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counterPsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: À
Hsequential/random_contrast/loop_body/adjust_contrast/pfor/while/IdentityIdentityIsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_3:z:0*
T0*
_output_shapes
: 
Jsequential/random_contrast/loop_body/adjust_contrast/pfor/while/Identity_1Identitysequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterations*
T0*
_output_shapes
: Â
Jsequential/random_contrast/loop_body/adjust_contrast/pfor/while/Identity_2IdentityIsequential/random_contrast/loop_body/adjust_contrast/pfor/while/add_2:z:0*
T0*
_output_shapes
: í
Jsequential/random_contrast/loop_body/adjust_contrast/pfor/while/Identity_3Identitytsequential/random_contrast/loop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Hsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identityQsequential/random_contrast/loop_body/adjust_contrast/pfor/while/Identity:output:0"¡
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identity_1Ssequential/random_contrast/loop_body/adjust_contrast/pfor/while/Identity_1:output:0"¡
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identity_2Ssequential/random_contrast/loop_body/adjust_contrast/pfor/while/Identity_2:output:0"¡
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identity_3Ssequential/random_contrast/loop_body/adjust_contrast/pfor/while/Identity_3:output:0"
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_strided_slicesequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_strided_slice_0"¸
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_pfor_addv2sequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0"
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_sequential_random_contrast_loop_body_gatherv2_pfor_gatherv2sequential_random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_sequential_random_contrast_loop_body_gatherv2_pfor_gatherv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  :)%
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ð
Ò
/loop_body_adjust_contrast_pfor_while_cond_26524Z
Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter`
\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations4
0loop_body_adjust_contrast_pfor_while_placeholder6
2loop_body_adjust_contrast_pfor_while_placeholder_1Z
Vloop_body_adjust_contrast_pfor_while_less_loop_body_adjust_contrast_pfor_strided_sliceq
mloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_cond_26524___redundant_placeholder0q
mloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_cond_26524___redundant_placeholder11
-loop_body_adjust_contrast_pfor_while_identity
Ü
)loop_body/adjust_contrast/pfor/while/LessLess0loop_body_adjust_contrast_pfor_while_placeholderVloop_body_adjust_contrast_pfor_while_less_loop_body_adjust_contrast_pfor_strided_slice*
T0*
_output_shapes
: 
-loop_body/adjust_contrast/pfor/while/IdentityIdentity-loop_body/adjust_contrast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "g
-loop_body_adjust_contrast_pfor_while_identity6loop_body/adjust_contrast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25726

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
à
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_24278¥
 sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counter«
¦sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterationsY
Usequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder[
Wsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1¥
 sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_less_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice¼
·sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_24278___redundant_placeholder0¼
·sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_24278___redundant_placeholder1¼
·sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_24278___redundant_placeholder2¼
·sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_24278___redundant_placeholder3V
Rsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity
ñ
Nsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/LessLessUsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_less_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice*
T0*
_output_shapes
: Ó
Rsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentityRsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Less:z:0*
T0
*
_output_shapes
: "±
Rsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity[sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:

ú
A__inference_conv2d_layer_call_and_return_conditional_losses_23358

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
£

ô
@__inference_dense_layer_call_and_return_conditional_losses_23452

inputs2
matmul_readvariableop_resource:
d.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
¢
Ê
arandom_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_25327¿
ºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterÅ
Àrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsf
brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderh
drandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1¿
ºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceÖ
Ñrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_25327___redundant_placeholder0Ö
Ñrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_25327___redundant_placeholder1Ö
Ñrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_25327___redundant_placeholder2Ö
Ñrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_25327___redundant_placeholder3c
_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity
¥
[random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/LessLessbrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: í
_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentity_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "Ë
_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityhrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
°

a
B__inference_dropout_layer_call_and_return_conditional_losses_25813

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¹
	
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_22900§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter­
¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsZ
Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice¾
¹loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_cond_22900___redundant_placeholder0	W
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity
õ
Oloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/LessLessVloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice*
T0*
_output_shapes
: Õ
Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitySloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Less:z:0*
T0
*
_output_shapes
: "³
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity\loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:


Srandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_25101£
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter©
¤random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsX
Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderZ
Vrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1£
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_sliceº
µrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_cond_25101___redundant_placeholder0	U
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity
í
Mrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/LessLessTrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: Ñ
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityQrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityZrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25746

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿPP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
­
á	
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_26017
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsM
Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderO
Kloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_26017___redundant_placeholder0¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_26017___redundant_placeholder1¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_26017___redundant_placeholder2J
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity
Á
Bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/LessLessIloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: »
Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityFloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityOloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
É	
ô
B__inference_dense_1_layer_call_and_return_conditional_losses_23468

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
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

ü
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23376

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿPP: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
Ç
	
\sequential_random_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_24154µ
°sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter»
¶sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsa
]sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderc
_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1µ
°sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_less_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_sliceÌ
Çsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_cond_24154___redundant_placeholder0	^
Zsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity

Vsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/LessLess]sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder°sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_less_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice*
T0*
_output_shapes
: ã
Zsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentityZsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "Á
Zsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identitycsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

a
E__inference_sequential_layer_call_and_return_conditional_losses_24888

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_25843

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_23548

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
É	
ô
B__inference_dense_1_layer_call_and_return_conditional_losses_25920

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
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
9


Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_26083~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsF
Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderH
Dloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1{
wloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0	C
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityE
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1E
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2E
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3y
uloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice	~
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ï
:loop_body/stateful_uniform_full_int/Bitcast/pfor/while/addAddV2Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderEloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stackPackBloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderUloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Nloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1Pack>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add:z:0Wloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ã
Dloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0Sloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack:output:0Uloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1:output:0Uloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÉ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/BitcastBitcastMloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Eloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims
ExpandDimsGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Bitcast:output:0Nloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
[loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemDloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderJloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ó
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1AddV2Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :«
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2AddV2zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counterGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ®
?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity@loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ñ
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: °
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2Identity@loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: Û
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3Identitykloop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityHloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3:output:0"ð
uloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slicewloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0"
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedsliceloop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ï>

G__inference_sequential_1_layer_call_and_return_conditional_losses_23852
sequential_input
sequential_23808:	
sequential_23810:	&
conv2d_23814:
conv2d_23816:(
conv2d_1_23820: 
conv2d_1_23822: (
conv2d_2_23826: @
conv2d_2_23828:@)
conv2d_3_23833:@
conv2d_3_23835:	
dense_23841:
d
dense_23843:	 
dense_1_23846:	
dense_1_23848:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall
"sequential/StatefulPartitionedCallStatefulPartitionedCallsequential_inputsequential_23808sequential_23810*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23249è
rescaling/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_23345
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_23814conv2d_23816*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23358ê
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23290
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_23820conv2d_1_23822*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23376ð
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23302
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_23826conv2d_2_23828*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23394ð
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_23314ï
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23581
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_3_23833conv2d_3_23835*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23419ñ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23326
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_23548Ú
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23439ü
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_23841dense_23843*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23452
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_23846dense_1_23848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_23468w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
*
_user_specified_namesequential_input

©
J__inference_random_contrast_layer_call_and_return_conditional_losses_26633

inputsI
;loop_body_stateful_uniform_full_int_rngreadandskip_resource:	
identity¢2loop_body/stateful_uniform_full_int/RngReadAndSkip¢=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
Rank/packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:W
	Max/inputPackstrided_slice:output:0*
N*
T0*
_output_shapes
:O
MaxMaxMax/input:output:0range:output:0*
T0*
_output_shapes
: h
&loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : 
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_sliceStridedSliceloop_body/Shape:output:0&loop_body/strided_slice/stack:output:0(loop_body/strided_slice/stack_1:output:0(loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :}
loop_body/GreaterGreater loop_body/strided_slice:output:0loop_body/Greater/y:output:0*
T0*
_output_shapes
: V
loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B :  
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:  s
)loop_body/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:s
)loop_body/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¹
(loop_body/stateful_uniform_full_int/ProdProd2loop_body/stateful_uniform_full_int/shape:output:02loop_body/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: l
*loop_body/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*loop_body/stateful_uniform_full_int/Cast_1Cast1loop_body/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2loop_body/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip;loop_body_stateful_uniform_full_int_rngreadandskip_resource3loop_body/stateful_uniform_full_int/Cast/x:output:0.loop_body/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
7loop_body/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9loop_body/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/stateful_uniform_full_int/strided_sliceStridedSlice:loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0@loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask£
+loop_body/stateful_uniform_full_int/BitcastBitcast:loop_body/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9loop_body/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3loop_body/stateful_uniform_full_int/strided_slice_1StridedSlice:loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0Bloop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:§
-loop_body/stateful_uniform_full_int/Bitcast_1Bitcast<loop_body/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0i
'loop_body/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :Ã
#loop_body/stateful_uniform_full_intStatelessRandomUniformFullIntV22loop_body/stateful_uniform_full_int/shape:output:06loop_body/stateful_uniform_full_int/Bitcast_1:output:04loop_body/stateful_uniform_full_int/Bitcast:output:00loop_body/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	^
loop_body/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 
loop_body/stackPack,loop_body/stateful_uniform_full_int:output:0loop_body/zeros_like:output:0*
N*
T0	*
_output_shapes

:p
loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ±
loop_body/strided_slice_1StridedSliceloop_body/stack:output:0(loop_body/strided_slice_1/stack:output:0*loop_body/strided_slice_1/stack_1:output:0*loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskk
(loop_body/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB k
&loop_body/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?k
&loop_body/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?¥
?loop_body/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"loop_body/strided_slice_1:output:0* 
_output_shapes
::
?loop_body/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
;loop_body/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV21loop_body/stateless_random_uniform/shape:output:0Eloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: °
&loop_body/stateless_random_uniform/subSub/loop_body/stateless_random_uniform/max:output:0/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: À
&loop_body/stateless_random_uniform/mulMulDloop_body/stateless_random_uniform/StatelessRandomUniformV2:output:0*loop_body/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ©
"loop_body/stateless_random_uniformAddV2*loop_body/stateless_random_uniform/mul:z:0/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
loop_body/adjust_contrastAdjustContrastv2loop_body/GatherV2:output:0&loop_body/stateless_random_uniform:z:0*$
_output_shapes
:  
"loop_body/adjust_contrast/IdentityIdentity"loop_body/adjust_contrast:output:0*
T0*$
_output_shapes
:  f
!loop_body/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C²
loop_body/clip_by_value/MinimumMinimum+loop_body/adjust_contrast/Identity:output:0*loop_body/clip_by_value/Minimum/y:output:0*
T0*$
_output_shapes
:  ^
loop_body/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
loop_body/clip_by_valueMaximum#loop_body/clip_by_value/Minimum:z:0"loop_body/clip_by_value/y:output:0*
T0*$
_output_shapes
:  \
pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
pfor/ReshapeReshapeMax:output:0pfor/Reshape/shape:output:0*
T0*
_output_shapes
:R
pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : R
pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :|

pfor/rangeRangepfor/range/start:output:0Max:output:0pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Kloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Mloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Mloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Tloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack:output:0Vloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1:output:0Vloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Sloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2TensorListReserve\loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Ploop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Jloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileWhileSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counter:output:0Yloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterations:output:0Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const:output:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2:handle:0Nloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0;loop_body_stateful_uniform_full_int_rngreadandskip_resource3loop_body/stateful_uniform_full_int/Cast/x:output:0.loop_body/stateful_uniform_full_int/Cast_1:y:03^loop_body/stateful_uniform_full_int/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *T
bodyLRJ
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_26018*T
condLRJ
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_26017*#
output_shapes
: : : : : : : : 
?loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ©
Xloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ´
Jloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:output:3aloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0
Floop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Bloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
=loop_body/stateful_uniform_full_int/strided_slice/pfor/concatConcatV2Oloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0:output:0@loop_body/stateful_uniform_full_int/strided_slice/stack:output:0Kloop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1ConcatV2Qloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0Mloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Dloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2ConcatV2Qloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0Mloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:å
Cloop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSliceStridedSliceSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Floop_body/stateful_uniform_full_int/strided_slice/pfor/concat:output:0Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1:output:0Hloop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Dloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Floop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Floop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Mloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack:output:0Oloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1:output:0Oloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÅ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2TensorListReserveUloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shape:output:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌx
6loop_body/stateful_uniform_full_int/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Iloop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Cloop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
6loop_body/stateful_uniform_full_int/Bitcast/pfor/whileStatelessWhileLloop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counter:output:0Rloop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterations:output:0?loop_body/stateful_uniform_full_int/Bitcast/pfor/Const:output:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2:handle:0Gloop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0Lloop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *M
bodyERC
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_26083*M
condERC
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_26082*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ{
8loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¢
Qloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Cloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2TensorListConcatV2?loop_body/stateful_uniform_full_int/Bitcast/pfor/while:output:3Zloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shape:output:0Aloop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Hloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
?loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concatConcatV2Qloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0:output:0Bloop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0Mloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Floop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
Aloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1ConcatV2Sloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Oloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Floop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
Aloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2ConcatV2Sloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0:output:0Dloop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0Oloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:í
Eloop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSliceStridedSliceSloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Hloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat:output:0Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1:output:0Jloop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Floop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Hloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Hloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Oloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack:output:0Qloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1:output:0Qloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿË
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2TensorListReserveWloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌz
8loop_body/stateful_uniform_full_int/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Kloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Eloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ®
8loop_body/stateful_uniform_full_int/Bitcast_1/pfor/whileStatelessWhileNloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counter:output:0Tloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterations:output:0Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const:output:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2:handle:0Iloop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0Nloop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *O
bodyGRE
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_26150*O
condGRE
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_26149*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ}
:loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¤
Sloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
Eloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV2Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while:output:3\loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
<loop_body/stateful_uniform_full_int/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ü
6loop_body/stateful_uniform_full_int/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Eloop_body/stateful_uniform_full_int/pfor/strided_slice/stack:output:0Gloop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1:output:0Gloop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Dloop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ­
6loop_body/stateful_uniform_full_int/pfor/TensorArrayV2TensorListReserveMloop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shape:output:0?loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐp
.loop_body/stateful_uniform_full_int/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform_full_int/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ}
;loop_body/stateful_uniform_full_int/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ë
.loop_body/stateful_uniform_full_int/pfor/whileStatelessWhileDloop_body/stateful_uniform_full_int/pfor/while/loop_counter:output:0Jloop_body/stateful_uniform_full_int/pfor/while/maximum_iterations:output:07loop_body/stateful_uniform_full_int/pfor/Const:output:0?loop_body/stateful_uniform_full_int/pfor/TensorArrayV2:handle:0?loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2:tensor:0Lloop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2:tensor:02loop_body/stateful_uniform_full_int/shape:output:00loop_body/stateful_uniform_full_int/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *E
body=R;
9loop_body_stateful_uniform_full_int_pfor_while_body_26207*E
cond=R;
9loop_body_stateful_uniform_full_int_pfor_while_cond_26206*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: s
0loop_body/stateful_uniform_full_int/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
Iloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿø
;loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2TensorListConcatV27loop_body/stateful_uniform_full_int/pfor/while:output:3Rloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shape:output:09loop_body/stateful_uniform_full_int/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0d
loop_body/stack/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/stack/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:f
$loop_body/stack/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :¹
loop_body/stack/pfor/ones_likeFill=loop_body/stack/pfor/ones_like/Shape/shape_as_tensor:output:0-loop_body/stack/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:u
"loop_body/stack/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¢
loop_body/stack/pfor/ReshapeReshape'loop_body/stack/pfor/ones_like:output:0+loop_body/stack/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:w
$loop_body/stack/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
loop_body/stack/pfor/Reshape_1Reshapepfor/Reshape:output:0-loop_body/stack/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:b
 loop_body/stack/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
loop_body/stack/pfor/concatConcatV2'loop_body/stack/pfor/Reshape_1:output:0%loop_body/stack/pfor/Reshape:output:0)loop_body/stack/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:e
#loop_body/stack/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : £
loop_body/stack/pfor/ExpandDims
ExpandDimsloop_body/zeros_like:output:0,loop_body/stack/pfor/ExpandDims/dim:output:0*
T0	*
_output_shapes

:£
loop_body/stack/pfor/TileTile(loop_body/stack/pfor/ExpandDims:output:0$loop_body/stack/pfor/concat:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
loop_body/stack/pfor/stackPackDloop_body/stateful_uniform_full_int/pfor/TensorListConcatV2:tensor:0"loop_body/stack/pfor/Tile:output:0*
N*
T0	*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

axisx
.loop_body/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: l
*loop_body/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
%loop_body/strided_slice_1/pfor/concatConcatV27loop_body/strided_slice_1/pfor/concat/values_0:output:0(loop_body/strided_slice_1/stack:output:03loop_body/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:z
0loop_body/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: n
,loop_body/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
'loop_body/strided_slice_1/pfor/concat_1ConcatV29loop_body/strided_slice_1/pfor/concat_1/values_0:output:0*loop_body/strided_slice_1/stack_1:output:05loop_body/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:z
0loop_body/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:n
,loop_body/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
'loop_body/strided_slice_1/pfor/concat_2ConcatV29loop_body/strided_slice_1/pfor/concat_2/values_0:output:0*loop_body/strided_slice_1/stack_2:output:05loop_body/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:í
+loop_body/strided_slice_1/pfor/StridedSliceStridedSlice#loop_body/stack/pfor/stack:output:0.loop_body/strided_slice_1/pfor/concat:output:00loop_body/strided_slice_1/pfor/concat_1:output:00loop_body/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask¢
Xloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¤
Zloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¤
Zloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_sliceStridedSlicepfor/Reshape:output:0aloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack:output:0cloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1:output:0cloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask«
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2TensorListReserveiloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shape:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ­
bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Tloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1TensorListReservekloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shape:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Jloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¨
]loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
Jloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/whileStatelessWhile`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counter:output:0floop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterations:output:0Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const:output:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2:handle:0]loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1:handle:0[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:04loop_body/strided_slice_1/pfor/StridedSlice:output:0*
T
	2	*
_lower_using_switch_merge(*
_num_original_outputs*3
_output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *a
bodyYRW
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_26307*a
condYRW
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_26306*2
output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ
Lloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¶
eloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   è
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2TensorListConcatV2Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:3nloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shape:output:0Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Lloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ¸
gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ì
Yloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1TensorListConcatV2Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:4ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shape:output:0Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Tloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:  
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ü
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlicepfor/Reshape:output:0]loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask§
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿõ
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReserveeloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Floop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¤
Yloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Sloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Æ

Floop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhile\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0bloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const:output:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0Wloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2:tensor:0bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1:tensor:01loop_body/stateless_random_uniform/shape:output:0Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *]
bodyURS
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_26376*]
condURS
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_26375*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 
Hloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ´
aloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÔ
Sloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while:output:3jloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0r
0loop_body/stateless_random_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :t
2loop_body/stateless_random_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : s
1loop_body/stateless_random_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ò
/loop_body/stateless_random_uniform/mul/pfor/addAddV2;loop_body/stateless_random_uniform/mul/pfor/Rank_1:output:0:loop_body/stateless_random_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: Ï
3loop_body/stateless_random_uniform/mul/pfor/MaximumMaximum3loop_body/stateless_random_uniform/mul/pfor/add:z:09loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: ½
1loop_body/stateless_random_uniform/mul/pfor/ShapeShape\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:Ë
/loop_body/stateless_random_uniform/mul/pfor/subSub7loop_body/stateless_random_uniform/mul/pfor/Maximum:z:09loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
9loop_body/stateless_random_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ü
3loop_body/stateless_random_uniform/mul/pfor/ReshapeReshape3loop_body/stateless_random_uniform/mul/pfor/sub:z:0Bloop_body/stateless_random_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
6loop_body/stateless_random_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ú
0loop_body/stateless_random_uniform/mul/pfor/TileTile?loop_body/stateless_random_uniform/mul/pfor/Tile/input:output:0<loop_body/stateless_random_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
?loop_body/stateless_random_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
9loop_body/stateless_random_uniform/mul/pfor/strided_sliceStridedSlice:loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Hloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Aloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Cloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Cloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
;loop_body/stateless_random_uniform/mul/pfor/strided_slice_1StridedSlice:loop_body/stateless_random_uniform/mul/pfor/Shape:output:0Jloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack:output:0Lloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1:output:0Lloop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masky
7loop_body/stateless_random_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
2loop_body/stateless_random_uniform/mul/pfor/concatConcatV2Bloop_body/stateless_random_uniform/mul/pfor/strided_slice:output:09loop_body/stateless_random_uniform/mul/pfor/Tile:output:0Dloop_body/stateless_random_uniform/mul/pfor/strided_slice_1:output:0@loop_body/stateless_random_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
5loop_body/stateless_random_uniform/mul/pfor/Reshape_1Reshape\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0;loop_body/stateless_random_uniform/mul/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
/loop_body/stateless_random_uniform/mul/pfor/MulMul>loop_body/stateless_random_uniform/mul/pfor/Reshape_1:output:0*loop_body/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,loop_body/stateless_random_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :p
.loop_body/stateless_random_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : o
-loop_body/stateless_random_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Æ
+loop_body/stateless_random_uniform/pfor/addAddV27loop_body/stateless_random_uniform/pfor/Rank_1:output:06loop_body/stateless_random_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: Ã
/loop_body/stateless_random_uniform/pfor/MaximumMaximum/loop_body/stateless_random_uniform/pfor/add:z:05loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
-loop_body/stateless_random_uniform/pfor/ShapeShape3loop_body/stateless_random_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:¿
+loop_body/stateless_random_uniform/pfor/subSub3loop_body/stateless_random_uniform/pfor/Maximum:z:05loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
5loop_body/stateless_random_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ð
/loop_body/stateless_random_uniform/pfor/ReshapeReshape/loop_body/stateless_random_uniform/pfor/sub:z:0>loop_body/stateless_random_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:|
2loop_body/stateless_random_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Î
,loop_body/stateless_random_uniform/pfor/TileTile;loop_body/stateless_random_uniform/pfor/Tile/input:output:08loop_body/stateless_random_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: 
;loop_body/stateless_random_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=loop_body/stateless_random_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=loop_body/stateless_random_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5loop_body/stateless_random_uniform/pfor/strided_sliceStridedSlice6loop_body/stateless_random_uniform/pfor/Shape:output:0Dloop_body/stateless_random_uniform/pfor/strided_slice/stack:output:0Floop_body/stateless_random_uniform/pfor/strided_slice/stack_1:output:0Floop_body/stateless_random_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
=loop_body/stateless_random_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7loop_body/stateless_random_uniform/pfor/strided_slice_1StridedSlice6loop_body/stateless_random_uniform/pfor/Shape:output:0Floop_body/stateless_random_uniform/pfor/strided_slice_1/stack:output:0Hloop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1:output:0Hloop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
3loop_body/stateless_random_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
.loop_body/stateless_random_uniform/pfor/concatConcatV2>loop_body/stateless_random_uniform/pfor/strided_slice:output:05loop_body/stateless_random_uniform/pfor/Tile:output:0@loop_body/stateless_random_uniform/pfor/strided_slice_1:output:0<loop_body/stateless_random_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ø
1loop_body/stateless_random_uniform/pfor/Reshape_1Reshape3loop_body/stateless_random_uniform/mul/pfor/Mul:z:07loop_body/stateless_random_uniform/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
-loop_body/stateless_random_uniform/pfor/AddV2AddV2:loop_body/stateless_random_uniform/pfor/Reshape_1:output:0/loop_body/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/addAddV2%loop_body/SelectV2/pfor/Rank:output:0&loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :`
loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : a
loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: 
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: 
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/SelectV2/pfor/TileTile+loop_body/SelectV2/pfor/Tile/input:output:0(loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: u
+loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%loop_body/SelectV2/pfor/strided_sliceStridedSlice&loop_body/SelectV2/pfor/Shape:output:04loop_body/SelectV2/pfor/strided_slice/stack:output:06loop_body/SelectV2/pfor/strided_slice/stack_1:output:06loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
'loop_body/SelectV2/pfor/strided_slice_1StridedSlice&loop_body/SelectV2/pfor/Shape:output:06loop_body/SelectV2/pfor/strided_slice_1/stack:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : î
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  |
2loop_body/adjust_contrast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4loop_body/adjust_contrast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/adjust_contrast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ô
,loop_body/adjust_contrast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0;loop_body/adjust_contrast/pfor/strided_slice/stack:output:0=loop_body/adjust_contrast/pfor/strided_slice/stack_1:output:0=loop_body/adjust_contrast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
:loop_body/adjust_contrast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
,loop_body/adjust_contrast/pfor/TensorArrayV2TensorListReserveCloop_body/adjust_contrast/pfor/TensorArrayV2/element_shape:output:05loop_body/adjust_contrast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
$loop_body/adjust_contrast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
7loop_body/adjust_contrast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿs
1loop_body/adjust_contrast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ï
$loop_body/adjust_contrast/pfor/whileStatelessWhile:loop_body/adjust_contrast/pfor/while/loop_counter:output:0@loop_body/adjust_contrast/pfor/while/maximum_iterations:output:0-loop_body/adjust_contrast/pfor/Const:output:05loop_body/adjust_contrast/pfor/TensorArrayV2:handle:05loop_body/adjust_contrast/pfor/strided_slice:output:0)loop_body/GatherV2/pfor/GatherV2:output:01loop_body/stateless_random_uniform/pfor/AddV2:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *;
body3R1
/loop_body_adjust_contrast_pfor_while_body_26525*;
cond3R1
/loop_body_adjust_contrast_pfor_while_cond_26524*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿi
&loop_body/adjust_contrast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
?loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ           Ú
1loop_body/adjust_contrast/pfor/TensorListConcatV2TensorListConcatV2-loop_body/adjust_contrast/pfor/while:output:3Hloop_body/adjust_contrast/pfor/TensorListConcatV2/element_shape:output:0/loop_body/adjust_contrast/pfor/Const_1:output:0*@
_output_shapes.
,:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0´
0loop_body/adjust_contrast/Identity/pfor/IdentityIdentity:loop_body/adjust_contrast/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  k
)loop_body/clip_by_value/Minimum/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/clip_by_value/Minimum/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/clip_by_value/Minimum/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/clip_by_value/Minimum/pfor/addAddV24loop_body/clip_by_value/Minimum/pfor/Rank_1:output:03loop_body/clip_by_value/Minimum/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/clip_by_value/Minimum/pfor/MaximumMaximum,loop_body/clip_by_value/Minimum/pfor/add:z:02loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/clip_by_value/Minimum/pfor/ShapeShape9loop_body/adjust_contrast/Identity/pfor/Identity:output:0*
T0*
_output_shapes
:¶
(loop_body/clip_by_value/Minimum/pfor/subSub0loop_body/clip_by_value/Minimum/pfor/Maximum:z:02loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/clip_by_value/Minimum/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/clip_by_value/Minimum/pfor/ReshapeReshape,loop_body/clip_by_value/Minimum/pfor/sub:z:0;loop_body/clip_by_value/Minimum/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/clip_by_value/Minimum/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/clip_by_value/Minimum/pfor/TileTile8loop_body/clip_by_value/Minimum/pfor/Tile/input:output:05loop_body/clip_by_value/Minimum/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/clip_by_value/Minimum/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/clip_by_value/Minimum/pfor/strided_sliceStridedSlice3loop_body/clip_by_value/Minimum/pfor/Shape:output:0Aloop_body/clip_by_value/Minimum/pfor/strided_slice/stack:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/clip_by_value/Minimum/pfor/strided_slice_1StridedSlice3loop_body/clip_by_value/Minimum/pfor/Shape:output:0Cloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack:output:0Eloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1:output:0Eloop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/clip_by_value/Minimum/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/clip_by_value/Minimum/pfor/concatConcatV2;loop_body/clip_by_value/Minimum/pfor/strided_slice:output:02loop_body/clip_by_value/Minimum/pfor/Tile:output:0=loop_body/clip_by_value/Minimum/pfor/strided_slice_1:output:09loop_body/clip_by_value/Minimum/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:æ
.loop_body/clip_by_value/Minimum/pfor/Reshape_1Reshape9loop_body/adjust_contrast/Identity/pfor/Identity:output:04loop_body/clip_by_value/Minimum/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Ø
,loop_body/clip_by_value/Minimum/pfor/MinimumMinimum7loop_body/clip_by_value/Minimum/pfor/Reshape_1:output:0*loop_body/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  c
!loop_body/clip_by_value/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :e
#loop_body/clip_by_value/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : d
"loop_body/clip_by_value/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :¥
 loop_body/clip_by_value/pfor/addAddV2,loop_body/clip_by_value/pfor/Rank_1:output:0+loop_body/clip_by_value/pfor/add/y:output:0*
T0*
_output_shapes
: ¢
$loop_body/clip_by_value/pfor/MaximumMaximum$loop_body/clip_by_value/pfor/add:z:0*loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: 
"loop_body/clip_by_value/pfor/ShapeShape0loop_body/clip_by_value/Minimum/pfor/Minimum:z:0*
T0*
_output_shapes
:
 loop_body/clip_by_value/pfor/subSub(loop_body/clip_by_value/pfor/Maximum:z:0*loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: t
*loop_body/clip_by_value/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¯
$loop_body/clip_by_value/pfor/ReshapeReshape$loop_body/clip_by_value/pfor/sub:z:03loop_body/clip_by_value/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:q
'loop_body/clip_by_value/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:­
!loop_body/clip_by_value/pfor/TileTile0loop_body/clip_by_value/pfor/Tile/input:output:0-loop_body/clip_by_value/pfor/Reshape:output:0*
T0*
_output_shapes
: z
0loop_body/clip_by_value/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2loop_body/clip_by_value/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2loop_body/clip_by_value/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
*loop_body/clip_by_value/pfor/strided_sliceStridedSlice+loop_body/clip_by_value/pfor/Shape:output:09loop_body/clip_by_value/pfor/strided_slice/stack:output:0;loop_body/clip_by_value/pfor/strided_slice/stack_1:output:0;loop_body/clip_by_value/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
2loop_body/clip_by_value/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:~
4loop_body/clip_by_value/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4loop_body/clip_by_value/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
,loop_body/clip_by_value/pfor/strided_slice_1StridedSlice+loop_body/clip_by_value/pfor/Shape:output:0;loop_body/clip_by_value/pfor/strided_slice_1/stack:output:0=loop_body/clip_by_value/pfor/strided_slice_1/stack_1:output:0=loop_body/clip_by_value/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskj
(loop_body/clip_by_value/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¨
#loop_body/clip_by_value/pfor/concatConcatV23loop_body/clip_by_value/pfor/strided_slice:output:0*loop_body/clip_by_value/pfor/Tile:output:05loop_body/clip_by_value/pfor/strided_slice_1:output:01loop_body/clip_by_value/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Í
&loop_body/clip_by_value/pfor/Reshape_1Reshape0loop_body/clip_by_value/Minimum/pfor/Minimum:z:0,loop_body/clip_by_value/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Â
&loop_body/clip_by_value/pfor/Maximum_1Maximum/loop_body/clip_by_value/pfor/Reshape_1:output:0"loop_body/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
IdentityIdentity*loop_body/clip_by_value/pfor/Maximum_1:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  »
NoOpNoOp3^loop_body/stateful_uniform_full_int/RngReadAndSkip>^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : 2h
2loop_body/stateful_uniform_full_int/RngReadAndSkip2loop_body/stateful_uniform_full_int/RngReadAndSkip2~
=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while=loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Û]
Ä
erandom_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_25259Ç
Ârandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counterÍ
Èrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsj
frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderl
hrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1l
hrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Ä
¿random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0«
¦random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0	g
crandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identityi
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1i
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2i
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3i
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4Â
½random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice©
¤random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice	¢
`random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Û
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/addAddV2frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderirandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/y:output:0*
T0*
_output_shapes
: ²
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stackPackfrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderyrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:´
rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1Packbrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add:z:0{random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:Á
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ï
hrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_sliceStridedSlice¦random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack:output:0yrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1:output:0yrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask¬
wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterqrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice:output:0* 
_output_shapes
::«
irandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims
ExpandDims}random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:key:0rrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemhrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholdernrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ­
krandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
grandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1
ExpandDimsrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:counter:0trandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:
random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemhrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderprandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1:output:0*
_output_shapes
: *
element_dtype0:éèÌ¤
brandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ß
`random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1AddV2frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderkrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ¤
brandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :¼
`random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2AddV2Ârandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counterkrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ö
crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitydrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2:z:0*
T0*
_output_shapes
: Ý
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1IdentityÈrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterations*
T0*
_output_shapes
: ø
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2Identitydrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1:z:0*
T0*
_output_shapes
: ¤
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3Identityrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: ¦
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4Identityrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "Ó
crandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identitylrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0"×
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1:output:0"×
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2:output:0"×
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3:output:0"×
erandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4:output:0"
½random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice¿random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0"Ð
¤random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice¦random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25756

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ú
A__inference_conv2d_layer_call_and_return_conditional_losses_25716

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
÷:

Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_22612
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsM
Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderO
Kloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0
{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0:	n
jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0n
jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0J
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityL
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1L
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2L
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice
yloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	l
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_xl
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1¢Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipÏ
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkip{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0*
_output_shapes
:
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¬
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsTloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0Uloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:¢
bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemKloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderQloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Cloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Aloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/addAddV2Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderLloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :È
Cloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1AddV2loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counterNloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityGloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1:z:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ë
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsC^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2IdentityEloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add:z:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ®
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3Identityrloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ó
Bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOpNoOpM^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityOloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3:output:0"Ö
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0"Ö
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_xjloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0"
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_sliceloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0"ø
yloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipLloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¦
»
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_22676~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsF
Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderH
Dloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_cond_22676___redundant_placeholder0	C
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity
¤
;loop_body/stateful_uniform_full_int/Bitcast/pfor/while/LessLessBloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderzloop_body_stateful_uniform_full_int_bitcast_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice*
T0*
_output_shapes
: ­
?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityHloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
¨
Î
,__inference_sequential_1_layer_call_fn_23918

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:
d
	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_23475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
®>
å
/loop_body_adjust_contrast_pfor_while_body_26525Z
Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter`
\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations4
0loop_body_adjust_contrast_pfor_while_placeholder6
2loop_body_adjust_contrast_pfor_while_placeholder_1W
Sloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_slice_0Y
Uloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0h
dloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_01
-loop_body_adjust_contrast_pfor_while_identity3
/loop_body_adjust_contrast_pfor_while_identity_13
/loop_body_adjust_contrast_pfor_while_identity_23
/loop_body_adjust_contrast_pfor_while_identity_3U
Qloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_sliceW
Sloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2f
bloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2l
*loop_body/adjust_contrast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¹
(loop_body/adjust_contrast/pfor/while/addAddV20loop_body_adjust_contrast_pfor_while_placeholder3loop_body/adjust_contrast/pfor/while/add/y:output:0*
T0*
_output_shapes
: |
:loop_body/adjust_contrast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : å
8loop_body/adjust_contrast/pfor/while/strided_slice/stackPack0loop_body_adjust_contrast_pfor_while_placeholderCloop_body/adjust_contrast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:~
<loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : å
:loop_body/adjust_contrast/pfor/while/strided_slice/stack_1Pack,loop_body/adjust_contrast/pfor/while/add:z:0Eloop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
:loop_body/adjust_contrast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
2loop_body/adjust_contrast/pfor/while/strided_sliceStridedSliceUloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0Aloop_body/adjust_contrast/pfor/while/strided_slice/stack:output:0Cloop_body/adjust_contrast/pfor/while/strided_slice/stack_1:output:0Cloop_body/adjust_contrast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*$
_output_shapes
:  *
ellipsis_mask*
shrink_axis_maskn
,loop_body/adjust_contrast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :½
*loop_body/adjust_contrast/pfor/while/add_1AddV20loop_body_adjust_contrast_pfor_while_placeholder5loop_body/adjust_contrast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ~
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : é
:loop_body/adjust_contrast/pfor/while/strided_slice_1/stackPack0loop_body_adjust_contrast_pfor_while_placeholderEloop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
>loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ë
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1Pack.loop_body/adjust_contrast/pfor/while/add_1:z:0Gloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
4loop_body/adjust_contrast/pfor/while/strided_slice_1StridedSlicedloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_0Cloop_body/adjust_contrast/pfor/while/strided_slice_1/stack:output:0Eloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1:output:0Eloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
ellipsis_mask*
shrink_axis_maskë
5loop_body/adjust_contrast/pfor/while/AdjustContrastv2AdjustContrastv2;loop_body/adjust_contrast/pfor/while/strided_slice:output:0=loop_body/adjust_contrast/pfor/while/strided_slice_1:output:0*$
_output_shapes
:  u
3loop_body/adjust_contrast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : î
/loop_body/adjust_contrast/pfor/while/ExpandDims
ExpandDims>loop_body/adjust_contrast/pfor/while/AdjustContrastv2:output:0<loop_body/adjust_contrast/pfor/while/ExpandDims/dim:output:0*
T0*(
_output_shapes
:  ¾
Iloop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem2loop_body_adjust_contrast_pfor_while_placeholder_10loop_body_adjust_contrast_pfor_while_placeholder8loop_body/adjust_contrast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒn
,loop_body/adjust_contrast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :½
*loop_body/adjust_contrast/pfor/while/add_2AddV20loop_body_adjust_contrast_pfor_while_placeholder5loop_body/adjust_contrast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: n
,loop_body/adjust_contrast/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ã
*loop_body/adjust_contrast/pfor/while/add_3AddV2Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter5loop_body/adjust_contrast/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
-loop_body/adjust_contrast/pfor/while/IdentityIdentity.loop_body/adjust_contrast/pfor/while/add_3:z:0*
T0*
_output_shapes
: º
/loop_body/adjust_contrast/pfor/while/Identity_1Identity\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations*
T0*
_output_shapes
: 
/loop_body/adjust_contrast/pfor/while/Identity_2Identity.loop_body/adjust_contrast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ·
/loop_body/adjust_contrast/pfor/while/Identity_3IdentityYloop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "g
-loop_body_adjust_contrast_pfor_while_identity6loop_body/adjust_contrast/pfor/while/Identity:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_18loop_body/adjust_contrast/pfor/while/Identity_1:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_28loop_body/adjust_contrast/pfor/while/Identity_2:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_38loop_body/adjust_contrast/pfor/while/Identity_3:output:0"¨
Qloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_sliceSloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_slice_0"Ê
bloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2dloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_0"¬
Sloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2Uloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  :)%
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
çw
¿
arandom_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_25328¿
ºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterÅ
Àrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsf
brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderh
drandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1¼
·random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0Ó
Îrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0×
Òrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0
random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shape_0­
¨random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0c
_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identitye
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1e
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2e
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3º
µrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceÑ
Ìrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2Õ
Ðrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1
random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shape«
¦random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ï
Zrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/addAddV2brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdererandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: ®
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : û
jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackbrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderurandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:°
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : û
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1Pack^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add:z:0wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:½
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
drandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSliceÎrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0srandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask 
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ó
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdergrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: °
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ÿ
lrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackbrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderwrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:²
prandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1Pack`random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0yrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¿
nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
frandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSliceÒrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0urandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0wrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÂ
orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shape_0mrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0orandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0¨random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0*
_output_shapes
: §
erandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : þ
arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDimsxrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0nrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes
:
{random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemdrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderjrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ 
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :Ó
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2brandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdergrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
:  
^random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :¬
\random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2ºrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_countergrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: î
_random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentity`random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: Ñ
arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1IdentityÀrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: ð
arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2Identity`random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: 
arandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identityrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "Ë
_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityhrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"Ï
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"Ï
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"Ï
arandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3jrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"¦
random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shaperandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_shape_0"Ô
¦random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg¨random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0"ò
µrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice·random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0"¨
Ðrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1Òrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0" 
Ìrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2Îrandom_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÚN

csequential_random_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_24090Ã
¾sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counterÉ
Äsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsh
dsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderj
fsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1À
»sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0À
±sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0:	¥
 sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0¥
 sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0e
asequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityg
csequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1g
csequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2g
csequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3¾
¹sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¾
¯sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	£
sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_x£
sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_1¢gsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip
gsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkip±sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0 sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0 sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0*
_output_shapes
:©
gsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ý
csequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsosequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0psequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:
}sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemfsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1dsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderlsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ 
^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Õ
\sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/addAddV2dsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholdergsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¢
`sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :´
^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1AddV2¾sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counterisequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: Ò
asequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentitybsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1:z:0^^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ·
csequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1IdentityÄsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterations^^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ò
csequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2Identity`sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add:z:0^^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
csequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3Identitysequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
]sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOpNoOph^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "Ï
asequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityjsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0"Ó
csequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1lsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1:output:0"Ó
csequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2lsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2:output:0"Ó
csequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3lsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3:output:0"Ä
sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_1 sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_1_0"Ä
sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_x sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_cast_x_0"ú
¹sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice»sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0"æ
¯sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource±sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2Ò
gsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipgsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	
p
E__inference_sequential_layer_call_and_return_conditional_losses_23271
random_contrast_input
identityÞ
random_contrast/PartitionedCallPartitionedCallrandom_contrast_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_22395é
random_zoom/PartitionedCallPartitionedCall(random_contrast/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_22401v
IdentityIdentity$random_zoom/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :h d
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
/
_user_specified_namerandom_contrast_input
Á
E
)__inference_dropout_1_layer_call_fn_25848

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_23431i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
ØQ
ô
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_26307§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter­
¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsZ
Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2¤
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0	W
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identityY
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4¢
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice	
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :«
Nloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/addAddV2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderYloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¢
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ×
^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stackPackVloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderiloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:¤
bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ×
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1PackRloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add:z:0kloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:±
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Xloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_sliceStridedSliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack:output:0iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1:output:0iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounteraloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice:output:0* 
_output_shapes
::
Yloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims
ExpandDimsmloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:key:0bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Ö
oloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemXloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ç
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1
ExpandDimsqloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:counter:0dloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:Ú
qloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemXloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¯
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1AddV2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ü
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2AddV2¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: Ö
Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentityTloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2:z:0*
T0*
_output_shapes
: ­
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1Identity¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ø
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2IdentityTloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1:z:0*
T0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3Identityloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4Identityloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "³
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity\loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4:output:0"Â
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_sliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0"
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedsliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
è

(__inference_conv2d_2_layer_call_fn_25765

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23394w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ(( : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( 
 
_user_specified_nameinputs
º

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_25870

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs

¢
*__inference_sequential_layer_call_fn_23265
random_contrast_input
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallrandom_contrast_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23249y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
/
_user_specified_namerandom_contrast_input
f
«
psequential_random_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_24379Ý
Øsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counterã
Þsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsu
qsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderw
ssequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1w
ssequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Ú
Õsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0Á
¼sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_sequential_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0	r
nsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identityt
psequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1t
psequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2t
psequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3t
psequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4Ø
Ósequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice¿
ºsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_sequential_random_contrast_loop_body_strided_slice_1_pfor_stridedslice	­
ksequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ü
isequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/addAddV2qsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholdertsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/y:output:0*
T0*
_output_shapes
: ½
{sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ©
ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stackPackqsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholdersequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:¿
}sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ©
{sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1Packmsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add:z:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:Ì
{sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ´
ssequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_sliceStridedSlice¼sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_sequential_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack:output:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1:output:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÃ
sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter|sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice:output:0* 
_output_shapes
::¶
tsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ±
psequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims
ExpandDimssequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:key:0}sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Ã
sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemssequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1qsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ¸
vsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¹
rsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1
ExpandDimssequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:counter:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:Ç
sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemssequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2qsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder{sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1:output:0*
_output_shapes
: *
element_dtype0:éèÌ¯
msequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
ksequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1AddV2qsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholdervsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ¯
msequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :è
ksequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2AddV2Øsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_countervsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
nsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentityosequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2:z:0*
T0*
_output_shapes
: þ
psequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1IdentityÞsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterations*
T0*
_output_shapes
: 
psequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2Identityosequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1:z:0*
T0*
_output_shapes
: º
psequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3Identitysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: ¼
psequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4Identitysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "é
nsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identitywsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0"í
psequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1:output:0"í
psequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2:output:0"í
psequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3:output:0"í
psequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4:output:0"®
Ósequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_sliceÕsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0"ü
ºsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_sequential_random_contrast_loop_body_strided_slice_1_pfor_stridedslice¼sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_sequential_random_contrast_loop_body_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ì

(__inference_conv2d_3_layer_call_fn_25822

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23419x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
L
Ú
?random_contrast_loop_body_adjust_contrast_pfor_while_body_25477z
vrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counter
|random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterationsD
@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderF
Brandom_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1w
srandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_strided_slice_0y
urandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2_0
random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0A
=random_contrast_loop_body_adjust_contrast_pfor_while_identityC
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_1C
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_2C
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_3u
qrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_strided_slicew
srandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2
random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2|
:random_contrast/loop_body/adjust_contrast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :é
8random_contrast/loop_body/adjust_contrast/pfor/while/addAddV2@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderCrandom_contrast/loop_body/adjust_contrast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Jrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Hrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stackPack@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderSrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Lrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Jrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1Pack<random_contrast/loop_body/adjust_contrast/pfor/while/add:z:0Urandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Jrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¯
Brandom_contrast/loop_body/adjust_contrast/pfor/while/strided_sliceStridedSliceurandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2_0Qrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack:output:0Srandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_1:output:0Srandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*$
_output_shapes
:  *
ellipsis_mask*
shrink_axis_mask~
<random_contrast/loop_body/adjust_contrast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :í
:random_contrast/loop_body/adjust_contrast/pfor/while/add_1AddV2@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderErandom_contrast/loop_body/adjust_contrast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Lrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Jrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stackPack@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderUrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Nrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Lrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1Pack>random_contrast/loop_body/adjust_contrast/pfor/while/add_1:z:0Wrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Lrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¹
Drandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1StridedSlicerandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0Srandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack:output:0Urandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1:output:0Urandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
ellipsis_mask*
shrink_axis_mask
Erandom_contrast/loop_body/adjust_contrast/pfor/while/AdjustContrastv2AdjustContrastv2Krandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice:output:0Mrandom_contrast/loop_body/adjust_contrast/pfor/while/strided_slice_1:output:0*$
_output_shapes
:  
Crandom_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?random_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims
ExpandDimsNrandom_contrast/loop_body/adjust_contrast/pfor/while/AdjustContrastv2:output:0Lrandom_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims/dim:output:0*
T0*(
_output_shapes
:  þ
Yrandom_contrast/loop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemBrandom_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderHrandom_contrast/loop_body/adjust_contrast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ~
<random_contrast/loop_body/adjust_contrast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :í
:random_contrast/loop_body/adjust_contrast/pfor/while/add_2AddV2@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderErandom_contrast/loop_body/adjust_contrast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ~
<random_contrast/loop_body/adjust_contrast/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :£
:random_contrast/loop_body/adjust_contrast/pfor/while/add_3AddV2vrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counterErandom_contrast/loop_body/adjust_contrast/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: ª
=random_contrast/loop_body/adjust_contrast/pfor/while/IdentityIdentity>random_contrast/loop_body/adjust_contrast/pfor/while/add_3:z:0*
T0*
_output_shapes
: ê
?random_contrast/loop_body/adjust_contrast/pfor/while/Identity_1Identity|random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterations*
T0*
_output_shapes
: ¬
?random_contrast/loop_body/adjust_contrast/pfor/while/Identity_2Identity>random_contrast/loop_body/adjust_contrast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ×
?random_contrast/loop_body/adjust_contrast/pfor/while/Identity_3Identityirandom_contrast/loop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
=random_contrast_loop_body_adjust_contrast_pfor_while_identityFrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity:output:0"
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_1Hrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity_1:output:0"
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_2Hrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity_2:output:0"
?random_contrast_loop_body_adjust_contrast_pfor_while_identity_3Hrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity_3:output:0"è
qrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_strided_slicesrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_strided_slice_0"
random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2random_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_1_random_contrast_loop_body_stateless_random_uniform_pfor_addv2_0"ì
srandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2urandom_contrast_loop_body_adjust_contrast_pfor_while_strided_slice_random_contrast_loop_body_gatherv2_pfor_gatherv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  :)%
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Æ
^
B__inference_flatten_layer_call_and_return_conditional_losses_23439

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 2  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs


lsequential_random_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_24448Õ
Ðsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterÛ
Ösequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsq
msequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholders
osequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Ò
Ísequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0é
äsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0í
èsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0¬
§sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_shape_0Ã
¾sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0n
jsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityp
lsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1p
lsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2p
lsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3Ð
Ësequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceç
âsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2ë
æsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1ª
¥sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_shapeÁ
¼sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg©
gsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ð
esequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/addAddV2msequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderpsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¹
wsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
usequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackmsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdersequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:»
ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
wsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1Packisequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add:z:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:È
wsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ë
osequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSliceäsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0~sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask«
isequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ô
gsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2msequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderrsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: »
ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
wsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackmsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdersequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:½
{sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : £
ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1Packksequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:Ê
ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
qsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSliceèsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
zsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2§sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_shape_0xsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0zsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0¾sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0*
_output_shapes
: ²
psequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :  
lsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDimssequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0ysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes
:³
sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemosequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1msequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderusequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ«
isequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ô
gsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2msequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderrsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: «
isequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :Ø
gsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2Ðsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterrsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
jsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityksequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: ò
lsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1IdentityÖsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: 
lsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2Identityksequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: ²
lsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identitysequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "á
jsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityssequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"å
lsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1usequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"å
lsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2usequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"å
lsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3usequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"Ò
¥sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_shape§sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_shape_0"
¼sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg¾sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0"
Ësequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceÍsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0"Ô
æsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1èsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0"Ì
âsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2äsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ï9
Ö
G__inference_sequential_1_layer_call_and_return_conditional_losses_23805
sequential_input&
conv2d_23767:
conv2d_23769:(
conv2d_1_23773: 
conv2d_1_23775: (
conv2d_2_23779: @
conv2d_2_23781:@)
conv2d_3_23786:@
conv2d_3_23788:	
dense_23794:
d
dense_23796:	 
dense_1_23799:	
dense_1_23801:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCallÏ
sequential/PartitionedCallPartitionedCallsequential_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_22404à
rescaling/PartitionedCallPartitionedCall#sequential/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_23345
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_23767conv2d_23769*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23358ê
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23290
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_23773conv2d_1_23775*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23376ð
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23302
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_23779conv2d_2_23781*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23394ð
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_23314ß
dropout/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23406
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_3_23786conv2d_3_23788*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23419ñ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23326ä
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_23431Ò
flatten/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23439ü
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_23794dense_23796*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23452
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_23799dense_1_23801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_23468w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
*
_user_specified_namesequential_input
®
Ö
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_cond_24596
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counter
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterationsO
Ksequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholderQ
Msequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1
sequential_random_contrast_loop_body_adjust_contrast_pfor_while_less_sequential_random_contrast_loop_body_adjust_contrast_pfor_strided_slice¨
£sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_while_cond_24596___redundant_placeholder0¨
£sequential_random_contrast_loop_body_adjust_contrast_pfor_while_sequential_random_contrast_loop_body_adjust_contrast_pfor_while_cond_24596___redundant_placeholder1L
Hsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identity
É
Dsequential/random_contrast/loop_body/adjust_contrast/pfor/while/LessLessKsequential_random_contrast_loop_body_adjust_contrast_pfor_while_placeholdersequential_random_contrast_loop_body_adjust_contrast_pfor_while_less_sequential_random_contrast_loop_body_adjust_contrast_pfor_strided_slice*
T0*
_output_shapes
: ¿
Hsequential/random_contrast/loop_body/adjust_contrast/pfor/while/IdentityIdentityHsequential/random_contrast/loop_body/adjust_contrast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Hsequential_random_contrast_loop_body_adjust_contrast_pfor_while_identityQsequential/random_contrast/loop_body/adjust_contrast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
¹
C
'__inference_dropout_layer_call_fn_25791

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23406h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï
a
E__inference_sequential_layer_call_and_return_conditional_losses_22404

inputs
identityÏ
random_contrast/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_22395é
random_zoom/PartitionedCallPartitionedCall(random_contrast/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_22401v
IdentityIdentity$random_zoom/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¹
	
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_26306§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter­
¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsZ
Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice¾
¹loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_cond_26306___redundant_placeholder0	W
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity
õ
Oloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/LessLessVloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice*
T0*
_output_shapes
: Õ
Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitySloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Less:z:0*
T0
*
_output_shapes
: "³
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity\loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
¶
·
?random_contrast_loop_body_adjust_contrast_pfor_while_cond_25476z
vrandom_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_loop_counter
|random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_maximum_iterationsD
@random_contrast_loop_body_adjust_contrast_pfor_while_placeholderF
Brandom_contrast_loop_body_adjust_contrast_pfor_while_placeholder_1z
vrandom_contrast_loop_body_adjust_contrast_pfor_while_less_random_contrast_loop_body_adjust_contrast_pfor_strided_slice
random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_cond_25476___redundant_placeholder0
random_contrast_loop_body_adjust_contrast_pfor_while_random_contrast_loop_body_adjust_contrast_pfor_while_cond_25476___redundant_placeholder1A
=random_contrast_loop_body_adjust_contrast_pfor_while_identity

9random_contrast/loop_body/adjust_contrast/pfor/while/LessLess@random_contrast_loop_body_adjust_contrast_pfor_while_placeholdervrandom_contrast_loop_body_adjust_contrast_pfor_while_less_random_contrast_loop_body_adjust_contrast_pfor_strided_slice*
T0*
_output_shapes
: ©
=random_contrast/loop_body/adjust_contrast/pfor/while/IdentityIdentity=random_contrast/loop_body/adjust_contrast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
=random_contrast_loop_body_adjust_contrast_pfor_while_identityFrandom_contrast/loop_body/adjust_contrast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Á

'__inference_dense_1_layer_call_fn_25910

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCall×
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
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_23468o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
ì
psequential_random_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_24378Ý
Øsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counterã
Þsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsu
qsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderw
ssequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1w
ssequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Ý
Øsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_sliceô
ïsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_cond_24378___redundant_placeholder0	r
nsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity
á
jsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/LessLessqsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderØsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice*
T0*
_output_shapes
: 
nsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitynsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Less:z:0*
T0
*
_output_shapes
: "é
nsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identitywsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
÷:

Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_26018
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsM
Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderO
Kloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0
{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0:	n
jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0n
jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0J
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityL
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1L
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2L
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice
yloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	l
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_xl
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1¢Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipÏ
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkip{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0*
_output_shapes
:
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¬
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsTloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0Uloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:¢
bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemKloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderQloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Cloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Aloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/addAddV2Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderLloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Eloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :È
Cloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1AddV2loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counterNloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityGloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add_1:z:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ë
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsC^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2IdentityEloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/add:z:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ®
Hloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3Identityrloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0C^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ó
Bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/NoOpNoOpM^loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityOloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_1Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_1:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_2Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_2:output:0"
Hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_3Qloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity_3:output:0"Ö
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1jloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_1_0"Ö
hloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_xjloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_cast_x_0"
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_sliceloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice_0"ø
yloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource{loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2
Lloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkipLloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
õ
`
B__inference_dropout_layer_call_and_return_conditional_losses_25801

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
R
Ó
 __inference__wrapped_model_22384
sequential_inputL
2sequential_1_conv2d_conv2d_readvariableop_resource:A
3sequential_1_conv2d_biasadd_readvariableop_resource:N
4sequential_1_conv2d_1_conv2d_readvariableop_resource: C
5sequential_1_conv2d_1_biasadd_readvariableop_resource: N
4sequential_1_conv2d_2_conv2d_readvariableop_resource: @C
5sequential_1_conv2d_2_biasadd_readvariableop_resource:@O
4sequential_1_conv2d_3_conv2d_readvariableop_resource:@D
5sequential_1_conv2d_3_biasadd_readvariableop_resource:	E
1sequential_1_dense_matmul_readvariableop_resource:
dA
2sequential_1_dense_biasadd_readvariableop_resource:	F
3sequential_1_dense_1_matmul_readvariableop_resource:	B
4sequential_1_dense_1_biasadd_readvariableop_resource:
identity¢*sequential_1/conv2d/BiasAdd/ReadVariableOp¢)sequential_1/conv2d/Conv2D/ReadVariableOp¢,sequential_1/conv2d_1/BiasAdd/ReadVariableOp¢+sequential_1/conv2d_1/Conv2D/ReadVariableOp¢,sequential_1/conv2d_2/BiasAdd/ReadVariableOp¢+sequential_1/conv2d_2/Conv2D/ReadVariableOp¢,sequential_1/conv2d_3/BiasAdd/ReadVariableOp¢+sequential_1/conv2d_3/Conv2D/ReadVariableOp¢)sequential_1/dense/BiasAdd/ReadVariableOp¢(sequential_1/dense/MatMul/ReadVariableOp¢+sequential_1/dense_1/BiasAdd/ReadVariableOp¢*sequential_1/dense_1/MatMul/ReadVariableOpb
sequential_1/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;d
sequential_1/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
sequential_1/rescaling/mulMulsequential_input&sequential_1/rescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ©
sequential_1/rescaling/addAddV2sequential_1/rescaling/mul:z:0(sequential_1/rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¤
)sequential_1/conv2d/Conv2D/ReadVariableOpReadVariableOp2sequential_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Û
sequential_1/conv2d/Conv2DConv2Dsequential_1/rescaling/add:z:01sequential_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

*sequential_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_1/conv2d/BiasAddBiasAdd#sequential_1/conv2d/Conv2D:output:02sequential_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
sequential_1/conv2d/ReluRelu$sequential_1/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Â
"sequential_1/max_pooling2d/MaxPoolMaxPool&sequential_1/conv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
ksize
*
paddingVALID*
strides
¨
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ê
sequential_1/conv2d_1/Conv2DConv2D+sequential_1/max_pooling2d/MaxPool:output:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *
paddingSAME*
strides

,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0¿
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP 
sequential_1/conv2d_1/ReluRelu&sequential_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP Æ
$sequential_1/max_pooling2d_1/MaxPoolMaxPool(sequential_1/conv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( *
ksize
*
paddingVALID*
strides
¨
+sequential_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ì
sequential_1/conv2d_2/Conv2DConv2D-sequential_1/max_pooling2d_1/MaxPool:output:03sequential_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*
paddingSAME*
strides

,sequential_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0¿
sequential_1/conv2d_2/BiasAddBiasAdd%sequential_1/conv2d_2/Conv2D:output:04sequential_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@
sequential_1/conv2d_2/ReluRelu&sequential_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@Æ
$sequential_1/max_pooling2d_2/MaxPoolMaxPool(sequential_1/conv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

sequential_1/dropout/IdentityIdentity-sequential_1/max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@©
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0æ
sequential_1/conv2d_3/Conv2DConv2D&sequential_1/dropout/Identity:output:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0À
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
$sequential_1/max_pooling2d_3/MaxPoolMaxPool(sequential_1/conv2d_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingVALID*
strides

sequential_1/dropout_1/IdentityIdentity-sequential_1/max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

k
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 2  ©
sequential_1/flatten/ReshapeReshape(sequential_1/dropout_1/Identity:output:0#sequential_1/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
(sequential_1/dense/MatMul/ReadVariableOpReadVariableOp1sequential_1_dense_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0¯
sequential_1/dense/MatMulMatMul%sequential_1/flatten/Reshape:output:00sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0°
sequential_1/dense/BiasAddBiasAdd#sequential_1/dense/MatMul:product:01sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
sequential_1/dense/ReluRelu#sequential_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0²
sequential_1/dense_1/MatMulMatMul%sequential_1/dense/Relu:activations:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%sequential_1/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp+^sequential_1/conv2d/BiasAdd/ReadVariableOp*^sequential_1/conv2d/Conv2D/ReadVariableOp-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp-^sequential_1/conv2d_2/BiasAdd/ReadVariableOp,^sequential_1/conv2d_2/Conv2D/ReadVariableOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp*^sequential_1/dense/BiasAdd/ReadVariableOp)^sequential_1/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 2X
*sequential_1/conv2d/BiasAdd/ReadVariableOp*sequential_1/conv2d/BiasAdd/ReadVariableOp2V
)sequential_1/conv2d/Conv2D/ReadVariableOp)sequential_1/conv2d/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_2/BiasAdd/ReadVariableOp,sequential_1/conv2d_2/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_2/Conv2D/ReadVariableOp+sequential_1/conv2d_2/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2V
)sequential_1/dense/BiasAdd/ReadVariableOp)sequential_1/dense/BiasAdd/ReadVariableOp2T
(sequential_1/dense/MatMul/ReadVariableOp(sequential_1/dense/MatMul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
*
_user_specified_namesequential_input

ô
9loop_body_stateful_uniform_full_int_pfor_while_cond_26206n
jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_countert
ploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations>
:loop_body_stateful_uniform_full_int_pfor_while_placeholder@
<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1n
jloop_body_stateful_uniform_full_int_pfor_while_less_loop_body_stateful_uniform_full_int_pfor_strided_slice
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_26206___redundant_placeholder0
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_26206___redundant_placeholder1
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_26206___redundant_placeholder2
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_26206___redundant_placeholder3;
7loop_body_stateful_uniform_full_int_pfor_while_identity

3loop_body/stateful_uniform_full_int/pfor/while/LessLess:loop_body_stateful_uniform_full_int_pfor_while_placeholderjloop_body_stateful_uniform_full_int_pfor_while_less_loop_body_stateful_uniform_full_int_pfor_strided_slice*
T0*
_output_shapes
: 
7loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentity7loop_body/stateful_uniform_full_int/pfor/while/Less:z:0*
T0
*
_output_shapes
: "{
7loop_body_stateful_uniform_full_int_pfor_while_identity@loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
Ð
Ò
/loop_body_adjust_contrast_pfor_while_cond_23118Z
Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter`
\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations4
0loop_body_adjust_contrast_pfor_while_placeholder6
2loop_body_adjust_contrast_pfor_while_placeholder_1Z
Vloop_body_adjust_contrast_pfor_while_less_loop_body_adjust_contrast_pfor_strided_sliceq
mloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_cond_23118___redundant_placeholder0q
mloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_cond_23118___redundant_placeholder11
-loop_body_adjust_contrast_pfor_while_identity
Ü
)loop_body/adjust_contrast/pfor/while/LessLess0loop_body_adjust_contrast_pfor_while_placeholderVloop_body_adjust_contrast_pfor_while_less_loop_body_adjust_contrast_pfor_strided_slice*
T0*
_output_shapes
: 
-loop_body/adjust_contrast/pfor/while/IdentityIdentity-loop_body/adjust_contrast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "g
-loop_body_adjust_contrast_pfor_while_identity6loop_body/adjust_contrast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:

b
F__inference_random_zoom_layer_call_and_return_conditional_losses_26649

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
û
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_23431

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs

ª
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_22969
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counter¥
 loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsV
Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderX
Tloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_22969___redundant_placeholder0¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_22969___redundant_placeholder1¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_22969___redundant_placeholder2¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_22969___redundant_placeholder3S
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity
å
Kloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/LessLessRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: Í
Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityOloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "«
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityXloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
ì

&__inference_conv2d_layer_call_fn_25705

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23358y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
²

Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_25158
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counter
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterationsN
Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderP
Lrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice¦
¡random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_25158___redundant_placeholder0¦
¡random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_25158___redundant_placeholder1¦
¡random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_25158___redundant_placeholder2¦
¡random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_25158___redundant_placeholder3K
Grandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity
Å
Crandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/LessLessJrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice*
T0*
_output_shapes
: ½
Grandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentityGrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Grandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identityPrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:

b
F__inference_random_zoom_layer_call_and_return_conditional_losses_22401

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
9


Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_22677~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsF
Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderH
Dloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1{
wloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0	C
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityE
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1E
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2E
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3y
uloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice	~
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ï
:loop_body/stateful_uniform_full_int/Bitcast/pfor/while/addAddV2Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderEloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stackPackBloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderUloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Nloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1Pack>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add:z:0Wloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Lloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ã
Dloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0Sloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack:output:0Uloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1:output:0Uloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÉ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/BitcastBitcastMloop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Eloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims
ExpandDimsGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Bitcast:output:0Nloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
[loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemDloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderJloop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ó
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1AddV2Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
>loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :«
<loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2AddV2zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counterGloop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ®
?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity@loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ñ
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: °
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2Identity@loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: Û
Aloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3Identitykloop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityHloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2:output:0"
Aloop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3Jloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3:output:0"ð
uloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slicewloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0"
loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedsliceloop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
±l
î
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_body_24279¥
 sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counter«
¦sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterationsY
Usequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder[
Wsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1¢
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice_0¿
ºsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0¿
ºsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_shape_0
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_alg_0V
Rsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identityX
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_1X
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_2X
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_3 
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice½
¸sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2½
¸sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_shape
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_alg
Osequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¨
Msequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/addAddV2Usequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderXsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¡
_sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ô
]sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stackPackUsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderhsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:£
asequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ô
_sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1PackQsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add:z:0jsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:°
_sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¿
Wsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_sliceStridedSliceºsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0fsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack:output:0hsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1:output:0hsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Qsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¬
Osequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1AddV2Usequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderZsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: £
asequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ø
_sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stackPackUsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderjsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:¥
csequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ú
asequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1PackSsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1:z:0lsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:²
asequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1StridedSliceºsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0hsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack:output:0jsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1:output:0jsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask¨
isequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2StatelessRandomUniformFullIntV2sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_shape_0`sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice:output:0bsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1:output:0sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0	
Xsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ë
Tsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims
ExpandDimsrsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2:output:0asequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
nsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemWsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1Usequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder]sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Qsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :¬
Osequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2AddV2Usequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderZsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Qsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ø
Osequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3AddV2 sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counterZsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: Ô
Rsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentitySsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3:z:0*
T0*
_output_shapes
: ª
Tsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_1Identity¦sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ö
Tsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_2IdentitySsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2:z:0*
T0*
_output_shapes
: 
Tsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_3Identity~sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "±
Rsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity[sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0"µ
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_1]sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_1:output:0"µ
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_2]sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_2:output:0"µ
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_3]sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_3:output:0" 
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_algsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_alg_0"¾
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slicesequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice_0"¤
sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_shapesequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_shape_0"ø
¸sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2ºsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0"ø
¸sequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2ºsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
­
á	
Hloop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_22611
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterationsM
Iloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderO
Kloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_22611___redundant_placeholder0¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_22611___redundant_placeholder1¤
loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_22611___redundant_placeholder2J
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity
Á
Bloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/LessLessIloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholderloop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: »
Floop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityFloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Floop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identityOloop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
¦
»
Aloop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_26082~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsF
Bloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderH
Dloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1~
zloop_body_stateful_uniform_full_int_bitcast_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_body_stateful_uniform_full_int_bitcast_pfor_while_cond_26082___redundant_placeholder0	C
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity
¤
;loop_body/stateful_uniform_full_int/Bitcast/pfor/while/LessLessBloop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderzloop_body_stateful_uniform_full_int_bitcast_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice*
T0*
_output_shapes
: ­
?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity?loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
?loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityHloop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
Æ
Ø
,__inference_sequential_1_layer_call_fn_23502
sequential_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:
d
	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_23475o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
*
_user_specified_namesequential_input
Ç
F
*__inference_sequential_layer_call_fn_24875

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_22404j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ö
`
D__inference_rescaling_layer_call_and_return_conditional_losses_23345

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
À

lsequential_random_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_24447Õ
Ðsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterÛ
Ösequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsq
msequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholders
osequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Õ
Ðsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceì
çsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_24447___redundant_placeholder0ì
çsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_24447___redundant_placeholder1ì
çsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_24447___redundant_placeholder2ì
çsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_24447___redundant_placeholder3n
jsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity
Ñ
fsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/LessLessmsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderÐsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_sequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: 
jsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityjsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "á
jsequential_random_contrast_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityssequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
É
G
+__inference_random_zoom_layer_call_fn_26638

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_random_zoom_layer_call_and_return_conditional_losses_22401j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23302

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
n
¿
F__inference_random_zoom_layer_call_and_return_conditional_losses_22517

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ð
Õ
Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_26149
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsH
Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderJ
Floop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_cond_26149___redundant_placeholder0	E
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity
¬
=loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/LessLessDloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: ±
Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityAloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityJloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
àD
í
Srandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_25102£
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter©
¤random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsX
Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderZ
Vrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1 
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0³
®random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0	U
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityW
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1W
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2W
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice±
¬random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice	
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¥
Lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/addAddV2Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderWrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
:  
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ñ
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stackPackTrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholdergrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:¢
`random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ñ
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1PackPrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add:z:0irandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¯
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¯
Vrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_sliceStridedSlice®random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0erandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack:output:0grandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1:output:0grandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskí
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/BitcastBitcast_random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Wrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ç
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims
ExpandDimsYrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Bitcast:output:0`random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Î
mrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemVrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :©
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1AddV2Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderYrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Prandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ô
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2AddV2random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counterYrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: Ò
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityRrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: §
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1Identity¤random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ô
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2IdentityRrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: ÿ
Srandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3Identity}random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityZrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0"³
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1:output:0"³
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2:output:0"³
Srandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3\random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3:output:0"º
random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slicerandom_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0"à
¬random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice®random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ü
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23394

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ(( : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( 
 
_user_specified_nameinputs
®>
å
/loop_body_adjust_contrast_pfor_while_body_23119Z
Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter`
\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations4
0loop_body_adjust_contrast_pfor_while_placeholder6
2loop_body_adjust_contrast_pfor_while_placeholder_1W
Sloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_slice_0Y
Uloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0h
dloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_01
-loop_body_adjust_contrast_pfor_while_identity3
/loop_body_adjust_contrast_pfor_while_identity_13
/loop_body_adjust_contrast_pfor_while_identity_23
/loop_body_adjust_contrast_pfor_while_identity_3U
Qloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_sliceW
Sloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2f
bloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2l
*loop_body/adjust_contrast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¹
(loop_body/adjust_contrast/pfor/while/addAddV20loop_body_adjust_contrast_pfor_while_placeholder3loop_body/adjust_contrast/pfor/while/add/y:output:0*
T0*
_output_shapes
: |
:loop_body/adjust_contrast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : å
8loop_body/adjust_contrast/pfor/while/strided_slice/stackPack0loop_body_adjust_contrast_pfor_while_placeholderCloop_body/adjust_contrast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:~
<loop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : å
:loop_body/adjust_contrast/pfor/while/strided_slice/stack_1Pack,loop_body/adjust_contrast/pfor/while/add:z:0Eloop_body/adjust_contrast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
:loop_body/adjust_contrast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
2loop_body/adjust_contrast/pfor/while/strided_sliceStridedSliceUloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0Aloop_body/adjust_contrast/pfor/while/strided_slice/stack:output:0Cloop_body/adjust_contrast/pfor/while/strided_slice/stack_1:output:0Cloop_body/adjust_contrast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*$
_output_shapes
:  *
ellipsis_mask*
shrink_axis_maskn
,loop_body/adjust_contrast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :½
*loop_body/adjust_contrast/pfor/while/add_1AddV20loop_body_adjust_contrast_pfor_while_placeholder5loop_body/adjust_contrast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ~
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : é
:loop_body/adjust_contrast/pfor/while/strided_slice_1/stackPack0loop_body_adjust_contrast_pfor_while_placeholderEloop_body/adjust_contrast/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
>loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ë
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1Pack.loop_body/adjust_contrast/pfor/while/add_1:z:0Gloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
<loop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
4loop_body/adjust_contrast/pfor/while/strided_slice_1StridedSlicedloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_0Cloop_body/adjust_contrast/pfor/while/strided_slice_1/stack:output:0Eloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_1:output:0Eloop_body/adjust_contrast/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
ellipsis_mask*
shrink_axis_maskë
5loop_body/adjust_contrast/pfor/while/AdjustContrastv2AdjustContrastv2;loop_body/adjust_contrast/pfor/while/strided_slice:output:0=loop_body/adjust_contrast/pfor/while/strided_slice_1:output:0*$
_output_shapes
:  u
3loop_body/adjust_contrast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : î
/loop_body/adjust_contrast/pfor/while/ExpandDims
ExpandDims>loop_body/adjust_contrast/pfor/while/AdjustContrastv2:output:0<loop_body/adjust_contrast/pfor/while/ExpandDims/dim:output:0*
T0*(
_output_shapes
:  ¾
Iloop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem2loop_body_adjust_contrast_pfor_while_placeholder_10loop_body_adjust_contrast_pfor_while_placeholder8loop_body/adjust_contrast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒn
,loop_body/adjust_contrast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :½
*loop_body/adjust_contrast/pfor/while/add_2AddV20loop_body_adjust_contrast_pfor_while_placeholder5loop_body/adjust_contrast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: n
,loop_body/adjust_contrast/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ã
*loop_body/adjust_contrast/pfor/while/add_3AddV2Vloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_loop_counter5loop_body/adjust_contrast/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
-loop_body/adjust_contrast/pfor/while/IdentityIdentity.loop_body/adjust_contrast/pfor/while/add_3:z:0*
T0*
_output_shapes
: º
/loop_body/adjust_contrast/pfor/while/Identity_1Identity\loop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_while_maximum_iterations*
T0*
_output_shapes
: 
/loop_body/adjust_contrast/pfor/while/Identity_2Identity.loop_body/adjust_contrast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ·
/loop_body/adjust_contrast/pfor/while/Identity_3IdentityYloop_body/adjust_contrast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "g
-loop_body_adjust_contrast_pfor_while_identity6loop_body/adjust_contrast/pfor/while/Identity:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_18loop_body/adjust_contrast/pfor/while/Identity_1:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_28loop_body/adjust_contrast/pfor/while/Identity_2:output:0"k
/loop_body_adjust_contrast_pfor_while_identity_38loop_body/adjust_contrast/pfor/while/Identity_3:output:0"¨
Qloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_sliceSloop_body_adjust_contrast_pfor_while_loop_body_adjust_contrast_pfor_strided_slice_0"Ê
bloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2dloop_body_adjust_contrast_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_pfor_addv2_0"¬
Sloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2Uloop_body_adjust_contrast_pfor_while_strided_slice_loop_body_gatherv2_pfor_gatherv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  :)%
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¿g
Ë
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_22970
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counter¥
 loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsV
Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderX
Tloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0³
®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0·
²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0u
qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0S
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityU
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1U
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2U
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice±
¬loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2µ
°loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1s
oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Jloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/addAddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderUloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdereloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
: 
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1PackNloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add:z:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:­
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      §
Tloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSlice®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0cloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
:  
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ï
\loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholdergloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:¢
`loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ñ
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1PackPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0iloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¯
^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
Vloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSlice²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0eloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0gloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÑ
_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0]loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0_loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Î
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDimshloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0^loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes
:Æ
kloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemTloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderZloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Nloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ì
Lloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counterWloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: Î
Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: ¡
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1Identity loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ð
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2IdentityPloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: û
Qloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identity{loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "«
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityXloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_1Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_2Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"¯
Qloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity_3Zloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"ä
oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shapeqloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_shape_0"
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_algloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_alg_0"²
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_sliceloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice_0"è
°loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1²loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_1_0"à
¬loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2®loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
çK
¾
^sequential_random_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_24222¹
´sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter¿
ºsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsc
_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholdere
asequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1¶
±sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0É
Äsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0	`
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityb
^sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1b
^sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2b
^sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3´
¯sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_sliceÇ
Âsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice	
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Æ
Wsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/addAddV2_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderbsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
: «
isequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ò
gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stackPack_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderrsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:­
ksequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ò
isequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1Pack[sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add:z:0tsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:º
isequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ñ
asequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_sliceStridedSliceÄsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0psequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack:output:0rsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1:output:0rsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
[sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/BitcastBitcastjsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0¤
bsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : è
^sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims
ExpandDimsdsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Bitcast:output:0ksequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:ú
xsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemasequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholdergsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
[sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ê
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1AddV2_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderdsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
[sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B : 
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2AddV2´sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counterdsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: è
\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentity]sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: È
^sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1Identityºsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: ê
^sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2Identity]sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: 
^sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3Identitysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "Å
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityesequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0"É
^sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1:output:0"É
^sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2:output:0"É
^sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3:output:0"æ
¯sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice±sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0"
Âsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedsliceÄsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ï
	
^sequential_random_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_24221¹
´sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter¿
ºsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsc
_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholdere
asequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1¹
´sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_sliceÐ
Ësequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_cond_24221___redundant_placeholder0	`
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity

Xsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/LessLess_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder´sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_less_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: ç
\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentity\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "Å
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityesequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
¶
K
/__inference_max_pooling2d_3_layer_call_fn_25838

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23326
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è

(__inference_conv2d_1_layer_call_fn_25735

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23376w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿPP: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP
 
_user_specified_nameinputs
ó
Ñ
G__inference_sequential_1_layer_call_and_return_conditional_losses_24870

inputsd
Vsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resource:	M
?sequential_random_zoom_stateful_uniform_rngreadandskip_resource:	?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@7
(conv2d_3_biasadd_readvariableop_resource:	8
$dense_matmul_readvariableop_resource:
d4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢Msequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip¢Xsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while¢6sequential/random_zoom/stateful_uniform/RngReadAndSkipV
 sequential/random_contrast/ShapeShapeinputs*
T0*
_output_shapes
:x
.sequential/random_contrast/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential/random_contrast/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential/random_contrast/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
(sequential/random_contrast/strided_sliceStridedSlice)sequential/random_contrast/Shape:output:07sequential/random_contrast/strided_slice/stack:output:09sequential/random_contrast/strided_slice/stack_1:output:09sequential/random_contrast/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&sequential/random_contrast/Rank/packedPack1sequential/random_contrast/strided_slice:output:0*
N*
T0*
_output_shapes
:a
sequential/random_contrast/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&sequential/random_contrast/range/startConst*
_output_shapes
: *
dtype0*
value	B : h
&sequential/random_contrast/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Ñ
 sequential/random_contrast/rangeRange/sequential/random_contrast/range/start:output:0(sequential/random_contrast/Rank:output:0/sequential/random_contrast/range/delta:output:0*
_output_shapes
:
$sequential/random_contrast/Max/inputPack1sequential/random_contrast/strided_slice:output:0*
N*
T0*
_output_shapes
: 
sequential/random_contrast/MaxMax-sequential/random_contrast/Max/input:output:0)sequential/random_contrast/range:output:0*
T0*
_output_shapes
: 
Asequential/random_contrast/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : Ó
;sequential/random_contrast/loop_body/PlaceholderWithDefaultPlaceholderWithDefaultJsequential/random_contrast/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: `
*sequential/random_contrast/loop_body/ShapeShapeinputs*
T0*
_output_shapes
:
8sequential/random_contrast/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:sequential/random_contrast/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:sequential/random_contrast/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2sequential/random_contrast/loop_body/strided_sliceStridedSlice3sequential/random_contrast/loop_body/Shape:output:0Asequential/random_contrast/loop_body/strided_slice/stack:output:0Csequential/random_contrast/loop_body/strided_slice/stack_1:output:0Csequential/random_contrast/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.sequential/random_contrast/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :Î
,sequential/random_contrast/loop_body/GreaterGreater;sequential/random_contrast/loop_body/strided_slice:output:07sequential/random_contrast/loop_body/Greater/y:output:0*
T0*
_output_shapes
: q
/sequential/random_contrast/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : 
-sequential/random_contrast/loop_body/SelectV2SelectV20sequential/random_contrast/loop_body/Greater:z:0Dsequential/random_contrast/loop_body/PlaceholderWithDefault:output:08sequential/random_contrast/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: t
2sequential/random_contrast/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
-sequential/random_contrast/loop_body/GatherV2GatherV2inputs6sequential/random_contrast/loop_body/SelectV2:output:0;sequential/random_contrast/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:  
Dsequential/random_contrast/loop_body/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/random_contrast/loop_body/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Csequential/random_contrast/loop_body/stateful_uniform_full_int/ProdProdMsequential/random_contrast/loop_body/stateful_uniform_full_int/shape:output:0Msequential/random_contrast/loop_body/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 
Esequential/random_contrast/loop_body/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :Ë
Esequential/random_contrast/loop_body/stateful_uniform_full_int/Cast_1CastLsequential/random_contrast/loop_body/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: î
Msequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipVsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resourceNsequential/random_contrast/loop_body/stateful_uniform_full_int/Cast/x:output:0Isequential/random_contrast/loop_body/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Rsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Tsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Tsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_sliceStridedSliceUsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0[sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack:output:0]sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0]sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÙ
Fsequential/random_contrast/loop_body/stateful_uniform_full_int/BitcastBitcastUsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Tsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Vsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Vsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Nsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1StridedSliceUsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip:value:0]sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0_sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0_sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ý
Hsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1BitcastWsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Bsequential/random_contrast/loop_body/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :Ê
>sequential/random_contrast/loop_body/stateful_uniform_full_intStatelessRandomUniformFullIntV2Msequential/random_contrast/loop_body/stateful_uniform_full_int/shape:output:0Qsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1:output:0Osequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast:output:0Ksequential/random_contrast/loop_body/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	y
/sequential/random_contrast/loop_body/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ç
*sequential/random_contrast/loop_body/stackPackGsequential/random_contrast/loop_body/stateful_uniform_full_int:output:08sequential/random_contrast/loop_body/zeros_like:output:0*
N*
T0	*
_output_shapes

:
:sequential/random_contrast/loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
<sequential/random_contrast/loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
<sequential/random_contrast/loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
4sequential/random_contrast/loop_body/strided_slice_1StridedSlice3sequential/random_contrast/loop_body/stack:output:0Csequential/random_contrast/loop_body/strided_slice_1/stack:output:0Esequential/random_contrast/loop_body/strided_slice_1/stack_1:output:0Esequential/random_contrast/loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
Csequential/random_contrast/loop_body/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 
Asequential/random_contrast/loop_body/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
Asequential/random_contrast/loop_body/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?Û
Zsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter=sequential/random_contrast/loop_body/strided_slice_1:output:0* 
_output_shapes
::
Zsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
Vsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Lsequential/random_contrast/loop_body/stateless_random_uniform/shape:output:0`sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0dsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0csequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: 
Asequential/random_contrast/loop_body/stateless_random_uniform/subSubJsequential/random_contrast/loop_body/stateless_random_uniform/max:output:0Jsequential/random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
Asequential/random_contrast/loop_body/stateless_random_uniform/mulMul_sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2:output:0Esequential/random_contrast/loop_body/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ú
=sequential/random_contrast/loop_body/stateless_random_uniformAddV2Esequential/random_contrast/loop_body/stateless_random_uniform/mul:z:0Jsequential/random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: é
4sequential/random_contrast/loop_body/adjust_contrastAdjustContrastv26sequential/random_contrast/loop_body/GatherV2:output:0Asequential/random_contrast/loop_body/stateless_random_uniform:z:0*$
_output_shapes
:  ·
=sequential/random_contrast/loop_body/adjust_contrast/IdentityIdentity=sequential/random_contrast/loop_body/adjust_contrast:output:0*
T0*$
_output_shapes
:  
<sequential/random_contrast/loop_body/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
:sequential/random_contrast/loop_body/clip_by_value/MinimumMinimumFsequential/random_contrast/loop_body/adjust_contrast/Identity:output:0Esequential/random_contrast/loop_body/clip_by_value/Minimum/y:output:0*
T0*$
_output_shapes
:  y
4sequential/random_contrast/loop_body/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ë
2sequential/random_contrast/loop_body/clip_by_valueMaximum>sequential/random_contrast/loop_body/clip_by_value/Minimum:z:0=sequential/random_contrast/loop_body/clip_by_value/y:output:0*
T0*$
_output_shapes
:  w
-sequential/random_contrast/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¸
'sequential/random_contrast/pfor/ReshapeReshape'sequential/random_contrast/Max:output:06sequential/random_contrast/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:m
+sequential/random_contrast/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : m
+sequential/random_contrast/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :è
%sequential/random_contrast/pfor/rangeRange4sequential/random_contrast/pfor/range/start:output:0'sequential/random_contrast/Max:output:04sequential/random_contrast/pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
fsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ²
hsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:²
hsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
`sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_sliceStridedSlice0sequential/random_contrast/pfor/Reshape:output:0osequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack:output:0qsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_1:output:0qsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¹
nsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ«
`sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2TensorListReservewsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0isequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
Xsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¶
ksequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ§
esequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
Xsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileWhilensequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/loop_counter:output:0tsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/maximum_iterations:output:0asequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const:output:0isequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorArrayV2:handle:0isequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/strided_slice:output:0Vsequential_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_resourceNsequential/random_contrast/loop_body/stateful_uniform_full_int/Cast/x:output:0Isequential/random_contrast/loop_body/stateful_uniform_full_int/Cast_1:y:0N^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *o
bodygRe
csequential_random_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_body_24090*o
condgRe
csequential_random_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_24089*#
output_shapes
: : : : : : : : 
Zsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 Ä
ssequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
esequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2asequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while:output:3|sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0csequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0«
asequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
]sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ã
Xsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concatConcatV2jsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/values_0:output:0[sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack:output:0fsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:­
csequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ¡
_sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
Zsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1ConcatV2lsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/values_0:output:0]sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_1:output:0hsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
csequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:¡
_sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
Zsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2ConcatV2lsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/values_0:output:0]sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/stack_2:output:0hsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:ì
^sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSliceStridedSlicensequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0asequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat:output:0csequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_1:output:0csequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask©
_sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: «
asequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:«
asequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_sliceStridedSlice0sequential/random_contrast/pfor/Reshape:output:0hsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack:output:0jsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_1:output:0jsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask²
gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2TensorListReservepsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2/element_shape:output:0bsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Qsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¯
dsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ 
^sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
Qsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/whileStatelessWhilegsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/loop_counter:output:0msequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/maximum_iterations:output:0Zsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const:output:0bsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorArrayV2:handle:0bsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/strided_slice:output:0gsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *h
body`R^
\sequential_random_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_24155*h
cond`R^
\sequential_random_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_cond_24154*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ
Ssequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ½
lsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
^sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2TensorListConcatV2Zsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while:output:3usequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2/element_shape:output:0\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0­
csequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: ¡
_sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
Zsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concatConcatV2lsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/values_0:output:0]sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack:output:0hsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¯
esequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: £
asequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
\sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1ConcatV2nsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/values_0:output:0_sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_1:output:0jsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:¯
esequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:£
asequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
\sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2ConcatV2nsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/values_0:output:0_sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/stack_2:output:0jsequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:ô
`sequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSliceStridedSlicensequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0csequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat:output:0esequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_1:output:0esequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask«
asequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ­
csequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:­
csequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
[sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_sliceStridedSlice0sequential/random_contrast/pfor/Reshape:output:0jsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack:output:0lsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_1:output:0lsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask´
isequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
[sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2TensorListReserversequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0dsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
Ssequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ±
fsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¢
`sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¡	
Ssequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/whileStatelessWhileisequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/loop_counter:output:0osequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/maximum_iterations:output:0\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const:output:0dsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorArrayV2:handle:0dsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/strided_slice:output:0isequential/random_contrast/loop_body/stateful_uniform_full_int/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *j
bodybR`
^sequential_random_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_24222*j
condbR`
^sequential_random_contrast_loop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_cond_24221*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ
Usequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ¿
nsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
`sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV2\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while:output:3wsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0^sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0¡
Wsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: £
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:£
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Qsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_sliceStridedSlice0sequential/random_contrast/pfor/Reshape:output:0`sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack:output:0bsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_1:output:0bsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskª
_sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿþ
Qsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2TensorListReservehsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2/element_shape:output:0Zsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
Isequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : §
\sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Vsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
Isequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/whileStatelessWhile_sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/loop_counter:output:0esequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while/maximum_iterations:output:0Rsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/Const:output:0Zsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorArrayV2:handle:0Zsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/strided_slice:output:0isequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast_1/pfor/TensorListConcatV2:tensor:0gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/TensorListConcatV2:tensor:0Msequential/random_contrast/loop_body/stateful_uniform_full_int/shape:output:0Ksequential/random_contrast/loop_body/stateful_uniform_full_int/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *`
bodyXRV
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_body_24279*`
condXRV
Tsequential_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_cond_24278*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 
Ksequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 µ
dsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿä
Vsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2TensorListConcatV2Rsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/while:output:3msequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2/element_shape:output:0Tsequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0
5sequential/random_contrast/loop_body/stack/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
Osequential/random_contrast/loop_body/stack/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
?sequential/random_contrast/loop_body/stack/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :
9sequential/random_contrast/loop_body/stack/pfor/ones_likeFillXsequential/random_contrast/loop_body/stack/pfor/ones_like/Shape/shape_as_tensor:output:0Hsequential/random_contrast/loop_body/stack/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:
=sequential/random_contrast/loop_body/stack/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿó
7sequential/random_contrast/loop_body/stack/pfor/ReshapeReshapeBsequential/random_contrast/loop_body/stack/pfor/ones_like:output:0Fsequential/random_contrast/loop_body/stack/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?sequential/random_contrast/loop_body/stack/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿå
9sequential/random_contrast/loop_body/stack/pfor/Reshape_1Reshape0sequential/random_contrast/pfor/Reshape:output:0Hsequential/random_contrast/loop_body/stack/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:}
;sequential/random_contrast/loop_body/stack/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
6sequential/random_contrast/loop_body/stack/pfor/concatConcatV2Bsequential/random_contrast/loop_body/stack/pfor/Reshape_1:output:0@sequential/random_contrast/loop_body/stack/pfor/Reshape:output:0Dsequential/random_contrast/loop_body/stack/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
>sequential/random_contrast/loop_body/stack/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ô
:sequential/random_contrast/loop_body/stack/pfor/ExpandDims
ExpandDims8sequential/random_contrast/loop_body/zeros_like:output:0Gsequential/random_contrast/loop_body/stack/pfor/ExpandDims/dim:output:0*
T0	*
_output_shapes

:ô
4sequential/random_contrast/loop_body/stack/pfor/TileTileCsequential/random_contrast/loop_body/stack/pfor/ExpandDims:output:0?sequential/random_contrast/loop_body/stack/pfor/concat:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
5sequential/random_contrast/loop_body/stack/pfor/stackPack_sequential/random_contrast/loop_body/stateful_uniform_full_int/pfor/TensorListConcatV2:tensor:0=sequential/random_contrast/loop_body/stack/pfor/Tile:output:0*
N*
T0	*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

axis
Isequential/random_contrast/loop_body/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Esequential/random_contrast/loop_body/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
@sequential/random_contrast/loop_body/strided_slice_1/pfor/concatConcatV2Rsequential/random_contrast/loop_body/strided_slice_1/pfor/concat/values_0:output:0Csequential/random_contrast/loop_body/strided_slice_1/stack:output:0Nsequential/random_contrast/loop_body/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Ksequential/random_contrast/loop_body/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Gsequential/random_contrast/loop_body/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
Bsequential/random_contrast/loop_body/strided_slice_1/pfor/concat_1ConcatV2Tsequential/random_contrast/loop_body/strided_slice_1/pfor/concat_1/values_0:output:0Esequential/random_contrast/loop_body/strided_slice_1/stack_1:output:0Psequential/random_contrast/loop_body/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Ksequential/random_contrast/loop_body/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Gsequential/random_contrast/loop_body/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ë
Bsequential/random_contrast/loop_body/strided_slice_1/pfor/concat_2ConcatV2Tsequential/random_contrast/loop_body/strided_slice_1/pfor/concat_2/values_0:output:0Esequential/random_contrast/loop_body/strided_slice_1/stack_2:output:0Psequential/random_contrast/loop_body/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:ô
Fsequential/random_contrast/loop_body/strided_slice_1/pfor/StridedSliceStridedSlice>sequential/random_contrast/loop_body/stack/pfor/stack:output:0Isequential/random_contrast/loop_body/strided_slice_1/pfor/concat:output:0Ksequential/random_contrast/loop_body/strided_slice_1/pfor/concat_1:output:0Ksequential/random_contrast/loop_body/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_mask½
ssequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¿
usequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¿
usequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ó
msequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_sliceStridedSlice0sequential/random_contrast/pfor/Reshape:output:0|sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack:output:0~sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_1:output:0~sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÆ
{sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÓ
msequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2TensorListReservesequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2/element_shape:output:0vsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌÈ
}sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ×
osequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1TensorListReservesequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1/element_shape:output:0vsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ§
esequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : Ã
xsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
rsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
esequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/whileStatelessWhile{sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/loop_counter:output:0sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/maximum_iterations:output:0nsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const:output:0vsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2:handle:0xsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorArrayV2_1:handle:0vsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/strided_slice:output:0Osequential/random_contrast/loop_body/strided_slice_1/pfor/StridedSlice:output:0*
T
	2	*
_lower_using_switch_merge(*
_num_original_outputs*3
_output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *|
bodytRr
psequential_random_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_24379*|
condtRr
psequential_random_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_24378*2
output_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿª
gsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 Ò
sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Õ
rsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2TensorListConcatV2nsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:3sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2/element_shape:output:0psequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0ª
gsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 Ô
sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ù
tsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1TensorListConcatV2nsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while:output:4sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1/element_shape:output:0psequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/Const_2:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0¹
osequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: »
qsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:»
qsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
isequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlice0sequential/random_contrast/pfor/Reshape:output:0xsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0zsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0zsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÂ
wsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
isequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReservesequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0rsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ£
asequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¿
tsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ°
nsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
asequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhilewsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0}sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0jsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const:output:0rsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0rsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0{sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2:tensor:0}sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/TensorListConcatV2_1:tensor:0Lsequential/random_contrast/loop_body/stateless_random_uniform/shape:output:0csequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *x
bodypRn
lsequential_random_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_body_24448*x
condpRn
lsequential_random_contrast_loop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_24447*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : ¦
csequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 Ï
|sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÁ
nsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2jsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while:output:3sequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0lsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Ksequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
Msequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
Lsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :£
Jsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/addAddV2Vsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank_1:output:0Usequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
:  
Nsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/MaximumMaximumNsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/add:z:0Tsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: ó
Lsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/ShapeShapewsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:
Jsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/subSubRsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Maximum:z:0Tsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
Tsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:­
Nsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/ReshapeReshapeNsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/sub:z:0]sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Qsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:«
Ksequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/TileTileZsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile/input:output:0Wsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: ¤
Zsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¦
\sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¦
\sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:²
Tsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_sliceStridedSliceUsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Shape:output:0csequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack:output:0esequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_1:output:0esequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask¦
\sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¨
^sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ¨
^sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
Vsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1StridedSliceUsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Shape:output:0esequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack:output:0gsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_1:output:0gsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask
Rsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ú
Msequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/concatConcatV2]sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice:output:0Tsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Tile:output:0_sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/strided_slice_1:output:0[sequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ú
Psequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape_1Reshapewsequential/random_contrast/loop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0Vsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
Jsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/MulMulYsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Reshape_1:output:0Esequential/random_contrast/loop_body/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Gsequential/random_contrast/loop_body/stateless_random_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
Isequential/random_contrast/loop_body/stateless_random_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
Hsequential/random_contrast/loop_body/stateless_random_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Fsequential/random_contrast/loop_body/stateless_random_uniform/pfor/addAddV2Rsequential/random_contrast/loop_body/stateless_random_uniform/pfor/Rank_1:output:0Qsequential/random_contrast/loop_body/stateless_random_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: 
Jsequential/random_contrast/loop_body/stateless_random_uniform/pfor/MaximumMaximumJsequential/random_contrast/loop_body/stateless_random_uniform/pfor/add:z:0Psequential/random_contrast/loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: Æ
Hsequential/random_contrast/loop_body/stateless_random_uniform/pfor/ShapeShapeNsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:
Fsequential/random_contrast/loop_body/stateless_random_uniform/pfor/subSubNsequential/random_contrast/loop_body/stateless_random_uniform/pfor/Maximum:z:0Psequential/random_contrast/loop_body/stateless_random_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
Psequential/random_contrast/loop_body/stateless_random_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¡
Jsequential/random_contrast/loop_body/stateless_random_uniform/pfor/ReshapeReshapeJsequential/random_contrast/loop_body/stateless_random_uniform/pfor/sub:z:0Ysequential/random_contrast/loop_body/stateless_random_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Msequential/random_contrast/loop_body/stateless_random_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
Gsequential/random_contrast/loop_body/stateless_random_uniform/pfor/TileTileVsequential/random_contrast/loop_body/stateless_random_uniform/pfor/Tile/input:output:0Ssequential/random_contrast/loop_body/stateless_random_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
:  
Vsequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¢
Xsequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¢
Xsequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Psequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_sliceStridedSliceQsequential/random_contrast/loop_body/stateless_random_uniform/pfor/Shape:output:0_sequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack:output:0asequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_1:output:0asequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask¢
Xsequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¤
Zsequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ¤
Zsequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
Rsequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1StridedSliceQsequential/random_contrast/loop_body/stateless_random_uniform/pfor/Shape:output:0asequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack:output:0csequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_1:output:0csequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask
Nsequential/random_contrast/loop_body/stateless_random_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : æ
Isequential/random_contrast/loop_body/stateless_random_uniform/pfor/concatConcatV2Ysequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice:output:0Psequential/random_contrast/loop_body/stateless_random_uniform/pfor/Tile:output:0[sequential/random_contrast/loop_body/stateless_random_uniform/pfor/strided_slice_1:output:0Wsequential/random_contrast/loop_body/stateless_random_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:©
Lsequential/random_contrast/loop_body/stateless_random_uniform/pfor/Reshape_1ReshapeNsequential/random_contrast/loop_body/stateless_random_uniform/mul/pfor/Mul:z:0Rsequential/random_contrast/loop_body/stateless_random_uniform/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
Hsequential/random_contrast/loop_body/stateless_random_uniform/pfor/AddV2AddV2Usequential/random_contrast/loop_body/stateless_random_uniform/pfor/Reshape_1:output:0Jsequential/random_contrast/loop_body/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7sequential/random_contrast/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : z
8sequential/random_contrast/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :å
6sequential/random_contrast/loop_body/SelectV2/pfor/addAddV2@sequential/random_contrast/loop_body/SelectV2/pfor/Rank:output:0Asequential/random_contrast/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: {
9sequential/random_contrast/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :{
9sequential/random_contrast/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : |
:sequential/random_contrast/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ë
8sequential/random_contrast/loop_body/SelectV2/pfor/add_1AddV2Bsequential/random_contrast/loop_body/SelectV2/pfor/Rank_2:output:0Csequential/random_contrast/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: æ
:sequential/random_contrast/loop_body/SelectV2/pfor/MaximumMaximumBsequential/random_contrast/loop_body/SelectV2/pfor/Rank_1:output:0:sequential/random_contrast/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: æ
<sequential/random_contrast/loop_body/SelectV2/pfor/Maximum_1Maximum<sequential/random_contrast/loop_body/SelectV2/pfor/add_1:z:0>sequential/random_contrast/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: 
8sequential/random_contrast/loop_body/SelectV2/pfor/ShapeShape.sequential/random_contrast/pfor/range:output:0*
T0*
_output_shapes
:ä
6sequential/random_contrast/loop_body/SelectV2/pfor/subSub@sequential/random_contrast/loop_body/SelectV2/pfor/Maximum_1:z:0Bsequential/random_contrast/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: 
@sequential/random_contrast/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ñ
:sequential/random_contrast/loop_body/SelectV2/pfor/ReshapeReshape:sequential/random_contrast/loop_body/SelectV2/pfor/sub:z:0Isequential/random_contrast/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
=sequential/random_contrast/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:ï
7sequential/random_contrast/loop_body/SelectV2/pfor/TileTileFsequential/random_contrast/loop_body/SelectV2/pfor/Tile/input:output:0Csequential/random_contrast/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Fsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Hsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Hsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
@sequential/random_contrast/loop_body/SelectV2/pfor/strided_sliceStridedSliceAsequential/random_contrast/loop_body/SelectV2/pfor/Shape:output:0Osequential/random_contrast/loop_body/SelectV2/pfor/strided_slice/stack:output:0Qsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Qsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Hsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Jsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Jsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
Bsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice_1StridedSliceAsequential/random_contrast/loop_body/SelectV2/pfor/Shape:output:0Qsequential/random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Ssequential/random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Ssequential/random_contrast/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask
>sequential/random_contrast/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
9sequential/random_contrast/loop_body/SelectV2/pfor/concatConcatV2Isequential/random_contrast/loop_body/SelectV2/pfor/strided_slice:output:0@sequential/random_contrast/loop_body/SelectV2/pfor/Tile:output:0Ksequential/random_contrast/loop_body/SelectV2/pfor/strided_slice_1:output:0Gsequential/random_contrast/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:é
<sequential/random_contrast/loop_body/SelectV2/pfor/Reshape_1Reshape.sequential/random_contrast/pfor/range:output:0Bsequential/random_contrast/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
;sequential/random_contrast/loop_body/SelectV2/pfor/SelectV2SelectV20sequential/random_contrast/loop_body/Greater:z:0Esequential/random_contrast/loop_body/SelectV2/pfor/Reshape_1:output:08sequential/random_contrast/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential/random_contrast/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
;sequential/random_contrast/loop_body/GatherV2/pfor/GatherV2GatherV2inputsDsequential/random_contrast/loop_body/SelectV2/pfor/SelectV2:output:0Isequential/random_contrast/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Msequential/random_contrast/loop_body/adjust_contrast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Osequential/random_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Osequential/random_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
Gsequential/random_contrast/loop_body/adjust_contrast/pfor/strided_sliceStridedSlice0sequential/random_contrast/pfor/Reshape:output:0Vsequential/random_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack:output:0Xsequential/random_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_1:output:0Xsequential/random_contrast/loop_body/adjust_contrast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask 
Usequential/random_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿà
Gsequential/random_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2TensorListReserve^sequential/random_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2/element_shape:output:0Psequential/random_contrast/loop_body/adjust_contrast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
?sequential/random_contrast/loop_body/adjust_contrast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Rsequential/random_contrast/loop_body/adjust_contrast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Lsequential/random_contrast/loop_body/adjust_contrast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ý
?sequential/random_contrast/loop_body/adjust_contrast/pfor/whileStatelessWhileUsequential/random_contrast/loop_body/adjust_contrast/pfor/while/loop_counter:output:0[sequential/random_contrast/loop_body/adjust_contrast/pfor/while/maximum_iterations:output:0Hsequential/random_contrast/loop_body/adjust_contrast/pfor/Const:output:0Psequential/random_contrast/loop_body/adjust_contrast/pfor/TensorArrayV2:handle:0Psequential/random_contrast/loop_body/adjust_contrast/pfor/strided_slice:output:0Dsequential/random_contrast/loop_body/GatherV2/pfor/GatherV2:output:0Lsequential/random_contrast/loop_body/stateless_random_uniform/pfor/AddV2:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*J
_output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *V
bodyNRL
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_body_24597*V
condNRL
Jsequential_random_contrast_loop_body_adjust_contrast_pfor_while_cond_24596*I
output_shapes8
6: : : : : :ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ
Asequential/random_contrast/loop_body/adjust_contrast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ³
Zsequential/random_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿ           Æ
Lsequential/random_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2TensorListConcatV2Hsequential/random_contrast/loop_body/adjust_contrast/pfor/while:output:3csequential/random_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2/element_shape:output:0Jsequential/random_contrast/loop_body/adjust_contrast/pfor/Const_1:output:0*@
_output_shapes.
,:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0ê
Ksequential/random_contrast/loop_body/adjust_contrast/Identity/pfor/IdentityIdentityUsequential/random_contrast/loop_body/adjust_contrast/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Dsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
Fsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
Esequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Csequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/addAddV2Osequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Rank_1:output:0Nsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/add/y:output:0*
T0*
_output_shapes
: 
Gsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/MaximumMaximumGsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/add:z:0Msequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: É
Esequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/ShapeShapeTsequential/random_contrast/loop_body/adjust_contrast/Identity/pfor/Identity:output:0*
T0*
_output_shapes
:
Csequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/subSubKsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Maximum:z:0Msequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Rank:output:0*
T0*
_output_shapes
: 
Msequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Gsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/ReshapeReshapeGsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/sub:z:0Vsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Jsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/TileTileSsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Tile/input:output:0Psequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Ssequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Usequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Usequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Msequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_sliceStridedSliceNsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Shape:output:0\sequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack:output:0^sequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_1:output:0^sequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Usequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:¡
Wsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
Wsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Osequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1StridedSliceNsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Shape:output:0^sequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack:output:0`sequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_1:output:0`sequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
Ksequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
Fsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/concatConcatV2Vsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice:output:0Msequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Tile:output:0Xsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/strided_slice_1:output:0Tsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:·
Isequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape_1ReshapeTsequential/random_contrast/loop_body/adjust_contrast/Identity/pfor/Identity:output:0Osequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ©
Gsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/MinimumMinimumRsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Reshape_1:output:0Esequential/random_contrast/loop_body/clip_by_value/Minimum/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ~
<sequential/random_contrast/loop_body/clip_by_value/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
>sequential/random_contrast/loop_body/clip_by_value/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
=sequential/random_contrast/loop_body/clip_by_value/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ö
;sequential/random_contrast/loop_body/clip_by_value/pfor/addAddV2Gsequential/random_contrast/loop_body/clip_by_value/pfor/Rank_1:output:0Fsequential/random_contrast/loop_body/clip_by_value/pfor/add/y:output:0*
T0*
_output_shapes
: ó
?sequential/random_contrast/loop_body/clip_by_value/pfor/MaximumMaximum?sequential/random_contrast/loop_body/clip_by_value/pfor/add:z:0Esequential/random_contrast/loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: ¸
=sequential/random_contrast/loop_body/clip_by_value/pfor/ShapeShapeKsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Minimum:z:0*
T0*
_output_shapes
:ï
;sequential/random_contrast/loop_body/clip_by_value/pfor/subSubCsequential/random_contrast/loop_body/clip_by_value/pfor/Maximum:z:0Esequential/random_contrast/loop_body/clip_by_value/pfor/Rank:output:0*
T0*
_output_shapes
: 
Esequential/random_contrast/loop_body/clip_by_value/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?sequential/random_contrast/loop_body/clip_by_value/pfor/ReshapeReshape?sequential/random_contrast/loop_body/clip_by_value/pfor/sub:z:0Nsequential/random_contrast/loop_body/clip_by_value/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Bsequential/random_contrast/loop_body/clip_by_value/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:þ
<sequential/random_contrast/loop_body/clip_by_value/pfor/TileTileKsequential/random_contrast/loop_body/clip_by_value/pfor/Tile/input:output:0Hsequential/random_contrast/loop_body/clip_by_value/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Ksequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Msequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
Esequential/random_contrast/loop_body/clip_by_value/pfor/strided_sliceStridedSliceFsequential/random_contrast/loop_body/clip_by_value/pfor/Shape:output:0Tsequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice/stack:output:0Vsequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_1:output:0Vsequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Msequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Osequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Osequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:í
Gsequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice_1StridedSliceFsequential/random_contrast/loop_body/clip_by_value/pfor/Shape:output:0Vsequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack:output:0Xsequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_1:output:0Xsequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
Csequential/random_contrast/loop_body/clip_by_value/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¯
>sequential/random_contrast/loop_body/clip_by_value/pfor/concatConcatV2Nsequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice:output:0Esequential/random_contrast/loop_body/clip_by_value/pfor/Tile:output:0Psequential/random_contrast/loop_body/clip_by_value/pfor/strided_slice_1:output:0Lsequential/random_contrast/loop_body/clip_by_value/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Asequential/random_contrast/loop_body/clip_by_value/pfor/Reshape_1ReshapeKsequential/random_contrast/loop_body/clip_by_value/Minimum/pfor/Minimum:z:0Gsequential/random_contrast/loop_body/clip_by_value/pfor/concat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
Asequential/random_contrast/loop_body/clip_by_value/pfor/Maximum_1MaximumJsequential/random_contrast/loop_body/clip_by_value/pfor/Reshape_1:output:0=sequential/random_contrast/loop_body/clip_by_value/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
sequential/random_zoom/ShapeShapeEsequential/random_contrast/loop_body/clip_by_value/pfor/Maximum_1:z:0*
T0*
_output_shapes
:t
*sequential/random_zoom/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential/random_zoom/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential/random_zoom/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$sequential/random_zoom/strided_sliceStridedSlice%sequential/random_zoom/Shape:output:03sequential/random_zoom/strided_slice/stack:output:05sequential/random_zoom/strided_slice/stack_1:output:05sequential/random_zoom/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
,sequential/random_zoom/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
.sequential/random_zoom/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿx
.sequential/random_zoom/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&sequential/random_zoom/strided_slice_1StridedSlice%sequential/random_zoom/Shape:output:05sequential/random_zoom/strided_slice_1/stack:output:07sequential/random_zoom/strided_slice_1/stack_1:output:07sequential/random_zoom/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
sequential/random_zoom/CastCast/sequential/random_zoom/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
,sequential/random_zoom/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
.sequential/random_zoom/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.sequential/random_zoom/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
&sequential/random_zoom/strided_slice_2StridedSlice%sequential/random_zoom/Shape:output:05sequential/random_zoom/strided_slice_2/stack:output:07sequential/random_zoom/strided_slice_2/stack_1:output:07sequential/random_zoom/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
sequential/random_zoom/Cast_1Cast/sequential/random_zoom/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: q
/sequential/random_zoom/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ì
-sequential/random_zoom/stateful_uniform/shapePack-sequential/random_zoom/strided_slice:output:08sequential/random_zoom/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:p
+sequential/random_zoom/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?p
+sequential/random_zoom/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?w
-sequential/random_zoom/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
,sequential/random_zoom/stateful_uniform/ProdProd6sequential/random_zoom/stateful_uniform/shape:output:06sequential/random_zoom/stateful_uniform/Const:output:0*
T0*
_output_shapes
: p
.sequential/random_zoom/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
.sequential/random_zoom/stateful_uniform/Cast_1Cast5sequential/random_zoom/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
6sequential/random_zoom/stateful_uniform/RngReadAndSkipRngReadAndSkip?sequential_random_zoom_stateful_uniform_rngreadandskip_resource7sequential/random_zoom/stateful_uniform/Cast/x:output:02sequential/random_zoom/stateful_uniform/Cast_1:y:0*
_output_shapes
:
;sequential/random_zoom/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential/random_zoom/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential/random_zoom/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/random_zoom/stateful_uniform/strided_sliceStridedSlice>sequential/random_zoom/stateful_uniform/RngReadAndSkip:value:0Dsequential/random_zoom/stateful_uniform/strided_slice/stack:output:0Fsequential/random_zoom/stateful_uniform/strided_slice/stack_1:output:0Fsequential/random_zoom/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask«
/sequential/random_zoom/stateful_uniform/BitcastBitcast>sequential/random_zoom/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
=sequential/random_zoom/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?sequential/random_zoom/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?sequential/random_zoom/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7sequential/random_zoom/stateful_uniform/strided_slice_1StridedSlice>sequential/random_zoom/stateful_uniform/RngReadAndSkip:value:0Fsequential/random_zoom/stateful_uniform/strided_slice_1/stack:output:0Hsequential/random_zoom/stateful_uniform/strided_slice_1/stack_1:output:0Hsequential/random_zoom/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:¯
1sequential/random_zoom/stateful_uniform/Bitcast_1Bitcast@sequential/random_zoom/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Dsequential/random_zoom/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
@sequential/random_zoom/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV26sequential/random_zoom/stateful_uniform/shape:output:0:sequential/random_zoom/stateful_uniform/Bitcast_1:output:08sequential/random_zoom/stateful_uniform/Bitcast:output:0Msequential/random_zoom/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
+sequential/random_zoom/stateful_uniform/subSub4sequential/random_zoom/stateful_uniform/max:output:04sequential/random_zoom/stateful_uniform/min:output:0*
T0*
_output_shapes
: à
+sequential/random_zoom/stateful_uniform/mulMulIsequential/random_zoom/stateful_uniform/StatelessRandomUniformV2:output:0/sequential/random_zoom/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
'sequential/random_zoom/stateful_uniformAddV2/sequential/random_zoom/stateful_uniform/mul:z:04sequential/random_zoom/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"sequential/random_zoom/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ë
sequential/random_zoom/concatConcatV2+sequential/random_zoom/stateful_uniform:z:0+sequential/random_zoom/stateful_uniform:z:0+sequential/random_zoom/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
(sequential/random_zoom/zoom_matrix/ShapeShape&sequential/random_zoom/concat:output:0*
T0*
_output_shapes
:
6sequential/random_zoom/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8sequential/random_zoom/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8sequential/random_zoom/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0sequential/random_zoom/zoom_matrix/strided_sliceStridedSlice1sequential/random_zoom/zoom_matrix/Shape:output:0?sequential/random_zoom/zoom_matrix/strided_slice/stack:output:0Asequential/random_zoom/zoom_matrix/strided_slice/stack_1:output:0Asequential/random_zoom/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
(sequential/random_zoom/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
&sequential/random_zoom/zoom_matrix/subSub!sequential/random_zoom/Cast_1:y:01sequential/random_zoom/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: q
,sequential/random_zoom/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¹
*sequential/random_zoom/zoom_matrix/truedivRealDiv*sequential/random_zoom/zoom_matrix/sub:z:05sequential/random_zoom/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 
8sequential/random_zoom/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
:sequential/random_zoom/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Å
2sequential/random_zoom/zoom_matrix/strided_slice_1StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_1/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_1/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masko
*sequential/random_zoom/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ó
(sequential/random_zoom/zoom_matrix/sub_1Sub3sequential/random_zoom/zoom_matrix/sub_1/x:output:0;sequential/random_zoom/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
&sequential/random_zoom/zoom_matrix/mulMul.sequential/random_zoom/zoom_matrix/truediv:z:0,sequential/random_zoom/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*sequential/random_zoom/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¦
(sequential/random_zoom/zoom_matrix/sub_2Subsequential/random_zoom/Cast:y:03sequential/random_zoom/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: s
.sequential/random_zoom/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¿
,sequential/random_zoom/zoom_matrix/truediv_1RealDiv,sequential/random_zoom/zoom_matrix/sub_2:z:07sequential/random_zoom/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 
8sequential/random_zoom/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
:sequential/random_zoom/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Å
2sequential/random_zoom/zoom_matrix/strided_slice_2StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_2/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_2/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masko
*sequential/random_zoom/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ó
(sequential/random_zoom/zoom_matrix/sub_3Sub3sequential/random_zoom/zoom_matrix/sub_3/x:output:0;sequential/random_zoom/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
(sequential/random_zoom/zoom_matrix/mul_1Mul0sequential/random_zoom/zoom_matrix/truediv_1:z:0,sequential/random_zoom/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8sequential/random_zoom/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
:sequential/random_zoom/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Å
2sequential/random_zoom/zoom_matrix/strided_slice_3StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_3/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_3/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masks
1sequential/random_zoom/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ü
/sequential/random_zoom/zoom_matrix/zeros/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0:sequential/random_zoom/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:s
.sequential/random_zoom/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Õ
(sequential/random_zoom/zoom_matrix/zerosFill8sequential/random_zoom/zoom_matrix/zeros/packed:output:07sequential/random_zoom/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
3sequential/random_zoom/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :à
1sequential/random_zoom/zoom_matrix/zeros_1/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0<sequential/random_zoom/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:u
0sequential/random_zoom/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Û
*sequential/random_zoom/zoom_matrix/zeros_1Fill:sequential/random_zoom/zoom_matrix/zeros_1/packed:output:09sequential/random_zoom/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8sequential/random_zoom/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
:sequential/random_zoom/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Å
2sequential/random_zoom/zoom_matrix/strided_slice_4StridedSlice&sequential/random_zoom/concat:output:0Asequential/random_zoom/zoom_matrix/strided_slice_4/stack:output:0Csequential/random_zoom/zoom_matrix/strided_slice_4/stack_1:output:0Csequential/random_zoom/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masku
3sequential/random_zoom/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :à
1sequential/random_zoom/zoom_matrix/zeros_2/packedPack9sequential/random_zoom/zoom_matrix/strided_slice:output:0<sequential/random_zoom/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:u
0sequential/random_zoom/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Û
*sequential/random_zoom/zoom_matrix/zeros_2Fill:sequential/random_zoom/zoom_matrix/zeros_2/packed:output:09sequential/random_zoom/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
.sequential/random_zoom/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
)sequential/random_zoom/zoom_matrix/concatConcatV2;sequential/random_zoom/zoom_matrix/strided_slice_3:output:01sequential/random_zoom/zoom_matrix/zeros:output:0*sequential/random_zoom/zoom_matrix/mul:z:03sequential/random_zoom/zoom_matrix/zeros_1:output:0;sequential/random_zoom/zoom_matrix/strided_slice_4:output:0,sequential/random_zoom/zoom_matrix/mul_1:z:03sequential/random_zoom/zoom_matrix/zeros_2:output:07sequential/random_zoom/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential/random_zoom/transform/ShapeShapeEsequential/random_contrast/loop_body/clip_by_value/pfor/Maximum_1:z:0*
T0*
_output_shapes
:~
4sequential/random_zoom/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
6sequential/random_zoom/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6sequential/random_zoom/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
.sequential/random_zoom/transform/strided_sliceStridedSlice/sequential/random_zoom/transform/Shape:output:0=sequential/random_zoom/transform/strided_slice/stack:output:0?sequential/random_zoom/transform/strided_slice/stack_1:output:0?sequential/random_zoom/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:p
+sequential/random_zoom/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¸
;sequential/random_zoom/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Esequential/random_contrast/loop_body/clip_by_value/pfor/Maximum_1:z:02sequential/random_zoom/zoom_matrix/concat:output:07sequential/random_zoom/transform/strided_slice:output:04sequential/random_zoom/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;W
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
rescaling/mulMulPsequential/random_zoom/transform/ImageProjectiveTransformV3:transformed_images:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0´
conv2d/Conv2DConv2Drescaling/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¨
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ã
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP ¬
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( *
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Å
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@¬
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout/dropout/MulMul max_pooling2d_2/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
dropout/dropout/ShapeShape max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
:¤
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Æ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¿
conv2d_3/Conv2DConv2Ddropout/dropout/Mul_1:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingVALID*
strides
\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_1/dropout/MulMul max_pooling2d_3/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

g
dropout_1/dropout/ShapeShape max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:©
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Í
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 2  
flatten/ReshapeReshapedropout_1/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOpN^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipY^sequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while7^sequential/random_zoom/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2
Msequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkipMsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip2´
Xsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/whileXsequential/random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while2p
6sequential/random_zoom/stateful_uniform/RngReadAndSkip6sequential/random_zoom/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

`
'__inference_dropout_layer_call_fn_25796

inputs
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23581w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Ý

erandom_contrast_loop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_cond_25258Ç
Ârandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counterÍ
Èrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsj
frandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderl
hrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1l
hrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Ç
Ârandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_sliceÞ
Ùrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_cond_25258___redundant_placeholder0	g
crandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity
µ
_random_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/LessLessfrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderÂrandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_less_random_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice*
T0*
_output_shapes
: õ
crandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentitycrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Less:z:0*
T0
*
_output_shapes
: "Ó
crandom_contrast_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identitylrandom_contrast/loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
ä

*__inference_sequential_layer_call_fn_24884

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23249y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ  : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ñ>
÷
G__inference_sequential_1_layer_call_and_return_conditional_losses_23698

inputs
sequential_23654:	
sequential_23656:	&
conv2d_23660:
conv2d_23662:(
conv2d_1_23666: 
conv2d_1_23668: (
conv2d_2_23672: @
conv2d_2_23674:@)
conv2d_3_23679:@
conv2d_3_23681:	
dense_23687:
d
dense_23689:	 
dense_1_23692:	
dense_1_23694:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢"sequential/StatefulPartitionedCallû
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_23654sequential_23656*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23249è
rescaling/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_23345
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_23660conv2d_23662*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23358ê
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23290
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_23666conv2d_1_23668*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23376ð
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23302
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_23672conv2d_2_23674*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_23394ð
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_23314ï
dropout/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23581
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_3_23679conv2d_3_23681*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_23419ñ
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23326
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_23548Ú
flatten/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23439ü
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_23687dense_23689*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23452
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_23692dense_1_23694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_23468w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿý
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
­
C
'__inference_flatten_layer_call_fn_25875

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23439a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
ô
U
*__inference_sequential_layer_call_fn_22407
random_contrast_input
identityÉ
PartitionedCallPartitionedCallrandom_contrast_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_22404j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :h d
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
/
_user_specified_namerandom_contrast_input

ô
9loop_body_stateful_uniform_full_int_pfor_while_cond_22800n
jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_countert
ploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations>
:loop_body_stateful_uniform_full_int_pfor_while_placeholder@
<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1n
jloop_body_stateful_uniform_full_int_pfor_while_less_loop_body_stateful_uniform_full_int_pfor_strided_slice
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_22800___redundant_placeholder0
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_22800___redundant_placeholder1
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_22800___redundant_placeholder2
loop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_cond_22800___redundant_placeholder3;
7loop_body_stateful_uniform_full_int_pfor_while_identity

3loop_body/stateful_uniform_full_int/pfor/while/LessLess:loop_body_stateful_uniform_full_int_pfor_while_placeholderjloop_body_stateful_uniform_full_int_pfor_while_less_loop_body_stateful_uniform_full_int_pfor_strided_slice*
T0*
_output_shapes
: 
7loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentity7loop_body/stateful_uniform_full_int/pfor/while/Less:z:0*
T0
*
_output_shapes
: "{
7loop_body_stateful_uniform_full_int_pfor_while_identity@loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
ÍB
¸	
G__inference_sequential_1_layer_call_and_return_conditional_losses_24008

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@7
(conv2d_3_biasadd_readvariableop_resource:	8
$dense_matmul_readvariableop_resource:
d4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	5
'dense_1_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOpU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;W
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    s
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0´
conv2d/Conv2DConv2Drescaling/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  h
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  ¨
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ã
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP *
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿPP ¬
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(( *
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Å
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ((@¬
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
x
dropout/IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0¿
conv2d_3/Conv2DConv2Ddropout/Identity:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingVALID*
strides
{
dropout_1/IdentityIdentity max_pooling2d_3/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 2  
flatten/ReshapeReshapedropout_1/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ÃJ

\sequential_random_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_24155µ
°sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter»
¶sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsa
]sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderc
_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1²
­sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0Å
Àsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0	^
Zsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity`
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1`
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2`
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3°
«sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_sliceÃ
¾sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice	
Wsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :À
Usequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/addAddV2]sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder`sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: ©
gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ì
esequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stackPack]sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderpsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:«
isequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ì
gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1PackYsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add:z:0rsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¸
gsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      å
_sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_sliceStridedSliceÀsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0nsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack:output:0psequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1:output:0psequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÿ
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/BitcastBitcasthsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0¢
`sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : â
\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims
ExpandDimsbsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Bitcast:output:0isequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:ò
vsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1]sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderesequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ä
Wsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1AddV2]sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderbsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Ysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
Wsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2AddV2°sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counterbsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ä
Zsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentity[sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: Â
\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1Identity¶sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: æ
\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2Identity[sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: 
\sequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3Identitysequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "Á
Zsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identitycsequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0"Å
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1esequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1:output:0"Å
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2esequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2:output:0"Å
\sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3esequential/random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3:output:0"Þ
«sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice­sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0"
¾sequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedsliceÀsequential_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_sequential_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_23314

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼C
Ç
Qrandom_contrast_loop_body_stateful_uniform_full_int_Bitcast_pfor_while_body_25035
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counter¥
 random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterationsV
Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderX
Trandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0¯
ªrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0	S
Orandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityU
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1U
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2U
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice­
¨random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice	
Lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Jrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/addAddV2Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderUrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
Zrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stackPackRrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholdererandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
: 
^random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ë
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1PackNrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add:z:0grandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:­
\random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      £
Trandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_sliceStridedSliceªrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0crandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack:output:0erandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_1:output:0erandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maské
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/BitcastBitcast]random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Urandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Á
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims
ExpandDimsWrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Bitcast:output:0^random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Æ
krandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemTrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholder_1Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderZrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
Lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1AddV2Rrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_placeholderWrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Nrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ì
Lrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2AddV2random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_loop_counterWrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: Î
Orandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/IdentityIdentityPrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: ¡
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1Identity random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ð
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2IdentityPrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: û
Qrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3Identity{random_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "«
Orandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identityXrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity:output:0"¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_1Zrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_1:output:0"¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_2Zrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_2:output:0"¯
Qrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_identity_3Zrandom_contrast/loop_body/stateful_uniform_full_int/Bitcast/pfor/while/Identity_3:output:0"²
random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slicerandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_strided_slice_0"Ø
¨random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedsliceªrandom_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23290

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23326

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
_
Û
__inference__traced_save_26925
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop)
%savev2_statevar_1_read_readvariableop	'
#savev2_statevar_read_readvariableop	3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ë
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*ô
valueêBç0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÍ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop%savev2_statevar_1_read_readvariableop#savev2_statevar_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220			
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*É
_input_shapes·
´: ::: : : @:@:@::
d::	:: : : : : : : : : ::::: : : @:@:@::
d::	:::: : : @:@:@::
d::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::&	"
 
_output_shapes
:
d:!


_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::
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
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::& "
 
_output_shapes
:
d:!!

_output_shapes	
::%"!

_output_shapes
:	: #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
: @: )

_output_shapes
:@:-*)
'
_output_shapes
:@:!+

_output_shapes	
::&,"
 
_output_shapes
:
d:!-

_output_shapes	
::%.!

_output_shapes
:	: /

_output_shapes
::0

_output_shapes
: 
ØP

9loop_body_stateful_uniform_full_int_pfor_while_body_22801n
jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_countert
ploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations>
:loop_body_stateful_uniform_full_int_pfor_while_placeholder@
<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1k
gloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0^
Zloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0\
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0;
7loop_body_stateful_uniform_full_int_pfor_while_identity=
9loop_body_stateful_uniform_full_int_pfor_while_identity_1=
9loop_body_stateful_uniform_full_int_pfor_while_identity_2=
9loop_body_stateful_uniform_full_int_pfor_while_identity_3i
eloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2\
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shapeZ
Vloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_algv
4loop_body/stateful_uniform_full_int/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :×
2loop_body/stateful_uniform_full_int/pfor/while/addAddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder=loop_body/stateful_uniform_full_int/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Bloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stackPack:loop_body_stateful_uniform_full_int_pfor_while_placeholderMloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1Pack6loop_body/stateful_uniform_full_int/pfor/while/add:z:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
<loop_body/stateful_uniform_full_int/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0Kloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack:output:0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1:output:0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskx
6loop_body/stateful_uniform_full_int/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Û
4loop_body/stateful_uniform_full_int/pfor/while/add_1AddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder?loop_body/stateful_uniform_full_int/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stackPack:loop_body_stateful_uniform_full_int_pfor_while_placeholderOloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1Pack8loop_body/stateful_uniform_full_int/pfor/while/add_1:z:0Qloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¥
>loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1StridedSliceloop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack:output:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1:output:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maské
Nloop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2StatelessRandomUniformFullIntV2Zloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0Eloop_body/stateful_uniform_full_int/pfor/while/strided_slice:output:0Gloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1:output:0Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0	
=loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
9loop_body/stateful_uniform_full_int/pfor/while/ExpandDims
ExpandDimsWloop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2:output:0Floop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
Sloop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1:loop_body_stateful_uniform_full_int_pfor_while_placeholderBloop_body/stateful_uniform_full_int/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐx
6loop_body/stateful_uniform_full_int/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :Û
4loop_body/stateful_uniform_full_int/pfor/while/add_2AddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder?loop_body/stateful_uniform_full_int/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: x
6loop_body/stateful_uniform_full_int/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
4loop_body/stateful_uniform_full_int/pfor/while/add_3AddV2jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_counter?loop_body/stateful_uniform_full_int/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
7loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentity8loop_body/stateful_uniform_full_int/pfor/while/add_3:z:0*
T0*
_output_shapes
: Ø
9loop_body/stateful_uniform_full_int/pfor/while/Identity_1Identityploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations*
T0*
_output_shapes
:  
9loop_body/stateful_uniform_full_int/pfor/while/Identity_2Identity8loop_body/stateful_uniform_full_int/pfor/while/add_2:z:0*
T0*
_output_shapes
: Ë
9loop_body/stateful_uniform_full_int/pfor/while/Identity_3Identitycloop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "{
7loop_body_stateful_uniform_full_int_pfor_while_identity@loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_1Bloop_body/stateful_uniform_full_int/pfor/while/Identity_1:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_2Bloop_body/stateful_uniform_full_int/pfor/while/Identity_2:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_3Bloop_body/stateful_uniform_full_int/pfor/while/Identity_3:output:0"²
Vloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_algXloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0"Ð
eloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slicegloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice_0"¶
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shapeZloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0"
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0"
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
Æ

/__inference_random_contrast_layer_call_fn_25932

inputs
unknown:	
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_23227y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ö
`
D__inference_rescaling_layer_call_and_return_conditional_losses_25696

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

b
)__inference_dropout_1_layer_call_fn_25853

inputs
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_23548x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
Å
E
)__inference_rescaling_layer_call_fn_25688

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_23345j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ØQ
ô
Uloop_body_stateless_random_uniform_StatelessRandomGetKeyCounter_pfor_while_body_22901§
¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter­
¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterationsZ
Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1\
Xloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2¤
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0	W
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identityY
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3Y
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4¢
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice	
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :«
Nloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/addAddV2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderYloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¢
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ×
^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stackPackVloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholderiloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:¤
bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ×
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1PackRloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add:z:0kloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:±
`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Xloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_sliceStridedSliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack:output:0iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_1:output:0iloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
gloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounteraloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/strided_slice:output:0* 
_output_shapes
::
Yloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ß
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims
ExpandDimsmloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:key:0bloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:Ö
oloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemXloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_1Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ç
Wloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1
ExpandDimsqloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/StatelessRandomGetKeyCounter:counter:0dloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1/dim:output:0*
T0*
_output_shapes

:Ú
qloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItemTensorListSetItemXloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder_2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder`loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/ExpandDims_1:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¯
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1AddV2Vloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_placeholder[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Rloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ü
Ploop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2AddV2¢loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_counter[loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: Ö
Sloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/IdentityIdentityTloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_2:z:0*
T0*
_output_shapes
: ­
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1Identity¨loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ø
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2IdentityTloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/add_1:z:0*
T0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3Identityloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 
Uloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4Identityloop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/TensorArrayV2Write_1/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "³
Sloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity\loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_1^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_1:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_2^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_2:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_3^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_3:output:0"·
Uloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_identity_4^loop_body/stateless_random_uniform/StatelessRandomGetKeyCounter/pfor/while/Identity_4:output:0"Â
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_sliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_strided_slice_0"
loop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedsliceloop_body_stateless_random_uniform_statelessrandomgetkeycounter_pfor_while_strided_slice_loop_body_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*2
_input_shapes!
: : : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Ï
#__inference_signature_wrapper_23889
sequential_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@
	unknown_6:	
	unknown_7:
d
	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_22384o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ  : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
*
_user_specified_namesequential_input
n
¿
F__inference_random_zoom_layer_call_and_return_conditional_losses_26751

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ  : 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¸:
¸

Cloop_body_stateful_uniform_full_int_Bitcast_1_pfor_while_body_26150
~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counter
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterationsH
Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderJ
Floop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1
{loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0	E
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityG
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1G
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2G
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3}
yloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice	
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :õ
<loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/addAddV2Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderGloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stackPackDloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderWloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Ploop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¡
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1Pack@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add:z:0Yloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Nloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
Floop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0Uloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack:output:0Wloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_1:output:0Wloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÍ
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/BitcastBitcastOloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Gloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims
ExpandDimsIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Bitcast:output:0Ploop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:
]loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemFloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholder_1Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderLloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ù
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1AddV2Dloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_placeholderIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
@loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :³
>loop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2AddV2~loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_counterIloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ²
Aloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/IdentityIdentityBloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: ÷
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1Identityloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: ´
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2IdentityBloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: ß
Cloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3Identitymloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Aloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identityJloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_1Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_1:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_2Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_2:output:0"
Cloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_identity_3Lloop_body/stateful_uniform_full_int/Bitcast_1/pfor/while/Identity_3:output:0"ø
yloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice{loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_loop_body_stateful_uniform_full_int_bitcast_1_pfor_strided_slice_0" 
loop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedsliceloop_body_stateful_uniform_full_int_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ØP

9loop_body_stateful_uniform_full_int_pfor_while_body_26207n
jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_countert
ploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations>
:loop_body_stateful_uniform_full_int_pfor_while_placeholder@
<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1k
gloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice_0
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0^
Zloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0\
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0;
7loop_body_stateful_uniform_full_int_pfor_while_identity=
9loop_body_stateful_uniform_full_int_pfor_while_identity_1=
9loop_body_stateful_uniform_full_int_pfor_while_identity_2=
9loop_body_stateful_uniform_full_int_pfor_while_identity_3i
eloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2\
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shapeZ
Vloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_algv
4loop_body/stateful_uniform_full_int/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :×
2loop_body/stateful_uniform_full_int/pfor/while/addAddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder=loop_body/stateful_uniform_full_int/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Bloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stackPack:loop_body_stateful_uniform_full_int_pfor_while_placeholderMloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1Pack6loop_body/stateful_uniform_full_int/pfor/while/add:z:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
<loop_body/stateful_uniform_full_int/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0Kloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack:output:0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1:output:0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskx
6loop_body/stateful_uniform_full_int/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Û
4loop_body/stateful_uniform_full_int/pfor/while/add_1AddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder?loop_body/stateful_uniform_full_int/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Dloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stackPack:loop_body_stateful_uniform_full_int_pfor_while_placeholderOloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Hloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1Pack8loop_body/stateful_uniform_full_int/pfor/while/add_1:z:0Qloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Floop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¥
>loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1StridedSliceloop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0Mloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack:output:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1:output:0Oloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maské
Nloop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2StatelessRandomUniformFullIntV2Zloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0Eloop_body/stateful_uniform_full_int/pfor/while/strided_slice:output:0Gloop_body/stateful_uniform_full_int/pfor/while/strided_slice_1:output:0Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0	
=loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
9loop_body/stateful_uniform_full_int/pfor/while/ExpandDims
ExpandDimsWloop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2:output:0Floop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
Sloop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem<loop_body_stateful_uniform_full_int_pfor_while_placeholder_1:loop_body_stateful_uniform_full_int_pfor_while_placeholderBloop_body/stateful_uniform_full_int/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐx
6loop_body/stateful_uniform_full_int/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :Û
4loop_body/stateful_uniform_full_int/pfor/while/add_2AddV2:loop_body_stateful_uniform_full_int_pfor_while_placeholder?loop_body/stateful_uniform_full_int/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: x
6loop_body/stateful_uniform_full_int/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
4loop_body/stateful_uniform_full_int/pfor/while/add_3AddV2jloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_loop_counter?loop_body/stateful_uniform_full_int/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: 
7loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentity8loop_body/stateful_uniform_full_int/pfor/while/add_3:z:0*
T0*
_output_shapes
: Ø
9loop_body/stateful_uniform_full_int/pfor/while/Identity_1Identityploop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations*
T0*
_output_shapes
:  
9loop_body/stateful_uniform_full_int/pfor/while/Identity_2Identity8loop_body/stateful_uniform_full_int/pfor/while/add_2:z:0*
T0*
_output_shapes
: Ë
9loop_body/stateful_uniform_full_int/pfor/while/Identity_3Identitycloop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "{
7loop_body_stateful_uniform_full_int_pfor_while_identity@loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_1Bloop_body/stateful_uniform_full_int/pfor/while/Identity_1:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_2Bloop_body/stateful_uniform_full_int/pfor/while/Identity_2:output:0"
9loop_body_stateful_uniform_full_int_pfor_while_identity_3Bloop_body/stateful_uniform_full_int/pfor/while/Identity_3:output:0"²
Vloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_algXloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_alg_0"Ð
eloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slicegloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_pfor_strided_slice_0"¶
Xloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shapeZloop_body_stateful_uniform_full_int_pfor_while_loop_body_stateful_uniform_full_int_shape_0"
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0"
loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2loop_body_stateful_uniform_full_int_pfor_while_strided_slice_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
a

Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_body_25159
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counter
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterationsN
Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderP
Lrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice_0©
¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0©
¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0~
zrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shape_0|
xrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_alg_0K
Grandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identityM
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_1M
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_2M
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_3
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice§
¢random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2§
¢random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2|
xrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shapez
vrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_alg
Drandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Brandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/addAddV2Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderMrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ³
Rrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stackPackJrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder]random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ³
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1PackFrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add:z:0_random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¥
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
Lrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_sliceStridedSlice¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0[random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack:output:0]random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_1:output:0]random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Drandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1AddV2Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderOrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ·
Trandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stackPackJrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Xrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¹
Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1PackHrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_1:z:0arandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:§
Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
Nrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1StridedSlice¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0]random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack:output:0_random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_1:output:0_random_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÙ
^random_contrast/loop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2StatelessRandomUniformFullIntV2zrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shape_0Urandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice:output:0Wrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/strided_slice_1:output:0xrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0	
Mrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ê
Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims
ExpandDimsgrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/StatelessRandomUniformFullIntV2:output:0Vrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
crandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemLrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholder_1Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderRrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
Drandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2AddV2Jrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_placeholderOrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Frandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
Drandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3AddV2random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_loop_counterOrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: ¾
Grandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/IdentityIdentityHrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_3:z:0*
T0*
_output_shapes
: 
Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_1Identityrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_while_maximum_iterations*
T0*
_output_shapes
: À
Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_2IdentityHrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/add_2:z:0*
T0*
_output_shapes
: ë
Irandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_3Identitysrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Grandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identityPrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity:output:0"
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_1Rrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_1:output:0"
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_2Rrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_2:output:0"
Irandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_identity_3Rrandom_contrast/loop_body/stateful_uniform_full_int/pfor/while/Identity_3:output:0"ò
vrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_algxrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_alg_0"
random_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slicerandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_pfor_strided_slice_0"ö
xrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shapezrandom_contrast_loop_body_stateful_uniform_full_int_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_shape_0"Ì
¢random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_1_random_contrast_loop_body_stateful_uniform_full_int_bitcast_pfor_tensorlistconcatv2_0"Ì
¢random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2¤random_contrast_loop_body_stateful_uniform_full_int_pfor_while_strided_slice_random_contrast_loop_body_stateful_uniform_full_int_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
²
I
-__inference_max_pooling2d_layer_call_fn_25721

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23290
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ª
Qloop_body_stateless_random_uniform_StatelessRandomUniformV2_pfor_while_cond_26375
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_counter¥
 loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsV
Rloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderX
Tloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_26375___redundant_placeholder0¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_26375___redundant_placeholder1¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_26375___redundant_placeholder2¶
±loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_cond_26375___redundant_placeholder3S
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identity
å
Kloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/LessLessRloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_placeholderloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: Í
Oloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityOloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "«
Oloop_body_stateless_random_uniform_statelessrandomuniformv2_pfor_while_identityXloop_body/stateless_random_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
»
Í
!__inference__traced_restore_27076
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel: .
 assignvariableop_3_conv2d_1_bias: <
"assignvariableop_4_conv2d_2_kernel: @.
 assignvariableop_5_conv2d_2_bias:@=
"assignvariableop_6_conv2d_3_kernel:@/
 assignvariableop_7_conv2d_3_bias:	3
assignvariableop_8_dense_kernel:
d,
assignvariableop_9_dense_bias:	5
"assignvariableop_10_dense_1_kernel:	.
 assignvariableop_11_dense_1_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: ,
assignvariableop_21_statevar_1:	*
assignvariableop_22_statevar:	B
(assignvariableop_23_adam_conv2d_kernel_m:4
&assignvariableop_24_adam_conv2d_bias_m:D
*assignvariableop_25_adam_conv2d_1_kernel_m: 6
(assignvariableop_26_adam_conv2d_1_bias_m: D
*assignvariableop_27_adam_conv2d_2_kernel_m: @6
(assignvariableop_28_adam_conv2d_2_bias_m:@E
*assignvariableop_29_adam_conv2d_3_kernel_m:@7
(assignvariableop_30_adam_conv2d_3_bias_m:	;
'assignvariableop_31_adam_dense_kernel_m:
d4
%assignvariableop_32_adam_dense_bias_m:	<
)assignvariableop_33_adam_dense_1_kernel_m:	5
'assignvariableop_34_adam_dense_1_bias_m:B
(assignvariableop_35_adam_conv2d_kernel_v:4
&assignvariableop_36_adam_conv2d_bias_v:D
*assignvariableop_37_adam_conv2d_1_kernel_v: 6
(assignvariableop_38_adam_conv2d_1_bias_v: D
*assignvariableop_39_adam_conv2d_2_kernel_v: @6
(assignvariableop_40_adam_conv2d_2_bias_v:@E
*assignvariableop_41_adam_conv2d_3_kernel_v:@7
(assignvariableop_42_adam_conv2d_3_bias_v:	;
'assignvariableop_43_adam_dense_kernel_v:
d4
%assignvariableop_44_adam_dense_bias_v:	<
)assignvariableop_45_adam_dense_1_kernel_v:	5
'assignvariableop_46_adam_dense_1_bias_v:
identity_48¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Î
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*ô
valueêBç0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÐ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ö
_output_shapesÃ
À::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220			[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_statevar_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_statevarIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv2d_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_3_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_3_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv2d_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv2d_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_dense_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ù
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_48IdentityIdentity_47:output:0^NoOp_1*
T0*
_output_shapes
: Æ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_48Identity_48:output:0*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
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
­
á
Xrandom_contrast_loop_body_stateful_uniform_full_int_RngReadAndSkip_pfor_while_cond_24969­
¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_loop_counter³
®random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_maximum_iterations]
Yrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_
[random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder_1­
¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_sliceÄ
¿random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_24969___redundant_placeholder0Ä
¿random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_24969___redundant_placeholder1Ä
¿random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_cond_24969___redundant_placeholder2Z
Vrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity

Rrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/LessLessYrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_placeholder¨random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_less_random_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: Û
Vrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/IdentityIdentityVrandom_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "¹
Vrandom_contrast_loop_body_stateful_uniform_full_int_rngreadandskip_pfor_while_identity_random_contrast/loop_body/stateful_uniform_full_int/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Æ
serving_default²
W
sequential_inputC
"serving_default_sequential_input:0ÿÿÿÿÿÿÿÿÿ  ;
dense_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:©
Ê
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ä
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_sequential
¥
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias
 /_jit_compiled_convolution_op"
_tf_keras_layer
¥
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

<kernel
=bias
 >_jit_compiled_convolution_op"
_tf_keras_layer
¥
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias
 M_jit_compiled_convolution_op"
_tf_keras_layer
¥
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator"
_tf_keras_layer
Ý
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias
 c_jit_compiled_convolution_op"
_tf_keras_layer
¥
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
p_random_generator"
_tf_keras_layer
¥
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
»
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

}kernel
~bias"
_tf_keras_layer
Â
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
x
-0
.1
<2
=3
K4
L5
a6
b7
}8
~9
10
11"
trackable_list_wrapper
x
-0
.1
<2
=3
K4
L5
a6
b7
}8
~9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
trace_0
trace_1
trace_2
trace_32û
,__inference_sequential_1_layer_call_fn_23502
,__inference_sequential_1_layer_call_fn_23918
,__inference_sequential_1_layer_call_fn_23951
,__inference_sequential_1_layer_call_fn_23762À
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
 ztrace_0ztrace_1ztrace_2ztrace_3
Ú
trace_0
trace_1
trace_2
trace_32ç
G__inference_sequential_1_layer_call_and_return_conditional_losses_24008
G__inference_sequential_1_layer_call_and_return_conditional_losses_24870
G__inference_sequential_1_layer_call_and_return_conditional_losses_23805
G__inference_sequential_1_layer_call_and_return_conditional_losses_23852À
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
 ztrace_0ztrace_1ztrace_2ztrace_3
ÔBÑ
 __inference__wrapped_model_22384sequential_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ì
	iter
beta_1
beta_2

decay
learning_rate-m¼.m½<m¾=m¿KmÀLmÁamÂbmÃ}mÄ~mÅ	mÆ	mÇ-vÈ.vÉ<vÊ=vËKvÌLvÍavÎbvÏ}vÐ~vÑ	vÒ	vÓ"
	optimizer
-
serving_default"
signature_map
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
 _random_generator"
_tf_keras_layer
Ã
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses
§_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
æ
­trace_0
®trace_1
¯trace_2
°trace_32ó
*__inference_sequential_layer_call_fn_22407
*__inference_sequential_layer_call_fn_24875
*__inference_sequential_layer_call_fn_24884
*__inference_sequential_layer_call_fn_23265À
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
 z­trace_0z®trace_1z¯trace_2z°trace_3
Ò
±trace_0
²trace_1
³trace_2
´trace_32ß
E__inference_sequential_layer_call_and_return_conditional_losses_24888
E__inference_sequential_layer_call_and_return_conditional_losses_25683
E__inference_sequential_layer_call_and_return_conditional_losses_23271
E__inference_sequential_layer_call_and_return_conditional_losses_23281À
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
 z±trace_0z²trace_1z³trace_2z´trace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
µnon_trainable_variables
¶layers
·metrics
 ¸layer_regularization_losses
¹layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
ï
ºtrace_02Ð
)__inference_rescaling_layer_call_fn_25688¢
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
 zºtrace_0

»trace_02ë
D__inference_rescaling_layer_call_and_return_conditional_losses_25696¢
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
 z»trace_0
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
ì
Átrace_02Í
&__inference_conv2d_layer_call_fn_25705¢
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
 zÁtrace_0

Âtrace_02è
A__inference_conv2d_layer_call_and_return_conditional_losses_25716¢
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
 zÂtrace_0
':%2conv2d/kernel
:2conv2d/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ó
Ètrace_02Ô
-__inference_max_pooling2d_layer_call_fn_25721¢
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
 zÈtrace_0

Étrace_02ï
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25726¢
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
 zÉtrace_0
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
î
Ïtrace_02Ï
(__inference_conv2d_1_layer_call_fn_25735¢
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
 zÏtrace_0

Ðtrace_02ê
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25746¢
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
 zÐtrace_0
):' 2conv2d_1/kernel
: 2conv2d_1/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
õ
Ötrace_02Ö
/__inference_max_pooling2d_1_layer_call_fn_25751¢
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
 zÖtrace_0

×trace_02ñ
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25756¢
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
 z×trace_0
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
î
Ýtrace_02Ï
(__inference_conv2d_2_layer_call_fn_25765¢
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
 zÝtrace_0

Þtrace_02ê
C__inference_conv2d_2_layer_call_and_return_conditional_losses_25776¢
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
 zÞtrace_0
):' @2conv2d_2/kernel
:@2conv2d_2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
õ
ätrace_02Ö
/__inference_max_pooling2d_2_layer_call_fn_25781¢
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
 zätrace_0

åtrace_02ñ
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_25786¢
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
 zåtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ænon_trainable_variables
çlayers
èmetrics
 élayer_regularization_losses
êlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
Ä
ëtrace_0
ìtrace_12
'__inference_dropout_layer_call_fn_25791
'__inference_dropout_layer_call_fn_25796´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zëtrace_0zìtrace_1
ú
ítrace_0
îtrace_12¿
B__inference_dropout_layer_call_and_return_conditional_losses_25801
B__inference_dropout_layer_call_and_return_conditional_losses_25813´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zítrace_0zîtrace_1
"
_generic_user_object
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
î
ôtrace_02Ï
(__inference_conv2d_3_layer_call_fn_25822¢
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
 zôtrace_0

õtrace_02ê
C__inference_conv2d_3_layer_call_and_return_conditional_losses_25833¢
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
 zõtrace_0
*:(@2conv2d_3/kernel
:2conv2d_3/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
õ
ûtrace_02Ö
/__inference_max_pooling2d_3_layer_call_fn_25838¢
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
 zûtrace_0

ütrace_02ñ
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_25843¢
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
 zütrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
È
trace_0
trace_12
)__inference_dropout_1_layer_call_fn_25848
)__inference_dropout_1_layer_call_fn_25853´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
þ
trace_0
trace_12Ã
D__inference_dropout_1_layer_call_and_return_conditional_losses_25858
D__inference_dropout_1_layer_call_and_return_conditional_losses_25870´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
í
trace_02Î
'__inference_flatten_layer_call_fn_25875¢
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
 ztrace_0

trace_02é
B__inference_flatten_layer_call_and_return_conditional_losses_25881¢
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
 ztrace_0
.
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
ë
trace_02Ì
%__inference_dense_layer_call_fn_25890¢
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
 ztrace_0

trace_02ç
@__inference_dense_layer_call_and_return_conditional_losses_25901¢
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
 ztrace_0
 :
d2dense/kernel
:2
dense/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
·
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
trace_02Î
'__inference_dense_1_layer_call_fn_25910¢
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
 ztrace_0

trace_02é
B__inference_dense_1_layer_call_and_return_conditional_losses_25920¢
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
 ztrace_0
!:	2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_1_layer_call_fn_23502sequential_input"À
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
þBû
,__inference_sequential_1_layer_call_fn_23918inputs"À
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
þBû
,__inference_sequential_1_layer_call_fn_23951inputs"À
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
B
,__inference_sequential_1_layer_call_fn_23762sequential_input"À
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
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_24008inputs"À
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
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_24870inputs"À
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
£B 
G__inference_sequential_1_layer_call_and_return_conditional_losses_23805sequential_input"À
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
£B 
G__inference_sequential_1_layer_call_and_return_conditional_losses_23852sequential_input"À
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÓBÐ
#__inference_signature_wrapper_23889sequential_input"
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
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ô
¢trace_0
£trace_12
/__inference_random_contrast_layer_call_fn_25925
/__inference_random_contrast_layer_call_fn_25932´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¢trace_0z£trace_1

¤trace_0
¥trace_12Ï
J__inference_random_contrast_layer_call_and_return_conditional_losses_25936
J__inference_random_contrast_layer_call_and_return_conditional_losses_26633´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¤trace_0z¥trace_1
/
¦
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
Ì
¬trace_0
­trace_12
+__inference_random_zoom_layer_call_fn_26638
+__inference_random_zoom_layer_call_fn_26645´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¬trace_0z­trace_1

®trace_0
¯trace_12Ç
F__inference_random_zoom_layer_call_and_return_conditional_losses_26649
F__inference_random_zoom_layer_call_and_return_conditional_losses_26751´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z®trace_0z¯trace_1
/
°
_generator"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
*__inference_sequential_layer_call_fn_22407random_contrast_input"À
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
üBù
*__inference_sequential_layer_call_fn_24875inputs"À
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
üBù
*__inference_sequential_layer_call_fn_24884inputs"À
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
B
*__inference_sequential_layer_call_fn_23265random_contrast_input"À
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
B
E__inference_sequential_layer_call_and_return_conditional_losses_24888inputs"À
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
B
E__inference_sequential_layer_call_and_return_conditional_losses_25683inputs"À
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
¦B£
E__inference_sequential_layer_call_and_return_conditional_losses_23271random_contrast_input"À
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
¦B£
E__inference_sequential_layer_call_and_return_conditional_losses_23281random_contrast_input"À
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
ÝBÚ
)__inference_rescaling_layer_call_fn_25688inputs"¢
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
øBõ
D__inference_rescaling_layer_call_and_return_conditional_losses_25696inputs"¢
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
ÚB×
&__inference_conv2d_layer_call_fn_25705inputs"¢
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
õBò
A__inference_conv2d_layer_call_and_return_conditional_losses_25716inputs"¢
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
áBÞ
-__inference_max_pooling2d_layer_call_fn_25721inputs"¢
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
üBù
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25726inputs"¢
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
ÜBÙ
(__inference_conv2d_1_layer_call_fn_25735inputs"¢
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
÷Bô
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25746inputs"¢
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
ãBà
/__inference_max_pooling2d_1_layer_call_fn_25751inputs"¢
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
þBû
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25756inputs"¢
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
ÜBÙ
(__inference_conv2d_2_layer_call_fn_25765inputs"¢
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
÷Bô
C__inference_conv2d_2_layer_call_and_return_conditional_losses_25776inputs"¢
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
ãBà
/__inference_max_pooling2d_2_layer_call_fn_25781inputs"¢
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
þBû
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_25786inputs"¢
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
íBê
'__inference_dropout_layer_call_fn_25791inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
íBê
'__inference_dropout_layer_call_fn_25796inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_25801inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_25813inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÜBÙ
(__inference_conv2d_3_layer_call_fn_25822inputs"¢
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
÷Bô
C__inference_conv2d_3_layer_call_and_return_conditional_losses_25833inputs"¢
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
ãBà
/__inference_max_pooling2d_3_layer_call_fn_25838inputs"¢
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
þBû
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_25843inputs"¢
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
ïBì
)__inference_dropout_1_layer_call_fn_25848inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ïBì
)__inference_dropout_1_layer_call_fn_25853inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_25858inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_25870inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÛBØ
'__inference_flatten_layer_call_fn_25875inputs"¢
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
öBó
B__inference_flatten_layer_call_and_return_conditional_losses_25881inputs"¢
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
ÙBÖ
%__inference_dense_layer_call_fn_25890inputs"¢
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
ôBñ
@__inference_dense_layer_call_and_return_conditional_losses_25901inputs"¢
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
ÛBØ
'__inference_dense_1_layer_call_fn_25910inputs"¢
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
öBó
B__inference_dense_1_layer_call_and_return_conditional_losses_25920inputs"¢
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
R
±	variables
²	keras_api

³total

´count"
_tf_keras_metric
c
µ	variables
¶	keras_api

·total

¸count
¹
_fn_kwargs"
_tf_keras_metric
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
õBò
/__inference_random_contrast_layer_call_fn_25925inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
õBò
/__inference_random_contrast_layer_call_fn_25932inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
J__inference_random_contrast_layer_call_and_return_conditional_losses_25936inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
J__inference_random_contrast_layer_call_and_return_conditional_losses_26633inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
/
º
_state_var"
_generic_user_object
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
ñBî
+__inference_random_zoom_layer_call_fn_26638inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ñBî
+__inference_random_zoom_layer_call_fn_26645inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_random_zoom_layer_call_and_return_conditional_losses_26649inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_random_zoom_layer_call_and_return_conditional_losses_26751inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
/
»
_state_var"
_generic_user_object
0
³0
´1"
trackable_list_wrapper
.
±	variables"
_generic_user_object
:  (2total
:  (2count
0
·0
¸1"
trackable_list_wrapper
.
µ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
:	2StateVar
:	2StateVar
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:, 2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, @2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
/:-@2Adam/conv2d_3/kernel/m
!:2Adam/conv2d_3/bias/m
%:#
d2Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:, 2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, @2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
/:-@2Adam/conv2d_3/kernel/v
!:2Adam/conv2d_3/bias/v
%:#
d2Adam/dense/kernel/v
:2Adam/dense/bias/v
&:$	2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v­
 __inference__wrapped_model_22384-.<=KLab}~C¢@
9¢6
41
sequential_inputÿÿÿÿÿÿÿÿÿ  
ª "1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ³
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25746l<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿPP
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿPP 
 
(__inference_conv2d_1_layer_call_fn_25735_<=7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿPP
ª " ÿÿÿÿÿÿÿÿÿPP ³
C__inference_conv2d_2_layer_call_and_return_conditional_losses_25776lKL7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ(( 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ((@
 
(__inference_conv2d_2_layer_call_fn_25765_KL7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ(( 
ª " ÿÿÿÿÿÿÿÿÿ((@´
C__inference_conv2d_3_layer_call_and_return_conditional_losses_25833mab7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
(__inference_conv2d_3_layer_call_fn_25822`ab7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿµ
A__inference_conv2d_layer_call_and_return_conditional_losses_25716p-.9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ  
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 
&__inference_conv2d_layer_call_fn_25705c-.9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ  
ª ""ÿÿÿÿÿÿÿÿÿ  ¥
B__inference_dense_1_layer_call_and_return_conditional_losses_25920_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
'__inference_dense_1_layer_call_fn_25910R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
@__inference_dense_layer_call_and_return_conditional_losses_25901^}~0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿd
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 z
%__inference_dense_layer_call_fn_25890Q}~0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿ¶
D__inference_dropout_1_layer_call_and_return_conditional_losses_25858n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ


p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ


 ¶
D__inference_dropout_1_layer_call_and_return_conditional_losses_25870n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ


p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ


 
)__inference_dropout_1_layer_call_fn_25848a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ


p 
ª "!ÿÿÿÿÿÿÿÿÿ


)__inference_dropout_1_layer_call_fn_25853a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ


p
ª "!ÿÿÿÿÿÿÿÿÿ

²
B__inference_dropout_layer_call_and_return_conditional_losses_25801l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ²
B__inference_dropout_layer_call_and_return_conditional_losses_25813l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
'__inference_dropout_layer_call_fn_25791_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@
'__inference_dropout_layer_call_fn_25796_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@¨
B__inference_flatten_layer_call_and_return_conditional_losses_25881b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ


ª "&¢#

0ÿÿÿÿÿÿÿÿÿd
 
'__inference_flatten_layer_call_fn_25875U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ


ª "ÿÿÿÿÿÿÿÿÿdí
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25756R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_1_layer_call_fn_25751R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_25786R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_2_layer_call_fn_25781R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_25843R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_3_layer_call_fn_25838R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25726R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_max_pooling2d_layer_call_fn_25721R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
J__inference_random_contrast_layer_call_and_return_conditional_losses_25936p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 Â
J__inference_random_contrast_layer_call_and_return_conditional_losses_26633tº=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ  
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 
/__inference_random_contrast_layer_call_fn_25925c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª ""ÿÿÿÿÿÿÿÿÿ  
/__inference_random_contrast_layer_call_fn_25932gº=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ  
p
ª ""ÿÿÿÿÿÿÿÿÿ  º
F__inference_random_zoom_layer_call_and_return_conditional_losses_26649p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 ¾
F__inference_random_zoom_layer_call_and_return_conditional_losses_26751t»=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ  
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 
+__inference_random_zoom_layer_call_fn_26638c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 
ª ""ÿÿÿÿÿÿÿÿÿ  
+__inference_random_zoom_layer_call_fn_26645g»=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ  
p
ª ""ÿÿÿÿÿÿÿÿÿ  ´
D__inference_rescaling_layer_call_and_return_conditional_losses_25696l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ  
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 
)__inference_rescaling_layer_call_fn_25688_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ  
ª ""ÿÿÿÿÿÿÿÿÿ  Ð
G__inference_sequential_1_layer_call_and_return_conditional_losses_23805-.<=KLab}~K¢H
A¢>
41
sequential_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ô
G__inference_sequential_1_layer_call_and_return_conditional_losses_23852º»-.<=KLab}~K¢H
A¢>
41
sequential_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
G__inference_sequential_1_layer_call_and_return_conditional_losses_24008z-.<=KLab}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 É
G__inference_sequential_1_layer_call_and_return_conditional_losses_24870~º»-.<=KLab}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 §
,__inference_sequential_1_layer_call_fn_23502w-.<=KLab}~K¢H
A¢>
41
sequential_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ«
,__inference_sequential_1_layer_call_fn_23762{º»-.<=KLab}~K¢H
A¢>
41
sequential_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_23918m-.<=KLab}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
,__inference_sequential_1_layer_call_fn_23951qº»-.<=KLab}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿÍ
E__inference_sequential_layer_call_and_return_conditional_losses_23271P¢M
F¢C
96
random_contrast_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 Ó
E__inference_sequential_layer_call_and_return_conditional_losses_23281º»P¢M
F¢C
96
random_contrast_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 ½
E__inference_sequential_layer_call_and_return_conditional_losses_24888tA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 Ã
E__inference_sequential_layer_call_and_return_conditional_losses_25683zº»A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ  
 ¤
*__inference_sequential_layer_call_fn_22407vP¢M
F¢C
96
random_contrast_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ  ª
*__inference_sequential_layer_call_fn_23265|º»P¢M
F¢C
96
random_contrast_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª ""ÿÿÿÿÿÿÿÿÿ  
*__inference_sequential_layer_call_fn_24875gA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ  
*__inference_sequential_layer_call_fn_24884mº»A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª ""ÿÿÿÿÿÿÿÿÿ  Ä
#__inference_signature_wrapper_23889-.<=KLab}~W¢T
¢ 
MªJ
H
sequential_input41
sequential_inputÿÿÿÿÿÿÿÿÿ  "1ª.
,
dense_1!
dense_1ÿÿÿÿÿÿÿÿÿ