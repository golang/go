// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package iface

import (
	"simd"
	"unsafe"
)

// A SIMD-dependent type alias
type MyInt8s = simd.Int8s

func Generic[T haslen](x int) int {
	var v T
	return x + v.Len()
}

// Non-SIMD-typed package variable with a SIMD-dependent initializer; midway
// keeps it as a single variable initialized via per-size init proxies.
var VL = Generic[MyInt8s](1)

func GetVL() int            { return VL }
func SetVL(x int)           { VL = x }
func ReadVLFromSimdFn() int { return VL + simd.Int8s{}.Len() }

// Explicitly typed parallel N:N SIMD-dependent package variables, including
// a blank "_" sibling.
var TypedSimdA, TypedSimdB VectorC = VectorC{Field: simd.BroadcastFloat32s(11)}, VectorC{Field: simd.BroadcastFloat32s(12)}
var _, TypedBlankHasBar HasBar = VectorC{}, VectorC{Field: simd.BroadcastFloat32s(13)}

// Multi-return N:1 package variables initialized via a clean-signature SIMD dispatcher.
func pairIntFunc() (int, int) {
	n := simd.Int8s{}.Len()
	return n + 10, n + 20
}

var PairIntA, PairIntB = pairIntFunc()

// Non-SIMD-typed package variables initialized with a SIMD-using closure
// or a boxed SIMD value.
var SimdClosure = func() int { return simd.Int8s{}.Len() }
var SimdAsHasLen haslen = simd.Int8s{}

// Regression test for cloneExpr on a shared type expression containing a
// FuncLit with a short variable declaration (AssignStmt).
var CloneExprA, CloneExprB [unsafe.Sizeof(func() {
	if x := 1; x > 0 {
	}
})]int = func() [unsafe.Sizeof(func() {
	if x := 1; x > 0 {
	}
})]int {
	var r [unsafe.Sizeof(func() {
		if x := 1; x > 0 {
		}
	})]int
	r[0] = simd.Int8s{}.Len()
	return r
}(), [unsafe.Sizeof(func() {
	if x := 1; x > 0 {
	}
})]int{0: 2}

// Single package variables with SIMD-dependent types (instantiated generic
// interface, phantom generic struct, direct SIMD vector, and SIMD-field struct).
var GenericIfaceVar = MakeHasFoo(VectorC{Field: simd.BroadcastFloat32s(9)})
var PhantomVar = Phantom[simd.Int8s]{}

var VBroadcast = simd.BroadcastFloat32s(3)
var VStruct = VectorC{Field: VBroadcast}

// Reverse-declared transitive dependency chain rooted at SIMD-initialized ChainC.
var ChainA = ChainB + 10
var ChainB = chainFunc()
var ChainC = Generic[MyInt8s](ChainD)
var ChainD = 5

func chainFunc() int {
	return ChainC * 2
}

// Untyped parallel N:N mixing an init-proxied non-SIMD variable and a
// SIMD-typed variable.
var MultiA, MultiB = Generic[MyInt8s](2), VectorC{Field: simd.BroadcastFloat32s(7)}

// Plain non-SIMD package variable (control case).
var UnrelatedInit = 99

// Multi-return N:1 calling a SIMD-signature function returning mixed
// non-SIMD and SIMD types, with and without a blank "_" target.
func pairFunc() (int, VectorC) {
	return Generic[MyInt8s](2), VectorC{Field: simd.BroadcastFloat32s(7)}
}

var PairA, PairB = pairFunc()
var _, BlankPartner = pairFunc()

// Explicitly typed parallel N:N mixing SIMD-dependent and plain constant
// initializers in both orders.
var MixedHoisted1, MixedUnhoisted2 int = Generic[MyInt8s](3), 30
var MixedUnhoisted1, MixedHoisted2 int = 40, Generic[MyInt8s](4)

// Package variables across diverse Go type forms initialized directly or
// transitively from SIMD-dependent variables (ChainA..C and VStruct).
type IntAlias = int

func makeIntAlias(x int) IntAlias {
	return x
}

type EmbeddedField int

type GenericBox[T any] struct {
	Val T
}

type GenericPair[A, B any] struct {
	First  A
	Second B
}

func makeAnonIface(int) interface {
	haslen
	Extra() int
} {
	return nil
}

var HoistedUnsafePtr = makeUnsafePtr(&ChainA)
var HoistedAlias = makeIntAlias(ChainA)
var HoistedPtr = &ChainA
var HoistedSlice = []int{ChainA}
var HoistedArray = [2]int{ChainA, ChainB}
var HoistedMap = map[string]int{"a": ChainA}
var HoistedChanBidi = make(chan int, ChainA)
var HoistedChanSend = (chan<- int)(HoistedChanBidi)
var HoistedChanRecv = (<-chan int)(HoistedChanBidi)
var HoistedAnonStruct = struct {
	EmbeddedField
	X int
	Y int `tag:"val"`
}{
	EmbeddedField: EmbeddedField(ChainA),
	X:             ChainB,
	Y:             ChainC,
}
var HoistedFunc = func(a int, b ...int) int {
	sum := ChainA + a
	for _, x := range b {
		sum += x
	}
	return sum
}
var HoistedIface = makeAnonIface(ChainA)
var HoistedGeneric1 = GenericBox[int]{Val: ChainA}
var HoistedGeneric2 = GenericPair[int, VectorC]{First: ChainA, Second: VStruct}

// User-defined init() that runs after SIMD-rewritten package variables and
// itself uses SIMD.
var UserInitVar int

func init() {
	UserInitVar = ChainA + Generic[MyInt8s](0)
}

// A struct dependent on SIMD
type VectorC struct {
	Field simd.Float32s
}

type Ftype func(x any) any

var Fvar Ftype

// A dependent function with a dependent signature
func (v *VectorC) MethodOfSimd() bool {
	return false
}

func (v VectorC) Data() simd.Float32s {
	return v.Field
}

func (v VectorC) Foo(x VectorC) VectorC {
	return VectorC{Field: v.Field.Add(x.Field)}
}

func (v VectorC) Bar(x VectorC) VectorC {
	return VectorC{Field: v.Field.Add(x.Field)}
}

type Vint interface {
	MethodOfSimd() bool
}

type haslen interface {
	Len() int
}

type HasFoo[T any] interface {
	Foo(x T) T
}

type HasBar interface {
	Bar(x VectorC) VectorC
}

//go:noinline
func MakeHasFoo[T HasFoo[T]](v T) HasFoo[T] {
	return v
}

func MakeHasBar(v VectorC) HasBar {
	return v
}

func VC(x simd.Float32s) VectorC {
	return VectorC{x}
}

type EmbedBar struct {
	HasBar
}

type EmbedFoo[T HasFoo[T]] struct {
	HasFoo[T]
}

func MakeHasEmbedFoo[T HasFoo[T]](v T) EmbedFoo[T] {
	return EmbedFoo[T]{MakeHasFoo[T](v)}
}

//go:noinline
func MakeHasEmbedBar(v VectorC) EmbedBar {
	return EmbedBar{MakeHasBar(v)}
}

type HasQux[T any] interface {
	Qux(x T) HasFoo[T]
}

type Q struct {
	q *Q
}

func (q *Q) Qux(v VectorC) HasFoo[VectorC] {
	return v
}

// Type alias receiver declaring an additional SIMD-dependent method on Q.
type QAlias = Q

func (q *QAlias) QuxAlias(v VectorC) HasFoo[VectorC] {
	return v
}

// Clean-signature methods on a non-SIMD receiver exercising unnamed,
// blank ("_"), and named receiver forwarding in dispatchers.
type PlainRecv struct {
	Base int
}

func (PlainRecv) UnnamedRecvSimd() int {
	return Generic[MyInt8s](0)
}

func (_ PlainRecv) BlankRecvSimd() int {
	return Generic[MyInt8s](0)
}

func (_ PlainRecv) BlankRecvSimdParam(_simdRecv int) int {
	return _simdRecv + Generic[MyInt8s](0)
}

func (p PlainRecv) NamedRecvSimd() int {
	return p.Base + Generic[MyInt8s](0)
}

// Clean-signature function with a blank ("_") parameter forwarded by its dispatcher.
func CleanBlankParam(_ int, x int) int {
	return x + Generic[MyInt8s](0)
}

// Clean-signature SIMD function exercising diverse Go statement forms in
// midway's AST DeepCopier.
func DeepCopyStmts(x int) int {
	v := simd.BroadcastInt8s(int8(x))
	m := map[int]int{0: v.Len()}
	ch := make(chan int, 2)
	defer func() {}()
	ch <- m[0]
	select {
	case got := <-ch:
		x += got
	default:
	}
EmptyLabel:
	;
	if false {
		goto EmptyLabel
	}
	arr := [2]int{1, 2}
	var idx int
	for idx = range arr {
		x += idx
	}
	x, newVar := x+1, v.Len()
	x += newVar
	if noop := func() {}; true {
		for noop(); false; noop() {
		}
	}
	return x
}

// Zero-sized generic struct whose SIMD dependence comes solely from its
// type argument.
type Phantom[T haslen] struct{}

func (Phantom[T]) GetLen() int {
	var v T
	return v.Len()
}

// Mutually recursive struct cycle reaching simd.Int8s through CycB
// (Tarjan SCC coverage).
type CycA struct{ B *CycB }
type CycB struct {
	C *CycC
	V simd.Int8s
}
type CycC struct{ A *CycA }

func CycLen() int {
	a := &CycA{}
	b := &CycB{}
	c := &CycC{A: a}
	a.B = b
	b.C = c
	return c.A.B.V.Len()
}

// Generic struct with no SIMD fields whose SIMD dependence comes solely
// from a method signature.
type GenMethodRecv[T any] struct{ X T }

func (g GenMethodRecv[T]) Method(v simd.Int8s) int {
	return v.Len()
}

func CallGenMethodRecv() int {
	g := GenMethodRecv[int]{X: 1}
	return g.X + g.Method(simd.Int8s{})
}

// Generic types, aliases, blank type parameters, and functions constrained
// by named, inline, and self-referential SIMD type unions.
type VecUnion interface {
	simd.Int8s | simd.Int16s
	Len() int
}

type ConstrainedBox[T VecUnion] struct {
	V T
}

func (b ConstrainedBox[T]) Len() int {
	return b.V.Len()
}

type ConstrainedAlias[T VecUnion] = []T

type BlankConstraintParam[_ VecUnion] struct {
	Tag int
}

func ConstrainedVecLen[T VecUnion](v T) int {
	return v.Len()
}

func ConstrainedVecSize[T VecUnion]() int {
	var z T
	return z.Len()
}

type InlineUnionBox[T simd.Int8s | simd.Int16s] struct {
	V T
}

type SelfVec[T SelfVec[T]] interface {
	simd.Int8s | simd.Int16s
	Add(T) T
	Len() int
}

func SelfVecAddLen[T SelfVec[T]](a, b T) int {
	return a.Add(b).Len()
}

// Type whose only SIMD-dependent method is declared on a type-alias receiver.
type QOnlyAliasBase struct{ X int }
type QOnlyAlias = QOnlyAliasBase

func (q *QOnlyAlias) Qux(v VectorC) HasFoo[VectorC] {
	return v
}

// Defined types wrapping a base SIMD type directly and transitively.
type MySimd simd.Int8s
type MySimd2 MySimd

func (m MySimd) AddOne() MySimd {
	return MySimd(simd.Int8s(m).Add(simd.BroadcastInt8s(1)))
}

// Multiple blank SIMD type declarations must not be suffixed or collide.
type _ = simd.Int8s
type _ = simd.Int8s
type _ simd.Int8s
type _ simd.Int8s

// Multiple blank package-level SIMD variables must not be suffixed or collide.
var _ simd.Int8s
var _ simd.Int8s

// Function-local SIMD type must not get a package-scope suffix.
func LocalSimdTypeLen() int {
	type localSimd simd.Int8s
	var v localSimd
	return simd.Int8s(v).Len()
}

// Multiple blank SIMD functions and methods must not be suffixed or collide.
type BlankMethodHost struct{}

func _(simd.Int8s) {}
func _(simd.Int8s) {}
func _() {
	var v simd.Int8s
	_ = v
}
func _() {
	var v simd.Int8s
	_ = v
}
func (BlankMethodHost) _(simd.Int8s) {}
func (BlankMethodHost) _(simd.Int8s) {}
func (BlankMethodHost) _() {
	var v simd.Int8s
	_ = v
}
func (BlankMethodHost) _() {
	var v simd.Int8s
	_ = v
}
func (VectorC) _(simd.Int8s) {}
func (VectorC) _(simd.Int8s) {}

// Zero-weight instantiation cycle with 7! permutations; analysis must not
// expand instantiated *types2.Named types.
type Perm[A, B, C, D, E, F, G any] struct {
	Swap *Perm[B, A, C, D, E, F, G]
	Rot  *Perm[B, C, D, E, F, G, A]
}

func PermLen() int {
	var p Perm[simd.Int8s, int8, int16, int32, int64, uint8, uint16]
	var z simd.Int8s
	_ = p
	return z.Len()
}

// Package-level variables exercising SIMD-union-constrained generic types
// and functions.
var (
	ConstrainedBoxVar   = ConstrainedBox[simd.Int8s]{V: simd.BroadcastInt8s(1)}
	ConstrainedAliasVar = ConstrainedAlias[simd.Int8s]{simd.BroadcastInt8s(2)}
	ConstrainedBlankVar = BlankConstraintParam[simd.Int8s]{Tag: 7}
	ConstrainedLenVar   = ConstrainedVecLen(simd.BroadcastInt8s(3))
	ConstrainedSizeVar  = ConstrainedVecSize[simd.Int16s]()
	InlineUnionBoxVar   = InlineUnionBox[simd.Int8s]{V: simd.BroadcastInt8s(4)}
	SelfVecAddLenVar    = SelfVecAddLen(simd.BroadcastInt8s(1), simd.BroadcastInt8s(2))
)
