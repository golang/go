// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package testdata_test

import (
	"reflect"
	"simd"
	"simd/testdata/iface"
	"testing"
)

func TestIFaceFoo(t *testing.T) {
	u := simd.BroadcastFloat32s(4)
	v := simd.BroadcastFloat32s(1)
	vc := iface.VC(v)
	uc := iface.VC(u)

	hv := iface.MakeHasFoo(vc) // generic interface w/ Foo method

	sum := hv.Foo(uc)

	s := make([]float32, u.Len())

	sum.Data().Store(s) // The method of a dependent type works.

	if s[0] != 5 {
		t.Errorf("(from Data()) expected 5, got %f", s[0])
	}

	sum.Field.Store(s)

	if s[0] != 5 {
		t.Errorf("(from Field) expected 5, got %f", s[0])
	}
}

func TestIFaceBar(t *testing.T) {
	u := simd.BroadcastFloat32s(4)
	v := simd.BroadcastFloat32s(1)
	vc := iface.VC(v)
	uc := iface.VC(u)

	hv := iface.MakeHasBar(vc) // non-generic interface w/ Foo method

	sum := hv.Bar(uc)

	s := make([]float32, u.Len())

	sum.Data().Store(s) // The method of a dependent type works.

	if s[0] != 5 {
		t.Errorf("(from Data()) expected 5, got %f", s[0])
	}

	sum.Field.Store(s)

	if s[0] != 5 {
		t.Errorf("(from Field) expected 5, got %f", s[0])
	}
}

func TestIFaceEmbedFoo(t *testing.T) {
	u := simd.BroadcastFloat32s(4)
	v := simd.BroadcastFloat32s(1)
	vc := iface.VC(v)
	uc := iface.VC(u)

	hv := iface.MakeHasEmbedFoo(vc) // generic interface w/ Foo method

	sum := hv.Foo(uc)

	s := make([]float32, u.Len())

	sum.Data().Store(s) // The method of a dependent type works.

	if s[0] != 5 {
		t.Errorf("(from Data()) expected 5, got %f", s[0])
	}

	sum.Field.Store(s)

	if s[0] != 5 {
		t.Errorf("(from Field) expected 5, got %f", s[0])
	}

	rv := reflect.ValueOf(hv)
	rt := rv.Type()

	t.Logf("reflect.value is %v", rv)
	t.Logf("reflect.type is %v", rt)
}

func TestIFaceEmbedBar(t *testing.T) {
	u := simd.BroadcastFloat32s(4)
	v := simd.BroadcastFloat32s(1)
	vc := iface.VC(v)
	uc := iface.VC(u)

	hv := iface.MakeHasEmbedBar(vc) // generic interface w/ Foo method

	sum := hv.Bar(uc)

	s := make([]float32, u.Len())

	sum.Data().Store(s) // The method of a dependent type works.

	if s[0] != 5 {
		t.Errorf("(from Data()) expected 5, got %f", s[0])
	}

	sum.Field.Store(s)

	if s[0] != 5 {
		t.Errorf("(from Field) expected 5, got %f", s[0])
	}

	rv := reflect.ValueOf(hv)
	rt := rv.Type()

	t.Logf("reflect.value is %v", rv)
	t.Logf("reflect.type is %v", rt)
}

func TestIFaceVL(t *testing.T) {
	var v simd.Int8s
	if a, b := iface.VL, iface.Generic[simd.Int8s](1); a != b {
		t.Errorf("expected iface.VL [%d] == iface.Generic[simd.Int8s](1) [%d], but not true", a, b)
	}
	if a, b := iface.VL, v.Len()+1; a != b {
		t.Errorf("expected iface.VL [%d] == v.Len()+1 [%d], but not true", a, b)
	}
}

// TestTypeDependenceCycles validates SIMD type-dependence analysis on cyclic
// type graphs:
//   - CycLen exercises a mutually recursive struct cycle (CycA -> *CycB ->
//     *CycC -> *CycA) where CycB holds a simd.Int8s field; Tarjan SCC analysis
//     must mark the entire cycle dependent rather than prematurely memoizing
//     false on entry.
//   - PermLen and iface.Perm[...] exercise a zero-weight generic instantiation
//     cycle with 7! type-argument permutations (Swap/Rot); analysis must not
//     expand instantiated *types2.Named types, avoiding combinatorial blowup.
func TestTypeDependenceCycles(t *testing.T) {
	n8 := (simd.Int8s{}).Len()

	if got := iface.CycLen(); got != n8 {
		t.Errorf("expected CycLen() == %d, got %d", n8, got)
	}
	if got := iface.PermLen(); got != n8 {
		t.Errorf("expected PermLen() == %d, got %d", n8, got)
	}
	var perm iface.Perm[simd.Int8s, int8, int16, int32, int64, uint8, uint16]
	_ = perm
}

// TestAliasAndGenericMethodReceivers validates SIMD type-dependence discovery
// through method sets declared on type-alias and generic receivers:
//   - QAlias (alias to Q) has SIMD-dependent methods declared on both *Q (Qux)
//     and *QAlias (QuxAlias), tested across package boundaries via HasQux and
//     direct calls.
//   - QOnlyAlias (alias to QOnlyAliasBase) has no SIMD fields and its only
//     SIMD-dependent method (Qux) is declared on the alias receiver *QOnlyAlias;
//     dependence analysis must still mark the underlying named type dependent so
//     cross-package HasQux[VectorC] satisfaction works.
//   - GenMethodRecv[T] has no SIMD fields and depends on SIMD solely through
//     its method signature Method(simd.Int8s), tested both intra-package
//     (CallGenMethodRecv) and instantiated cross-package (GenMethodRecv[int]).
func TestAliasAndGenericMethodReceivers(t *testing.T) {
	n8 := (simd.Int8s{}).Len()
	vc := iface.VC(simd.BroadcastFloat32s(3))
	uc := iface.VC(simd.BroadcastFloat32s(4))
	s32 := make([]float32, (simd.Float32s{}).Len())

	var hQux iface.HasQux[iface.VectorC] = &iface.QAlias{}
	hQux.Qux(vc).Foo(uc).Data().Store(s32)
	if s32[0] != 7 {
		t.Errorf("expected QAlias.Qux.Foo == 7, got %f", s32[0])
	}

	clear(s32)
	(&iface.QAlias{}).QuxAlias(vc).Foo(uc).Data().Store(s32)
	if s32[0] != 7 {
		t.Errorf("expected QAlias.QuxAlias.Foo == 7, got %f", s32[0])
	}

	clear(s32)
	var hOnlyAlias iface.HasQux[iface.VectorC] = &iface.QOnlyAlias{}
	hOnlyAlias.Qux(vc).Foo(uc).Data().Store(s32)
	if s32[0] != 7 {
		t.Errorf("expected QOnlyAlias.Qux.Foo == 7, got %f", s32[0])
	}

	if got := iface.CallGenMethodRecv(); got != 1+n8 {
		t.Errorf("expected CallGenMethodRecv() == %d, got %d", 1+n8, got)
	}
	g := iface.GenMethodRecv[int]{X: 2}
	if got := g.X + g.Method(simd.Int8s{}); got != 2+n8 {
		t.Errorf("expected cross-pkg GenMethodRecv == %d, got %d", 2+n8, got)
	}
}

// TestDefinedAndLocalSimdTypes validates defined wrapper types over base SIMD
// types and function-local SIMD type declarations:
//   - MySimd (type MySimd simd.Int8s) and MySimd2 (type MySimd2 MySimd) test
//     direct and transitive defined SIMD wrappers, specialized wrapper methods
//     (AddOne), and conversions back to simd.Int8s.
//   - LocalSimdTypeLen tests that a function-local defined SIMD type does not
//     receive a package-scope @simd suffix when the enclosing function is
//     specialized.
func TestDefinedAndLocalSimdTypes(t *testing.T) {
	n8 := (simd.Int8s{}).Len()

	m := iface.MySimd(simd.BroadcastInt8s(5)).AddOne()
	m2 := iface.MySimd2(m)
	s8 := make([]int8, n8)
	simd.Int8s(m2).Store(s8)
	if s8[0] != 6 {
		t.Errorf("expected MySimd.AddOne == 6, got %d", s8[0])
	}

	if got := iface.LocalSimdTypeLen(); got != n8 {
		t.Errorf("expected LocalSimdTypeLen == %d, got %d", n8, got)
	}
}

// TestGenericTypeDependence validates SIMD type-dependence analysis and
// specialization for generic types, aliases, and functions whose type arguments
// or type constraints reference SIMD types:
//   - Phantom[simd.Int8s], a zero-sized generic struct whose only SIMD
//     dependence is its type argument (used inside GetLen, not in struct fields).
//   - Constrained generic structs (ConstrainedBox, InlineUnionBox), a generic
//     alias (ConstrainedAlias), and a blank constrained type parameter
//     (BlankConstraintParam) constrained by named (VecUnion), inline
//     (simd.Int8s|simd.Int16s), or self-referential (SelfVec[T SelfVec[T]])
//     SIMD type unions.
//   - Constrained generic functions where T appears in parameter types
//     (ConstrainedVecLen, SelfVecAddLen) or only in the type parameter list
//     (ConstrainedVecSize).
func TestGenericTypeDependence(t *testing.T) {
	n8 := (simd.Int8s{}).Len()
	n16 := (simd.Int16s{}).Len()

	if got := (iface.Phantom[simd.Int8s]{}).GetLen(); got != n8 {
		t.Errorf("expected Phantom.GetLen() == %d, got %d", n8, got)
	}

	crossBox := iface.ConstrainedBox[simd.Int8s]{V: simd.BroadcastInt8s(5)}
	if got := crossBox.Len(); got != n8 {
		t.Errorf("expected cross-pkg ConstrainedBox.Len() == %d, got %d", n8, got)
	}
	alias := iface.ConstrainedAlias[simd.Int8s]{simd.BroadcastInt8s(2)}
	if len(alias) != 1 || alias[0].Len() != n8 {
		t.Errorf("unexpected ConstrainedAlias len=%d", len(alias))
	}
	if bp := (iface.BlankConstraintParam[simd.Int8s]{Tag: 7}); bp.Tag != 7 {
		t.Errorf("expected BlankConstraintParam.Tag == 7, got %d", bp.Tag)
	}
	if got := iface.ConstrainedVecLen(simd.BroadcastInt8s(5)); got != n8 {
		t.Errorf("expected cross-pkg ConstrainedVecLen == %d, got %d", n8, got)
	}
	if got := iface.ConstrainedVecSize[simd.Int16s](); got != n16 {
		t.Errorf("expected cross-pkg ConstrainedVecSize[Int16s] == %d, got %d", n16, got)
	}
	if got := iface.ConstrainedVecSize[simd.Int8s](); got != n8 {
		t.Errorf("expected cross-pkg ConstrainedVecSize == %d, got %d", n8, got)
	}
	if ib := (iface.InlineUnionBox[simd.Int8s]{V: simd.BroadcastInt8s(4)}); ib.V.Len() != n8 {
		t.Errorf("expected InlineUnionBox.V.Len() == %d, got %d", n8, ib.V.Len())
	}
	if got := iface.SelfVecAddLen(simd.BroadcastInt8s(1), simd.BroadcastInt8s(2)); got != n8 {
		t.Errorf("expected cross-pkg SelfVecAddLen == %d, got %d", n8, got)
	}
}

// TestCleanDispatchersAndDeepCopy validates clean-signature dispatcher
// generation and AST deep-copying for functions/methods whose signatures do not
// mention SIMD but whose bodies do:
//   - Methods on non-SIMD receiver PlainRecv with unnamed, blank ("_"), and
//     named receivers, verifying unnamed and blank receivers receive synthetic
//     names so dispatchers can forward calls to specialized methods.
//   - CleanBlankParam, verifying blank ("_") parameters in clean-signature
//     functions are named in the dispatcher so arguments forward properly.
//   - DeepCopyStmts, verifying midway's DeepCopier handles diverse statement
//     forms (map literal, channel send/recv, defer, select with default, labeled
//     empty statement, goto, for-range, mixed ":=", and if/for init statements).
func TestCleanDispatchersAndDeepCopy(t *testing.T) {
	n8 := (simd.Int8s{}).Len()

	pr := iface.PlainRecv{Base: 10}
	if got := pr.UnnamedRecvSimd(); got != n8 {
		t.Errorf("expected pr.UnnamedRecvSimd() == %d, got %d", n8, got)
	}
	if got := pr.BlankRecvSimd(); got != n8 {
		t.Errorf("expected pr.BlankRecvSimd() == %d, got %d", n8, got)
	}
	if got := pr.BlankRecvSimdParam(7); got != 7+n8 {
		t.Errorf("expected pr.BlankRecvSimdParam(7) == %d, got %d", 7+n8, got)
	}
	if got := pr.NamedRecvSimd(); got != 10+n8 {
		t.Errorf("expected pr.NamedRecvSimd() == %d, got %d", 10+n8, got)
	}
	if got := iface.CleanBlankParam(999, 5); got != 5+n8 {
		t.Errorf("expected CleanBlankParam(999, 5) == %d, got %d", 5+n8, got)
	}
	if got := iface.DeepCopyStmts(1); got != 3+2*n8 {
		t.Errorf("expected DeepCopyStmts(1) == %d, got %d", 3+2*n8, got)
	}
}

// Compile-time regression test: multiple blank package-level SIMD type and
// function declarations must not be suffixed or collide during specialization.
type _ = simd.Int8s
type _ = simd.Int8s

func _(simd.Int8s) {}
func _(simd.Int8s) {}
