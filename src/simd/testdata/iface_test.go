// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

package testdata_test

import (
	"reflect"
	"simd"
	"simd/testdata/iface"
	"slices"
	"testing"
	"unsafe"
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

// TestIFaceVL validates a non-SIMD-typed package variable (iface.VL, type int)
// initialized via a SIMD-dependent generic call using a SIMD type alias
// (Generic[MyInt8s](1)). Because VL's type does not depend on SIMD, midway
// initializes it through per-size init proxies rather than splitting VL into
// per-size variables; cross-package writes, clean functions (GetVL/SetVL), and
// SIMD-specialized functions (ReadVLFromSimdFn) must all access the same single
// variable.
func TestIFaceVL(t *testing.T) {
	n8 := (simd.Int8s{}).Len()

	if a, b := iface.VL, iface.Generic[simd.Int8s](1); a != b {
		t.Errorf("expected iface.VL [%d] == iface.Generic[simd.Int8s](1) [%d], but not true", a, b)
	}
	if a, b := iface.VL, n8+1; a != b {
		t.Errorf("expected iface.VL [%d] == (simd.Int8s{}).Len()+1 [%d], but not true", a, b)
	}

	origVL := iface.VL
	defer func() { iface.VL = origVL }()
	iface.VL = 1000
	if got := iface.GetVL(); got != 1000 {
		t.Errorf("after VL=1000, GetVL() = %d, want 1000", got)
	}
	if got := iface.ReadVLFromSimdFn(); got != 1000+n8 {
		t.Errorf("after VL=1000, ReadVLFromSimdFn() = %d, want %d", got, 1000+n8)
	}
	iface.SetVL(7)
	if iface.VL != 7 || iface.GetVL() != 7 {
		t.Errorf("after SetVL(7), VL=%d GetVL()=%d, want 7, 7", iface.VL, iface.GetVL())
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
// or type constraints reference SIMD types, covering both cross-package
// instantiations and package-level variables initialized inside iface:
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

	// Cross-package instantiations of SIMD-constrained generic types and functions.
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

	// Package-level variables initialized inside package iface.
	if got := iface.ConstrainedBoxVar.Len(); got != n8 {
		t.Errorf("expected ConstrainedBoxVar.Len() == %d, got %d", n8, got)
	}
	if len(iface.ConstrainedAliasVar) != 1 || iface.ConstrainedAliasVar[0].Len() != n8 {
		t.Errorf("unexpected ConstrainedAliasVar len=%d", len(iface.ConstrainedAliasVar))
	}
	if iface.ConstrainedBlankVar.Tag != 7 {
		t.Errorf("expected ConstrainedBlankVar.Tag == 7, got %d", iface.ConstrainedBlankVar.Tag)
	}
	if iface.ConstrainedLenVar != n8 {
		t.Errorf("expected ConstrainedLenVar == %d, got %d", n8, iface.ConstrainedLenVar)
	}
	if iface.ConstrainedSizeVar != n16 {
		t.Errorf("expected ConstrainedSizeVar == %d, got %d", n16, iface.ConstrainedSizeVar)
	}
	if got := iface.InlineUnionBoxVar.V.Len(); got != n8 {
		t.Errorf("expected InlineUnionBoxVar.V.Len() == %d, got %d", n8, got)
	}
	if iface.SelfVecAddLenVar != n8 {
		t.Errorf("expected SelfVecAddLenVar == %d, got %d", n8, iface.SelfVecAddLenVar)
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

// TestPackageVarSimdValues validates single package-level variables whose types
// or initializers involve SIMD values:
//   - Direct SIMD vector (VBroadcast) and SIMD-field struct (VStruct) variables,
//     which midway splits into per-size copies (<name>@simd<size>).
//   - Instantiated generic interface (GenericIfaceVar, HasFoo[VectorC]) and
//     zero-sized generic struct (PhantomVar, Phantom[simd.Int8s]), whose SIMD
//     dependence comes solely from their type arguments.
//   - Non-SIMD-typed variables holding a SIMD-using closure (SimdClosure) or a
//     boxed SIMD value in a non-SIMD interface (SimdAsHasLen), which remain
//     single variables initialized via per-size init proxies.
func TestPackageVarSimdValues(t *testing.T) {
	n8 := (simd.Int8s{}).Len()
	s := make([]float32, (simd.Float32s{}).Len())

	iface.VBroadcast.Store(s)
	if s[0] != 3 {
		t.Errorf("expected VBroadcast[0] == 3, got %f", s[0])
	}

	clear(s)
	iface.VStruct.Data().Store(s)
	if s[0] != 3 {
		t.Errorf("expected VStruct.Data()[0] == 3, got %f", s[0])
	}

	clear(s)
	iface.GenericIfaceVar.Foo(iface.VC(simd.BroadcastFloat32s(1))).Data().Store(s)
	if s[0] != 10 {
		t.Errorf("expected GenericIfaceVar.Foo(1)[0] == 10, got %f", s[0])
	}

	if got := iface.PhantomVar.GetLen(); got != n8 {
		t.Errorf("expected PhantomVar.GetLen() == %d, got %d", n8, got)
	}
	if got := iface.SimdClosure(); got != n8 {
		t.Errorf("expected SimdClosure() == %d, got %d", n8, got)
	}
	if got := iface.SimdAsHasLen.Len(); got != n8 {
		t.Errorf("expected SimdAsHasLen.Len() == %d, got %d", n8, got)
	}
}

// TestPackageVarChainsAndMultiFile validates transitive package-variable
// dependency chains, user-defined func init(), and multi-file SIMD rewriting:
//   - ChainA -> ChainB (via chainFunc) -> ChainC (SIMD init proxy) -> ChainD
//     (plain constant 5), declared in reverse dependency order.
//   - UserInitVar, assigned in iface's func init() using both ChainA and a
//     SIMD-dependent generic call, ensuring user init() runs after SIMD
//     package-var initializers and is itself dispatched.
//   - UnrelatedInit, verifying plain non-SIMD package vars remain unaffected.
//   - ExtraSimdLen and ExtraSimdInit, defined in iface/extra.go (which does not
//     import "simd"), verifying midway injects its private imports and rewrites
//     both functions and package-var initializers across all files in a package.
func TestPackageVarChainsAndMultiFile(t *testing.T) {
	n8 := (simd.Int8s{}).Len()

	wantC := 5 + n8
	wantB := wantC * 2
	wantA := wantB + 10
	if iface.ChainD != 5 {
		t.Errorf("expected ChainD == 5, got %d", iface.ChainD)
	}
	if iface.ChainC != wantC {
		t.Errorf("expected ChainC == %d, got %d", wantC, iface.ChainC)
	}
	if iface.ChainB != wantB {
		t.Errorf("expected ChainB == %d, got %d", wantB, iface.ChainB)
	}
	if iface.ChainA != wantA {
		t.Errorf("expected ChainA == %d, got %d", wantA, iface.ChainA)
	}

	wantUserInit := wantA + n8
	if iface.UserInitVar != wantUserInit {
		t.Errorf("expected UserInitVar == %d, got %d", wantUserInit, iface.UserInitVar)
	}
	if iface.UnrelatedInit != 99 {
		t.Errorf("expected UnrelatedInit == 99, got %d", iface.UnrelatedInit)
	}

	if got := iface.ExtraSimdLen(); got != n8 {
		t.Errorf("expected ExtraSimdLen() == %d, got %d", n8, got)
	}
	if iface.ExtraSimdInit != 2+n8 {
		t.Errorf("expected ExtraSimdInit == %d, got %d", 2+n8, iface.ExtraSimdInit)
	}
}

// TestPackageVarMultiDecl validates midway's rewriting of multi-variable
// package declarations (both parallel N:N and multi-return N:1 forms):
//   - Untyped parallel N:N mixing an init-proxied non-SIMD var and a SIMD-typed
//     var (MultiA, MultiB).
//   - Explicitly typed parallel N:N with SIMD-dependent types (TypedSimdA,
//     TypedSimdB) and a blank "_" sibling (_, TypedBlankHasBar), which splits
//     the declaration and clones the shared type AST.
//   - Explicitly typed parallel N:N mixing SIMD-dependent and plain initializers
//     in both orders (MixedHoisted1/MixedUnhoisted2, MixedUnhoisted1/MixedHoisted2).
//   - Parallel N:N sharing a complex array-length type expression containing a
//     FuncLit with a short variable declaration (CloneExprA, CloneExprB),
//     regression-testing cloneExpr.
//   - Multi-return N:1 calling a SIMD-signature function returning mixed
//     non-SIMD and SIMD types (PairA/PairB and _, BlankPartner), and calling a
//     clean-signature SIMD dispatcher returning only non-SIMD types (PairIntA/PairIntB).
func TestPackageVarMultiDecl(t *testing.T) {
	n8 := (simd.Int8s{}).Len()
	s := make([]float32, (simd.Float32s{}).Len())

	// Parallel N:N untyped (mixed init-proxied int and SIMD-typed VectorC).
	if iface.MultiA != 2+n8 {
		t.Errorf("expected MultiA == %d, got %d", 2+n8, iface.MultiA)
	}
	iface.MultiB.Data().Store(s)
	if s[0] != 7 {
		t.Errorf("expected MultiB.Data()[0] == 7, got %f", s[0])
	}

	// Multi-return N:1 with SIMD-dependent return type, with and without blank "_".
	if iface.PairA != 2+n8 {
		t.Errorf("expected PairA == %d, got %d", 2+n8, iface.PairA)
	}
	clear(s)
	iface.PairB.Data().Store(s)
	if s[0] != 7 {
		t.Errorf("expected PairB.Data()[0] == 7, got %f", s[0])
	}
	clear(s)
	iface.BlankPartner.Data().Store(s)
	if s[0] != 7 {
		t.Errorf("expected BlankPartner.Data()[0] == 7, got %f", s[0])
	}

	// Multi-return N:1 calling a clean-signature dispatcher returning (int, int).
	if iface.PairIntA != n8+10 || iface.PairIntB != n8+20 {
		t.Errorf("expected PairIntA=%d PairIntB=%d, got %d %d", n8+10, n8+20, iface.PairIntA, iface.PairIntB)
	}

	// Parallel N:N explicitly typed mixing SIMD-dependent and plain initializers.
	if iface.MixedHoisted1 != 3+n8 || iface.MixedUnhoisted2 != 30 {
		t.Errorf("expected MixedHoisted1=%d MixedUnhoisted2=30, got %d %d", 3+n8, iface.MixedHoisted1, iface.MixedUnhoisted2)
	}
	if iface.MixedUnhoisted1 != 40 || iface.MixedHoisted2 != 4+n8 {
		t.Errorf("expected MixedUnhoisted1=40 MixedHoisted2=%d, got %d %d", 4+n8, iface.MixedUnhoisted1, iface.MixedHoisted2)
	}

	// Parallel N:N explicitly typed with SIMD-dependent types, including blank "_".
	clear(s)
	iface.TypedSimdA.Data().Store(s)
	if s[0] != 11 {
		t.Errorf("expected TypedSimdA[0] == 11, got %f", s[0])
	}
	clear(s)
	iface.TypedSimdB.Data().Store(s)
	if s[0] != 12 {
		t.Errorf("expected TypedSimdB[0] == 12, got %f", s[0])
	}
	clear(s)
	iface.TypedBlankHasBar.Bar(iface.VC(simd.BroadcastFloat32s(1))).Data().Store(s)
	if s[0] != 14 {
		t.Errorf("expected TypedBlankHasBar.Bar(1)[0] == 14, got %f", s[0])
	}

	// Parallel N:N cloning a shared type expression containing a FuncLit with ":=".
	if iface.CloneExprA[0] != n8 || iface.CloneExprB[0] != 2 {
		t.Errorf("expected CloneExprA[0]=%d CloneExprB[0]=2, got %d %d", n8, iface.CloneExprA[0], iface.CloneExprB[0])
	}
}

// TestPackageVarDiverseTypes validates package-level variables across diverse
// Go type forms (unsafe.Pointer, type alias, pointer, slice, array, map,
// bidirectional/directional channels, anonymous struct with embedding and tags,
// variadic function, anonymous interface, and generic structs with non-SIMD and
// SIMD type arguments) initialized directly or transitively from SIMD-dependent
// values (ChainA..C and VStruct). It also verifies via HoistedUnsafePtr and
// HoistedPtr that a non-SIMD variable (ChainA) retains a single address shared
// between package initialization and external callers.
func TestPackageVarDiverseTypes(t *testing.T) {
	n8 := (simd.Int8s{}).Len()
	wantC := 5 + n8
	wantB := wantC * 2
	wantA := wantB + 10
	s := make([]float32, (simd.Float32s{}).Len())

	if *(*int)(iface.HoistedUnsafePtr) != wantA {
		t.Errorf("expected *HoistedUnsafePtr == %d, got %d", wantA, *(*int)(iface.HoistedUnsafePtr))
	}
	if iface.HoistedUnsafePtr != unsafe.Pointer(&iface.ChainA) {
		t.Errorf("expected HoistedUnsafePtr == &ChainA (%p), got %p", &iface.ChainA, iface.HoistedUnsafePtr)
	}
	if iface.HoistedAlias != wantA {
		t.Errorf("expected HoistedAlias == %d, got %d", wantA, iface.HoistedAlias)
	}
	if *iface.HoistedPtr != wantA {
		t.Errorf("expected *HoistedPtr == %d, got %d", wantA, *iface.HoistedPtr)
	}
	if iface.HoistedPtr != &iface.ChainA {
		t.Errorf("expected HoistedPtr == &ChainA (%p), got %p", &iface.ChainA, iface.HoistedPtr)
	}
	if len(iface.HoistedSlice) != 1 || iface.HoistedSlice[0] != wantA {
		t.Errorf("expected HoistedSlice == [%d], got %v", wantA, iface.HoistedSlice)
	}
	if iface.HoistedArray != [2]int{wantA, wantB} {
		t.Errorf("expected HoistedArray == [%d %d], got %v", wantA, wantB, iface.HoistedArray)
	}
	if iface.HoistedMap["a"] != wantA {
		t.Errorf("expected HoistedMap[\"a\"] == %d, got %d", wantA, iface.HoistedMap["a"])
	}
	if cap(iface.HoistedChanBidi) != wantA || iface.HoistedChanSend == nil || iface.HoistedChanRecv == nil {
		t.Errorf("unexpected HoistedChan state: cap=%d", cap(iface.HoistedChanBidi))
	}
	if int(iface.HoistedAnonStruct.EmbeddedField) != wantA || iface.HoistedAnonStruct.X != wantB || iface.HoistedAnonStruct.Y != wantC {
		t.Errorf("unexpected HoistedAnonStruct: %+v", iface.HoistedAnonStruct)
	}
	if got := iface.HoistedFunc(1, 2, 3); got != wantA+6 {
		t.Errorf("expected HoistedFunc(1, 2, 3) == %d, got %d", wantA+6, got)
	}
	if iface.HoistedIface != nil {
		t.Errorf("expected HoistedIface == nil, got %v", iface.HoistedIface)
	}
	if iface.HoistedGeneric1.Val != wantA {
		t.Errorf("expected HoistedGeneric1.Val == %d, got %d", wantA, iface.HoistedGeneric1.Val)
	}
	if iface.HoistedGeneric2.First != wantA {
		t.Errorf("expected HoistedGeneric2.First == %d, got %d", wantA, iface.HoistedGeneric2.First)
	}
	iface.HoistedGeneric2.Second.Data().Store(s)
	if s[0] != 3 {
		t.Errorf("expected HoistedGeneric2.Second.Data()[0] == 3, got %f", s[0])
	}
}

// TestPackageVarInitOrder validates that midway's InitOrderPlan preserves exact
// Go-spec package-variable initialization order after AST expansion (including
// under non-default SIMD configurations such as GODEBUG=simd=0):
//   - InitTrace / Order*: interleaving of plain vars, SIMD-initialized vars,
//     single and multi-return blank ("_") side-effect vars, and a reverse-
//     declared chain depending on a later SIMD-typed var (OrderRevVec).
//     OrderMiddle must initialize before OrderRevVec, after which OrderRevMid
//     and OrderRevAfter unblock ahead of OrderAfter.
//   - C1Trace / C1*: atomic placement of per-size specializations when a
//     SIMD-typed var (C1V) is declared after multiple dependents (C1A -> C1X ->
//     C1V and C1Y -> C1V); C1A sets shared state read by C1Y, so non-default
//     specializations of C1Y must not run before C1A.
//   - C1DTrace / C1D*: same atomic-order invariant when the earlier dependent
//     (C1DX) reaches the SIMD-typed var (C1DV) indirectly through a clean-
//     signature dispatcher function (c1dGetVLen) while C1DY references C1DV
//     directly.
func TestPackageVarInitOrder(t *testing.T) {
	n8 := (simd.Int8s{}).Len()

	wantTrace := []string{"before", "simdInit", "blank1", "blank2", "blankPair", "middle", "revVec", "revMid", "revAfter", "after"}
	if !slices.Equal(iface.InitTrace, wantTrace) {
		t.Errorf("expected InitTrace == %v, got %v", wantTrace, iface.InitTrace)
	}
	if iface.OrderBefore != 1 || iface.OrderSimdInit != n8 || iface.OrderMiddle != 2 || iface.OrderRevMid != n8 || iface.OrderRevAfter != n8+1 || iface.OrderAfter != 3 {
		t.Errorf("unexpected Order* values: before=%d simdInit=%d middle=%d revMid=%d revAfter=%d after=%d",
			iface.OrderBefore, iface.OrderSimdInit, iface.OrderMiddle, iface.OrderRevMid, iface.OrderRevAfter, iface.OrderAfter)
	}

	wantC1Trace := []string{"V", "X", "A", "Y"}
	if !slices.Equal(iface.C1Trace, wantC1Trace) || iface.C1X != n8 || iface.C1A != n8 || iface.C1Y != 100+n8 {
		t.Errorf("C1 init order mismatch: trace=%v (want %v) X=%d A=%d Y=%d (want Y=%d)",
			iface.C1Trace, wantC1Trace, iface.C1X, iface.C1A, iface.C1Y, 100+n8)
	}

	wantC1DTrace := []string{"V", "X", "Y"}
	if !slices.Equal(iface.C1DTrace, wantC1DTrace) || iface.C1DX != n8 || iface.C1DY != 100+n8 {
		t.Errorf("C1D init order mismatch: trace=%v (want %v) X=%d Y=%d (want Y=%d)",
			iface.C1DTrace, wantC1DTrace, iface.C1DX, iface.C1DY, 100+n8)
	}
}

// Compile-time regression test: multiple blank package-level SIMD type and
// function declarations must not be suffixed or collide during specialization.
type _ = simd.Int8s
type _ = simd.Int8s

func _(simd.Int8s) {}
func _(simd.Int8s) {}
