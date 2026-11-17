// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"iter"
	"slices"
	"strings"
	"testing"
)

func TestSCCs(t *testing.T) {
	// Graph with:
	// - mutual cycle {A, B, C} reaching leaf D
	// - self-loop on E reached from D
	// - diamond from root R -> {A, F}, F -> D
	// - separate cycle {G, H} reached from B
	graph := map[string][]string{
		"R": {"A", "F"},
		"A": {"B"},
		"B": {"C", "G"},
		"C": {"A", "D"},
		"F": {"D"},
		"D": {"E"},
		"E": {"E"},
		"G": {"H"},
		"H": {"G"},
	}

	edges := func(v string) iter.Seq[string] {
		return slices.Values(graph[v])
	}

	var got [][]string
	seen := make(map[string]int) // node -> component index
	for comp := range sccs("R", edges) {
		c := slices.Clone(comp)
		slices.Sort(c)
		compIdx := len(got)
		for _, node := range c {
			if prev, dup := seen[node]; dup {
				t.Fatalf("node %q yielded in both SCC %d and %d", node, prev, compIdx)
			}
			seen[node] = compIdx
		}
		got = append(got, c)
	}

	// Check reverse topological order: for every edge u -> v,
	// v's SCC must be yielded at or before u's SCC.
	for u, vs := range graph {
		uComp, ok := seen[u]
		if !ok {
			t.Errorf("node %q not visited", u)
			continue
		}
		for _, v := range vs {
			vComp, ok := seen[v]
			if !ok {
				t.Errorf("target %q of %q not visited", v, u)
				continue
			}
			if vComp > uComp {
				t.Errorf("topological violation: edge %q (SCC %d) -> %q (SCC %d)", u, uComp, v, vComp)
			}
		}
	}

	wantSets := [][]string{
		{"A", "B", "C"},
		{"D"},
		{"E"},
		{"F"},
		{"G", "H"},
		{"R"},
	}
	if len(got) != len(wantSets) {
		t.Fatalf("got %d SCCs %v, want %d %v", len(got), got, len(wantSets), wantSets)
	}
	gotSorted := slices.Clone(got)
	slices.SortFunc(gotSorted, slices.Compare)
	for i := range wantSets {
		if !slices.Equal(gotSorted[i], wantSets[i]) {
			t.Errorf("SCC set %d = %v, want %v", i, gotSorted[i], wantSets[i])
		}
	}

	// Early termination: stop after first yielded SCC without panicking.
	count := 0
	for range sccs("R", edges) {
		count++
		break
	}
	if count != 1 {
		t.Errorf("early break yielded %d SCCs, want 1", count)
	}
}

type mapImporter map[string]*types2.Package

func (m mapImporter) Import(path string) (*types2.Package, error) {
	return m[path], nil
}

func checkPackage(t *testing.T, path, src string, imports mapImporter) (*types2.Package, *types2.Info, []*syntax.File) {
	t.Helper()
	f, err := syntax.Parse(syntax.NewFileBase(path+".go"), strings.NewReader(src), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	info := &types2.Info{
		Defs: make(map[*syntax.Name]types2.Object),
		Uses: make(map[*syntax.Name]types2.Object),
	}
	conf := types2.Config{Importer: imports}
	pkg, err := conf.Check(path, []*syntax.File{f}, info)
	if err != nil {
		t.Fatalf("typecheck %s: %v", path, err)
	}
	return pkg, info, []*syntax.File{f}
}

func TestAnalyzer(t *testing.T) {
	simdSrc := `package simd
type _simd struct{}
type Int8s struct {
	_ _simd
	v [16]int8
}
`
	simdPkg, _, _ := checkPackage(t, "simd", simdSrc, nil)

	depSrc := `package dep
import "simd"

// Struct whose SIMD dependence appears only in its method set.
type ImportedMethodOnly struct{ X int }
func (ImportedMethodOnly) Take(simd.Int8s) {}
`
	depPkg, _, _ := checkPackage(t, "dep", depSrc, mapImporter{"simd": simdPkg})

	userSrc := `package user
import (
	"dep"
	"simd"
)

// Uncached base SIMD type reached through a cycle must memoize every cycle member.
type CycA struct { B *CycB }
type CycB struct { A *CycA; V simd.Int8s }

// Plain non-SIMD cycle.
type PlainA struct { B *PlainB }
type PlainB struct { A *PlainA }

// Defined types wrapping a base SIMD type directly and transitively.
type MySimd simd.Int8s
type MySimd2 MySimd

// Alias-receiver method only (no direct method on base).
type AliasBase struct{ X int }
type AliasRecv = AliasBase
func (*AliasRecv) Use(simd.Int8s) {}

// Imported type whose SIMD dependence appears only in its method set.
var UseImported dep.ImportedMethodOnly

// Embedded interface and union constraint.
type BaseUnion interface {
	simd.Int8s | int
}
type EmbedUnion interface {
	BaseUnion
	M()
}
type BoxEmbedUnion[T EmbedUnion] struct{ V T }

// Blank declarations (types, vars, functions, methods) must not be marked dependent.
type _ = simd.Int8s
type _ = simd.Int8s
type _ simd.Int8s
var _ simd.Int8s
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
func (PlainA) _(simd.Int8s) {}
func (PlainA) _() {
	var v simd.Int8s
	_ = v
}

// Zero-weight instantiation cycle with 7! permutations (7 distinct type args).
type Perm[A, B, C, D, E, F, G any] struct {
	Swap *Perm[B, A, C, D, E, F, G]
	Rot  *Perm[B, C, D, E, F, G, A]
}
type PermAlias[A, B, C, D, E, F, G any] = Perm[A, B, C, D, E, F, G]
var PermSimd Perm[simd.Int8s, int8, int16, int32, int64, uint8, uint16]
var PermPlain Perm[int, int8, int16, int32, int64, uint8, uint16]
var PermAliasSimd PermAlias[simd.Int8s, int8, int16, int32, int64, uint8, uint16]
var PermAliasPlain PermAlias[int, int8, int16, int32, int64, uint8, uint16]

// Function with empty params and results (nil *types2.Tuple).
func EmptySig() {}
func DepSig(simd.Int8s) {}

// Local type inside function must not be marked in isDependentObj.
func WithLocal() {
	type localVec simd.Int8s
	var v localVec
	_ = v
}
`
	pkg, info, files := checkPackage(t, "user", userSrc, mapImporter{"simd": simdPkg, "dep": depPkg})
	a := NewAnalyzer(pkg, info)

	// Explicitly test CycA first, when simd.Int8s is still uncached in a.visited.
	cycA := pkg.Scope().Lookup("CycA").Type()
	cycB := pkg.Scope().Lookup("CycB").Type()
	if !a.isDependentType(cycA) {
		t.Fatal("isDependentType(CycA) = false, want true")
	}
	for _, typ := range []types2.Type{cycA, cycB} {
		if dep, ok := a.visited[typ]; !ok || !dep {
			t.Errorf("a.visited[%v] = (%v, %v), want (true, true)", typ, dep, ok)
		}
	}

	if a.HasDependentSignature(nil) {
		t.Errorf("HasDependentSignature(nil) = true, want false")
	}

	if !a.Analyze(files) {
		t.Fatal("Analyze returned false, want true")
	}

	if a.HasDependentSignature(pkg.Scope().Lookup("EmptySig").(*types2.Func).Signature()) {
		t.Errorf("HasDependentSignature(EmptySig) = true, want false")
	}
	if !a.HasDependentSignature(pkg.Scope().Lookup("DepSig").(*types2.Func).Signature()) {
		t.Errorf("HasDependentSignature(DepSig) = false, want true")
	}

	wantDep := map[string]bool{
		"CycA":           true,
		"CycB":           true,
		"PlainA":         false,
		"PlainB":         false,
		"MySimd":         true,
		"MySimd2":        true,
		"AliasBase":      true,
		"AliasRecv":      true,
		"UseImported":    true,
		"BaseUnion":      true,
		"EmbedUnion":     true,
		"BoxEmbedUnion":  true,
		"Perm":           false,
		"PermAlias":      false,
		"PermSimd":       true,
		"PermPlain":      false,
		"PermAliasSimd":  true,
		"PermAliasPlain": false,
		"EmptySig":       false,
		"DepSig":         true,
		"WithLocal":      true,
	}
	for name, want := range wantDep {
		obj := pkg.Scope().Lookup(name)
		if obj == nil {
			t.Fatalf("missing object %q", name)
		}
		if got := a.isDependentObj[obj]; got != want {
			t.Errorf("isDependentObj[%s] = %v, want %v", name, got, want)
		}
	}
	if importedObj := depPkg.Scope().Lookup("ImportedMethodOnly"); !a.isDependentObj[importedObj] {
		t.Errorf("isDependentObj[dep.ImportedMethodOnly] = false, want true")
	}

	// Ensure Perm/PermAlias did not expand the 7! = 5040 instantiation orbit.
	if n := len(a.visited); n >= 200 {
		t.Errorf("len(a.visited) = %d, want < 200 (instantiated generic types may have been expanded)", n)
	}

	// Ensure no blank declaration or function-local type was marked dependent.
	for obj := range a.isDependentObj {
		switch obj := obj.(type) {
		case *types2.TypeName, *types2.Var, *types2.Func:
			if obj.Name() == "_" {
				t.Errorf("blank %T marked in isDependentObj", obj)
			}
			if obj.Name() == "localVec" {
				t.Errorf("function-local type %q marked in isDependentObj", obj.Name())
			}
		}
	}

	// Ensure no typed-nil *types2.Tuple was inserted into a.visited.
	for typ := range a.visited {
		if tup, ok := typ.(*types2.Tuple); ok && tup == nil {
			t.Errorf("typed-nil *types2.Tuple found in a.visited")
		}
	}
}
