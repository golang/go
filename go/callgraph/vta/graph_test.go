// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vta

import (
	"fmt"
	"go/types"
	"reflect"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/ssa/ssautil"
)

func TestNodeInterface(t *testing.T) {
	// Since ssa package does not allow explicit creation of ssa
	// values, we use the values from the program testdata/simple.go:
	//   - basic type int
	//   - struct X with two int fields a and b
	//   - global variable "gl"
	//   - "main" function and its
	//   - first register instruction t0 := *gl
	prog, _, err := testProg("testdata/simple.go")
	if err != nil {
		t.Fatalf("couldn't load testdata/simple.go program: %v", err)
	}

	pkg := prog.AllPackages()[0]
	main := pkg.Func("main")
	reg := firstRegInstr(main) // t0 := *gl
	X := pkg.Type("X").Type()
	gl := pkg.Var("gl")
	glPtrType, ok := gl.Type().(*types.Pointer)
	if !ok {
		t.Fatalf("could not cast gl variable to pointer type")
	}
	bint := glPtrType.Elem()

	pint := types.NewPointer(bint)
	i := types.NewInterface(nil, nil)

	voidFunc := main.Signature.Underlying()

	for _, test := range []struct {
		n node
		s string
		t types.Type
	}{
		{constant{typ: bint}, "Constant(int)", bint},
		{pointer{typ: pint}, "Pointer(*int)", pint},
		{mapKey{typ: bint}, "MapKey(int)", bint},
		{mapValue{typ: pint}, "MapValue(*int)", pint},
		{sliceElem{typ: bint}, "Slice([]int)", bint},
		{channelElem{typ: pint}, "Channel(chan *int)", pint},
		{field{StructType: X, index: 0}, "Field(testdata.X:a)", bint},
		{field{StructType: X, index: 1}, "Field(testdata.X:b)", bint},
		{global{val: gl}, "Global(gl)", gl.Type()},
		{local{val: reg}, "Local(t0)", bint},
		{indexedLocal{val: reg, typ: X, index: 0}, "Local(t0[0])", X},
		{function{f: main}, "Function(main)", voidFunc},
		{nestedPtrInterface{typ: i}, "PtrInterface(interface{})", i},
		{nestedPtrFunction{typ: voidFunc}, "PtrFunction(func())", voidFunc},
		{panicArg{}, "Panic", nil},
		{recoverReturn{}, "Recover", nil},
	} {
		if test.s != test.n.String() {
			t.Errorf("want %s; got %s", test.s, test.n.String())
		}
		if test.t != test.n.Type() {
			t.Errorf("want %s; got %s", test.t, test.n.Type())
		}
	}
}

func TestVtaGraph(t *testing.T) {
	// Get the basic type int from a real program.
	prog, _, err := testProg("testdata/simple.go")
	if err != nil {
		t.Fatalf("couldn't load testdata/simple.go program: %v", err)
	}

	glPtrType, ok := prog.AllPackages()[0].Var("gl").Type().(*types.Pointer)
	if !ok {
		t.Fatalf("could not cast gl variable to pointer type")
	}
	bint := glPtrType.Elem()

	n1 := constant{typ: bint}
	n2 := pointer{typ: types.NewPointer(bint)}
	n3 := mapKey{typ: types.NewMap(bint, bint)}
	n4 := mapValue{typ: types.NewMap(bint, bint)}

	// Create graph
	//   n1   n2
	//    \  / /
	//     n3 /
	//     | /
	//     n4
	g := make(vtaGraph)
	g.addEdge(n1, n3)
	g.addEdge(n2, n3)
	g.addEdge(n3, n4)
	g.addEdge(n2, n4)
	// for checking duplicates
	g.addEdge(n1, n3)

	want := vtaGraph{
		n1: map[node]bool{n3: true},
		n2: map[node]bool{n3: true, n4: true},
		n3: map[node]bool{n4: true},
	}

	if !reflect.DeepEqual(want, g) {
		t.Errorf("want %v; got %v", want, g)
	}

	for _, test := range []struct {
		n node
		l int
	}{
		{n1, 1},
		{n2, 2},
		{n3, 1},
		{n4, 0},
	} {
		if sl := len(g.successors(test.n)); sl != test.l {
			t.Errorf("want %d successors; got %d", test.l, sl)
		}
	}
}

// vtaGraphStr stringifies vtaGraph into a list of strings
// where each string represents an edge set of the format
// node -> succ_1, ..., succ_n. succ_1, ..., succ_n are
// sorted in alphabetical order.
func vtaGraphStr(g vtaGraph) []string {
	var vgs []string
	for n, succ := range g {
		var succStr []string
		for s := range succ {
			succStr = append(succStr, s.String())
		}
		sort.Strings(succStr)
		entry := fmt.Sprintf("%v -> %v", n.String(), strings.Join(succStr, ", "))
		vgs = append(vgs, entry)
	}
	return vgs
}

// subGraph checks if a graph `g1` is a subgraph of graph `g2`.
// Assumes that each element in `g1` and `g2` is an edge set
// for a particular node in a fixed yet arbitrary format.
func subGraph(g1, g2 []string) bool {
	m := make(map[string]bool)
	for _, s := range g2 {
		m[s] = true
	}

	for _, s := range g1 {
		if _, ok := m[s]; !ok {
			return false
		}
	}
	return true
}

func TestVTAGraphConstruction(t *testing.T) {
	for _, file := range []string{
		"testdata/store.go",
		"testdata/phi.go",
		"testdata/type_conversions.go",
		"testdata/type_assertions.go",
		"testdata/fields.go",
		"testdata/node_uniqueness.go",
		"testdata/store_load_alias.go",
		"testdata/phi_alias.go",
		"testdata/channels.go",
		"testdata/select.go",
		"testdata/stores_arrays.go",
		"testdata/maps.go",
		"testdata/ranges.go",
		"testdata/closures.go",
		"testdata/function_alias.go",
		"testdata/static_calls.go",
		"testdata/dynamic_calls.go",
		"testdata/returns.go",
		"testdata/panic.go",
	} {
		t.Run(file, func(t *testing.T) {
			prog, want, err := testProg(file)
			if err != nil {
				t.Fatalf("couldn't load test file '%s': %s", file, err)
			}
			if len(want) == 0 {
				t.Fatalf("couldn't find want in `%s`", file)
			}

			g, _ := typePropGraph(ssautil.AllFunctions(prog), cha.CallGraph(prog))
			if gs := vtaGraphStr(g); !subGraph(want, gs) {
				t.Errorf("`%s`: want superset of %v;\n got %v", file, want, gs)
			}
		})
	}
}
