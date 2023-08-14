// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DO NOT EDIT (use 'go test -v -update-expected' instead.)
// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt
// for more information on the format of this file.
// <endfilepreamble>

package stub

// stub.go T_stub 16 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
func T_stub() {
}

func ThisFunctionShouldBeIgnored(x int) {
	println(x)
}

// stub.go init.0 27 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
func init() {
	ThisFunctionShouldBeIgnored(1)
}

// stub.go T_contains_closures 43 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
// stub.go T_contains_closures.func1 44 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
// stub.go T_contains_closures.func2 46 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
func T_contains_closures(q int) func() {
	f := func() { M["a"] = 9 }
	f()
	f2 := func() { M["a"] = 4 }
	if M["b"] != 9 {
		return f
	}
	return f2
}

// stub.go T_Unique[go.shape.int] 69 0 4
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
// stub.go T_Unique[go.shape.string] 69 1 4
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
// stub.go T_Unique[int] 69 2 4
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
// stub.go T_Unique[string] 69 3 4
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
func T_Unique[T comparable](set []T) []T {
	nset := make([]T, 0, 8)
loop:
	for _, s := range set {
		for _, e := range nset {
			if s == e {
				continue loop
			}
		}
		nset = append(nset, s)
	}

	return nset
}

// stub.go T_uniq_int_count 88 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
func T_uniq_int_count(s []int) int {
	return len(T_Unique[int](s))
}

// stub.go T_uniq_string_count 96 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
func T_uniq_string_count(s []string) int {
	return len(T_Unique[string](s))
}

// stub.go T_epilog 104 0 1
// <endpropsdump>
// {"Flags":0,"ParamFlags":null,"ResultFlags":null}
// <endfuncpreamble>
func T_epilog() {
}

var M = map[string]int{}
