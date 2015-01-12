// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cgo rewrote C.var to *_Cvar_var, but left
// C.var.field as _Cvar.var.field.  It now rewrites
// the latter as (*_Cvar_var).field.
// See https://golang.org/issue/9557.

package cgotest

// struct issue9557_t {
//   int a;
// } test9557bar = { 42 };
//
// struct issue9557_t *issue9557foo = &test9557bar;
import "C"
import "testing"

func test9557(t *testing.T) {
	// implicitly dereference a Go variable
	foo := C.issue9557foo
	if v := foo.a; v != 42 {
		t.Fatalf("foo.a expected 42, but got %d", v)
	}

	// explicitly dereference a C variable
	if v := (*C.issue9557foo).a; v != 42 {
		t.Fatalf("(*C.issue9557foo).a expected 42, but is %d", v)
	}

	// implicitly dereference a C variable
	if v := C.issue9557foo.a; v != 42 {
		t.Fatalf("C.issue9557foo.a expected 42, but is %d", v)
	}
}
