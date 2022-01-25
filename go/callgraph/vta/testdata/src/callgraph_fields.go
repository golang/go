// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

type A struct {
	I
}

func (a *A) Do() {
	a.Foo()
}

type B struct{}

func (b B) Foo() {}

func NewA(b B) *A {
	return &A{I: &b}
}

func Baz(b B) {
	a := NewA(b)
	a.Do()
}

// Relevant SSA:
// func Baz(b B):
//        t0 = local B (b)
//        *t0 = b
//        t1 = *t0
//        t2 = NewA(t1)
//        t3 = (*A).Do(t2)
//        return
//
// func (a *A) Do():
//        t0 = &a.I [#0]
//        t1 = *t0
//        t2 = invoke t1.Foo()
//        return
//
// Name: (testdata.A).Foo
// Synthetic: wrapper for func (testdata.I).Foo()
// Location: testdata/callgraph_fields.go:10:2
// func (arg0 testdata.A) Foo():
//	  t0 = local testdata.A ()
//        *t0 = arg0
//        t1 = &t0.I [#0]
//        t2 = *t1
//        t3 = invoke t2.Foo()
//        return
//
// Name: (*testdata.A).Foo
// Synthetic: wrapper for func (testdata.I).Foo()
// Location: testdata/callgraph_fields.go:10:2
// func (arg0 *testdata.A) Foo():
//        t0 = &arg0.I [#0]
//        t1 = *t0
//        t2 = invoke t1.Foo()
//        return
//
// func (b B) Foo():
//        t0 = local B (b)
//        *t0 = b
//        return
//
// func (b *testdata.B) Foo():
//        t0 = ssa:wrapnilchk(b, "testdata.B":string, "Foo":string)
//        t1 = *t0
//        t2 = (testdata.B).Foo(t1)
//        return
//
// func NewA(b B) *A:
//        t0 = new B (b)
//        *t0 = b
//        t1 = new A (complit)
//        t2 = &t1.I [#0]
//        t3 = make I <- *B (t0)
//        *t2 = t3
//        return t1

// WANT:
// Baz: (*A).Do(t2) -> A.Do; NewA(t1) -> NewA
// A.Do: invoke t1.Foo() -> B.Foo
