// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"reflect"
	"testing"
)

type T1 int
type T2 int

func f[P T1 | T2, _ []P]() {}

var _ = f[T1]

// test case from issue

type BaseT interface {
	Type1 | Type2
}
type BaseType int
type Type1 BaseType
type Type2 BaseType // float64

type ValueT[T BaseT] struct {
	A1 T
}

func NewType1() *ValueT[Type1] {
	r := NewT[Type1]()
	return r
}
func NewType2() *ValueT[Type2] {
	r := NewT[Type2]()
	return r
}

func NewT[TBase BaseT, TVal ValueT[TBase]]() *TVal {
	ret := TVal{}
	return &ret
}
func TestGoType(t *testing.T) {
	r1 := NewType1()
	r2 := NewType2()
	t.Log(r1, r2)
	t.Log(reflect.TypeOf(r1), reflect.TypeOf(r2))
	fooT1(r1.A1)
	fooT2(r2.A1)
}

func fooT1(t1 Type1) {

}
func fooT2(t2 Type2) {

}
