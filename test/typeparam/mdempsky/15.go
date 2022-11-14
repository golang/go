// run -goexperiment fieldtrack

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that generics, promoted methods, and //go:nointerface
// interoperate as expected.

package main

import (
	"reflect"
)

func TypeString[T any]() string {
	return reflect.TypeOf(new(T)).Elem().String()
}

func Test[T, Bad, Good any]() {
	switch interface{}(new(T)).(type) {
	case Bad:
		println("FAIL:", TypeString[T](), "matched", TypeString[Bad]())
	case Good:
		// ok
	default:
		println("FAIL:", TypeString[T](), "did not match", TypeString[Good]())
	}
}

func TestE[T any]() { Test[T, interface{ EBad() }, interface{ EGood() }]() }
func TestX[T any]() { Test[T, interface{ XBad() }, interface{ XGood() }]() }

type E struct{}

//go:nointerface
func (E) EBad()  {}
func (E) EGood() {}

type X[T any] struct{ E }

//go:nointerface
func (X[T]) XBad()  {}
func (X[T]) XGood() {}

type W struct{ X[int] }

func main() {
	_ = E.EGood
	_ = E.EBad

	TestE[E]()

	_ = X[int].EGood
	_ = X[int].EBad
	_ = X[int].XGood
	_ = X[int].XBad

	TestE[X[int]]()
	TestX[X[int]]()

	_ = W.EGood
	_ = W.EBad
	_ = W.XGood
	_ = W.XBad

	TestE[W]()
	TestX[W]()
}
