// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is used to generate an object file which
// serves as test file for gcimporter_test.go.

package exports

import "go/ast"

// Issue 3682: Correctly read dotted identifiers from export data.
const init1 = 0

func init() {}

const (
	C0  int     = 0
	C1          = 3.14159265
	C2          = 2.718281828i
	C3          = -123.456e-789
	C4          = +123.456e+789
	C5          = 1234i
	C6          = "foo\n"
	C7          = `bar\n`
	C8          = 42
	C9  int     = 42
	C10 float64 = 42
)

type (
	T1  int
	T2  [10]int
	T3  []int
	T4  *int
	T5  chan int
	T6a chan<- int
	T6b chan (<-chan int)
	T6c chan<- (chan int)
	T7  <-chan *ast.File
	T8  struct{}
	T9  struct {
		a    int
		b, c float32
		d    []string `go:"tag"`
	}
	T10 struct {
		T8
		T9
		_ *T10
	}
	T11 map[int]string
	T12 any
	T13 interface {
		m1()
		m2(int) float32
	}
	T14 interface {
		T12
		T13
		m3(x ...struct{}) []T9
	}
	T15 func()
	T16 func(int)
	T17 func(x int)
	T18 func() float32
	T19 func() (x float32)
	T20 func(...any)
	T21 struct{ next *T21 }
	T22 struct{ link *T23 }
	T23 struct{ link *T22 }
	T24 *T24
	T25 *T26
	T26 *T27
	T27 *T25
	T28 func(T28) T28
)

var (
	V0 int
	V1         = -991.0
	V2 float32 = 1.2
)

func F1()         {}
func F2(x int)    {}
func F3() int     { return 0 }
func F4() float32 { return 0 }
func F5(a, b, c int, u, v, w struct{ x, y T1 }, more ...any) (p, q, r chan<- T10)

func (p *T1) M1()
