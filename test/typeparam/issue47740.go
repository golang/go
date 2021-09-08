// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type Exp[Ty any] interface {
	Eval() Ty
}

type Lit[Ty any] Ty

func (lit Lit[Ty]) Eval() Ty       { return Ty(lit) }
func (lit Lit[Ty]) String() string { return fmt.Sprintf("(lit %v)", Ty(lit)) }

type Eq[Ty any] struct {
	a Exp[Ty]
	b Exp[Ty]
}

func (e Eq[Ty]) String() string {
	return fmt.Sprintf("(eq %v %v)", e.a, e.b)
}

var (
	e0 = Eq[int]{Lit[int](128), Lit[int](64)}
	e1 = Eq[bool]{Lit[bool](true), Lit[bool](true)}
)

func main() {
	fmt.Printf("%v\n", e0)
	fmt.Printf("%v\n", e1)
}
