// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Fooer[t any] interface {
	foo(Barer[t])
}
type Barer[t any] interface {
	bar(Bazer[t])
}
type Bazer[t any] interface {
	Fooer[t]
	baz(t)
}

type Int int

func (n Int) baz(int) {}
func (n Int) foo(b Barer[int]) { b.bar(n) }

type F[t any] interface { f(G[t]) }
type G[t any] interface { g(H[t]) }
type H[t any] interface { F[t] }

type T struct{}
func (n T) f(b G[T]) { b.g(n) }
