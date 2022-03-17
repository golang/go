// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Fooer interface {
	Foo()
}

type Fooable[F /* ERROR instantiation cycle */ Fooer] struct {
	ptr F
}

func (f *Fooable[F]) Adapter() *Fooable[*FooerImpl[F]] {
	return &Fooable[*FooerImpl[F]]{&FooerImpl[F]{}}
}

type FooerImpl[F Fooer] struct {
}

func (fi *FooerImpl[F]) Foo() {}
