// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T struct{ p *int64 }

type i struct{}

func G() *T { return &T{nil} }

func (j i) F(a, b *T) *T {
	n := *a.p + *b.p
	return &T{&n}
}

func (j i) G() *T {
	return &T{}
}

type I[Idx any] interface {
	G() Idx
	F(a, b Idx) Idx
}

func Gen() I[*T] {
	return i{}
}
