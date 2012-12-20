// compile

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4529: escape analysis crashes on "go f(g())"
// when g has multiple returns.

package main

type M interface{}

type A struct {
	a string
	b chan M
}

func (a *A) I() (b <-chan M, c chan<- M) {
	a.b, c = make(chan M), make(chan M)
	b = a.b

	return
}

func Init(a string, b *A, c interface {
	I() (<-chan M, chan<- M)
}) {
	b.a = a
	go b.c(c.I())
}

func (a *A) c(b <-chan M, _ chan<- M) {}
