// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Fooer interface {
	Foo() Barer
}

type Barer interface {
	Bar()
}

type impl struct{}

func (r *impl) Foo() Barer {
	return r
}

func (r *impl) Bar() {}

func f1() {
	var r Fooer = &impl{}
	r.Foo().Bar()
}
