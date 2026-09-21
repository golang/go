// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type E struct{}

var Called bool

func (*E) NilCheck() {
	Called = true
}

func (E) ValueNilCheck() {
	Called = true
}

type T0 struct {
	E
}

type T1 struct {
	X int
	E
}

type ValueT0 struct {
	E
}

type ValueT1 struct {
	X int
	E
}
