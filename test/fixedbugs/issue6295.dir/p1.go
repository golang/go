// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p1

import "./p0"

type T1 interface {
	p0.T0
	m1()
}

type S1 struct {
	p0.S0
}

func (S1) m1() {}

func NewT0() p0.T0 {
	return S1{}
}

func NewT1() T1 {
	return S1{}
}
