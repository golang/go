// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test method expressions with arguments.

package method4a

type T1 int

type T2 struct {
	F int
}

type I1 interface {
	Sum([]int, int) int
}

type I2 interface {
	Sum(a []int, b int) int
}

func (i T1) Sum(a []int, b int) int {
	r := int(i) + b
	for _, v := range a {
		r += v
	}
	return r
}

func (p *T2) Sum(a []int, b int) int {
	r := p.F + b
	for _, v := range a {
		r += v
	}
	return r
}
