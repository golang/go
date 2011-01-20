// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const a_const = 0

const (
	pi    = /* the usual */ 3.14159265358979323
	e     = 2.718281828
	mask1 int = 1 << iota
	mask2 = 1 << iota
	mask3 = 1 << iota
	mask4 = 1 << iota
)

type (
	Empty interface{}
	Point struct {
		x, y int
	}
	Point2 Point
)

func (p *Point) Initialize(x, y int) *Point {
	p.x, p.y = x, y
	return p
}

func (p *Point) Distance() int {
	return p.x*p.x + p.y*p.y
}

var (
	x1      int
	x2      int
	u, v, w float32
)

func foo() {}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func swap(x, y int) (u, v int) {
	u = y
	v = x
	return
}

func control_structs() {
	var p *Point = new(Point).Initialize(2, 3)
	i := p.Distance()
	var f float32 = 0.3
	_ = f
	for {
	}
	for {
	}
	for j := 0; j < i; j++ {
		if i == 0 {
		} else {
			i = 0
		}
		var x float32
		_ = x
	}
foo: // a label
	var j int
	switch y := 0; true {
	case i < y:
		fallthrough
	case i < j:
	case i == 0, i == 1, i == j:
		i++
		i++
		goto foo
	default:
		i = -+-+i
		break
	}
}

func main() {
}
