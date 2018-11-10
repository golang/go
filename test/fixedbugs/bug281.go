// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// https://golang.org/issue/807

package main

type Point struct {
	X, Y int64
}

type Rect struct {
	Min, Max Point
}

func (p Point) Sub(q Point) Point {
	return Point{p.X-q.X, p.Y-q.Y}
}

type Obj struct {
	bbox Rect
}

func (o *Obj) Bbox() Rect {
	return o.bbox
}

func (o *Obj) Points() [2]Point{
	return [2]Point{o.bbox.Min, o.bbox.Max}
}

var x = 0

func main() {
	o := &Obj{Rect{Point{800, 0}, Point{}}}
	p := Point{800, 300}
	q := p.Sub(o.Bbox().Min)
	if q.X != 0 || q.Y != 300 {
		println("BUG dot: ", q.X, q.Y)
		return
	}
	
	q = p.Sub(o.Points()[0])
	if q.X != 0 || q.Y != 300 {
		println("BUG index const: ", q.X, q.Y)
	}
	
	q = p.Sub(o.Points()[x])
	if q.X != 0 || q.Y != 300 {
		println("BUG index var: ", q.X, q.Y)
	}
}
