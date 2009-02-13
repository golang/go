// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import "vector"

type IntVector struct {
	// TODO do not export field
	vector.Vector;
}


func (p *IntVector) Init(len int) *IntVector {
	p.Vector.Init(len);
	return p;
}


func NewIntVector(len int) *IntVector {
	return new(IntVector).Init(len)
}


func (p *IntVector) At(i int) int {
	return p.Vector.At(i).(int)
}


func (p *IntVector) Set(i int, x int) {
	p.Vector.Set(i, x)
}


func (p *IntVector) Last() int {
	return p.Vector.Last().(int)
}


func (p *IntVector) Insert(i int, x int) {
	p.Vector.Insert(i, x)
}


func (p *IntVector) Delete(i int) int {
	return p.Vector.Delete(i).(int)
}


func (p *IntVector) Push(x int) {
	p.Vector.Push(x)
}


func (p *IntVector) Pop() int {
	return p.Vector.Pop().(int)
}


// SortInterface support
func (p *IntVector) Less(i, j int) bool {
	return p.At(i) < p.At(j)
}
