// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// *** DEPRECATED PACKAGE - USE package vector INSTEAD ***
//

package array

import "array"

type IntArray struct {
	// TODO do not export field
	array.Array;
}


func (p *IntArray) Init(len int) *IntArray {
	p.Array.Init(len);
	return p;
}


func NewIntArray(len int) *IntArray {
	return new(IntArray).Init(len)
}


func (p *IntArray) At(i int) int {
	return p.Array.At(i).(int)
}


func (p *IntArray) Set(i int, x int) {
	p.Array.Set(i, x)
}


func (p *IntArray) Last() int {
	return p.Array.Last().(int)
}


func (p *IntArray) Insert(i int, x int) {
	p.Array.Insert(i, x)
}


func (p *IntArray) Delete(i int) int {
	return p.Array.Delete(i).(int)
}


func (p *IntArray) Push(x int) {
	p.Array.Push(x)
}


func (p *IntArray) Pop() int {
	return p.Array.Pop().(int)
}


// SortInterface support
func (p *IntArray) Less(i, j int) bool {
	return p.At(i) < p.At(j)
}
