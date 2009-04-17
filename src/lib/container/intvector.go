// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import "container/vector"

// IntVector is a specialization of Vector that hides the wrapping of Elements around ints.
type IntVector struct {
	// TODO do not export field
	vector.Vector;
}


// Init initializes a new or resized vector.  The initial length may be <= 0 to
// request a default length.  If initial_len is shorter than the current
// length of the IntVector, trailing elements of the IntVector will be cleared.
func (p *IntVector) Init(len int) *IntVector {
	p.Vector.Init(len);
	return p;
}


// NewIntVector returns an initialized new IntVector with length at least len.
func NewIntVector(len int) *IntVector {
	return new(IntVector).Init(len)
}


// At returns the i'th element of the vector.
func (p *IntVector) At(i int) int {
	return p.Vector.At(i).(int)
}


// Set sets the i'th element of the vector to value x.
func (p *IntVector) Set(i int, x int) {
	p.Vector.Set(i, x)
}


// Last returns the element in the vector of highest index.
func (p *IntVector) Last() int {
	return p.Vector.Last().(int)
}


// Insert inserts into the vector an element of value x before
// the current element at index i.
func (p *IntVector) Insert(i int, x int) {
	p.Vector.Insert(i, x)
}


// Delete deletes the i'th element of the vector.  The gap is closed so the old
// element at index i+1 has index i afterwards.
func (p *IntVector) Delete(i int) int {
	return p.Vector.Delete(i).(int)
}


// Push appends x to the end of the vector.
func (p *IntVector) Push(x int) {
	p.Vector.Push(x)
}


// Pop deletes and returns the last element of the vector.
func (p *IntVector) Pop() int {
	return p.Vector.Pop().(int)
}


// SortInterface support
// Less returns a boolean denoting whether the i'th element is less than the j'th element.
func (p *IntVector) Less(i, j int) bool {
	return p.At(i) < p.At(j)
}
