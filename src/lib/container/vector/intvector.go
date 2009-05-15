// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import "container/vector"

// IntVector is a specialization of Vector that hides the wrapping of Elements around ints.
type IntVector struct {
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


// Last returns the element in the vector of highest index.
func (p *IntVector) Last() int {
	return p.Vector.Last().(int)
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
