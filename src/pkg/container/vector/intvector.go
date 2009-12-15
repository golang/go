// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector


// IntVector is a specialization of Vector that hides the wrapping of Elements around ints.
type IntVector struct {
	Vector
}


// Resize changes the length and capacity of a vector.
// If the new length is shorter than the current length, Resize discards
// trailing elements. If the new length is longer than the current length,
// Resize adds 0 elements. The capacity parameter is ignored unless the
// new length or capacity is longer that the current capacity.
func (p *IntVector) Resize(length, capacity int) *IntVector {
	i := p.Len()
	p.Vector.Resize(length, capacity)
	for a := p.a; i < len(a); i++ {
		a[i] = 0
	}
	return p
}


// At returns the i'th element of the vector.
func (p *IntVector) At(i int) int { return p.Vector.At(i).(int) }


// Set sets the i'th element of the vector to value x.
func (p *IntVector) Set(i int, x int) { p.a[i] = x }


// Last returns the element in the vector of highest index.
func (p *IntVector) Last() int { return p.Vector.Last().(int) }


// Data returns all the elements as a slice.
func (p *IntVector) Data() []int {
	arr := make([]int, p.Len())
	for i, v := range p.a {
		arr[i] = v.(int)
	}
	return arr
}


// Insert inserts into the vector an element of value x before
// the current element at index i.
func (p *IntVector) Insert(i int, x int) { p.Vector.Insert(i, x) }


// InsertVector inserts into the vector the contents of the Vector
// x such that the 0th element of x appears at index i after insertion.
func (p *IntVector) InsertVector(i int, x *IntVector) {
	p.Vector.InsertVector(i, &x.Vector)
}


// Slice returns a new IntVector by slicing the old one to extract slice [i:j].
// The elements are copied. The original vector is unchanged.
func (p *IntVector) Slice(i, j int) *IntVector {
	return &IntVector{*p.Vector.Slice(i, j)}
}


// Push appends x to the end of the vector.
func (p *IntVector) Push(x int) { p.Vector.Push(x) }


// Pop deletes and returns the last element of the vector.
func (p *IntVector) Pop() int { return p.Vector.Pop().(int) }


// AppendVector appends the entire IntVector x to the end of this vector.
func (p *IntVector) AppendVector(x *IntVector) {
	p.Vector.InsertVector(len(p.a), &x.Vector)
}


// sort.Interface support
// Less returns a boolean denoting whether the i'th element is less than the j'th element.
func (p *IntVector) Less(i, j int) bool { return p.At(i) < p.At(j) }


// Iterate over all elements; driver for range
func (p *IntVector) iterate(c chan<- int) {
	for _, v := range p.a {
		c <- v.(int)
	}
	close(c)
}


// Channel iterator for range.
func (p *IntVector) Iter() <-chan int {
	c := make(chan int)
	go p.iterate(c)
	return c
}
