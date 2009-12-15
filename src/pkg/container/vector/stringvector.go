// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

// StringVector is a specialization of Vector that hides the wrapping of Elements around strings.
type StringVector struct {
	Vector
}


// Resize changes the length and capacity of a vector.
// If the new length is shorter than the current length, Resize discards
// trailing elements. If the new length is longer than the current length,
// Resize adds "" elements. The capacity parameter is ignored unless the
// new length or capacity is longer that the current capacity.
func (p *StringVector) Resize(length, capacity int) *StringVector {
	i := p.Len()
	p.Vector.Resize(length, capacity)
	for a := p.a; i < len(a); i++ {
		a[i] = ""
	}
	return p
}


// At returns the i'th element of the vector.
func (p *StringVector) At(i int) string { return p.Vector.At(i).(string) }


// Set sets the i'th element of the vector to value x.
func (p *StringVector) Set(i int, x string) { p.a[i] = x }


// Last returns the element in the vector of highest index.
func (p *StringVector) Last() string { return p.Vector.Last().(string) }


// Data returns all the elements as a slice.
func (p *StringVector) Data() []string {
	arr := make([]string, p.Len())
	for i, v := range p.a {
		arr[i] = v.(string)
	}
	return arr
}


// Insert inserts into the vector an element of value x before
// the current element at index i.
func (p *StringVector) Insert(i int, x string) {
	p.Vector.Insert(i, x)
}


// InsertVector inserts into the vector the contents of the Vector
// x such that the 0th element of x appears at index i after insertion.
func (p *StringVector) InsertVector(i int, x *StringVector) {
	p.Vector.InsertVector(i, &x.Vector)
}


// Slice returns a new StringVector by slicing the old one to extract slice [i:j].
// The elements are copied. The original vector is unchanged.
func (p *StringVector) Slice(i, j int) *StringVector {
	return &StringVector{*p.Vector.Slice(i, j)}
}


// Push appends x to the end of the vector.
func (p *StringVector) Push(x string) { p.Vector.Push(x) }


// Pop deletes and returns the last element of the vector.
func (p *StringVector) Pop() string { return p.Vector.Pop().(string) }


// AppendVector appends the entire StringVector x to the end of this vector.
func (p *StringVector) AppendVector(x *StringVector) {
	p.Vector.InsertVector(len(p.a), &x.Vector)
}


// sort.Interface support
// Less returns a boolean denoting whether the i'th element is less than the j'th element.
func (p *StringVector) Less(i, j int) bool { return p.At(i) < p.At(j) }


// Iterate over all elements; driver for range
func (p *StringVector) iterate(c chan<- string) {
	for _, v := range p.a {
		c <- v.(string)
	}
	close(c)
}


// Channel iterator for range.
func (p *StringVector) Iter() <-chan string {
	c := make(chan string)
	go p.iterate(c)
	return c
}
