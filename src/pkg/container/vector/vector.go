// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The vector package implements a container for managing sequences
// of elements. Vectors grow and shrink dynamically as necessary.
package vector

// Vector is the container itself.
// The zero value for Vector is an empty vector ready to use.
type Vector struct {
	a []interface{};
}


// Insert n elements at position i.
func expand(a []interface{}, i, n int) []interface{} {
	// make sure we have enough space
	len0 := len(a);
	len1 := len0 + n;
	if len1 <= cap(a) {
		// enough space - just expand
		a = a[0:len1]
	} else {
		// not enough space - double capacity
		capb := cap(a) * 2;
		if capb <= len1 {
			// still not enough - use required length
			capb = len1
		}
		// capb > len1
		b := make([]interface{}, len1, capb);
		copy(b, a);
		a = b;
	}

	// make a hole
	for j := len0 - 1; j >= i; j-- {
		a[j+n] = a[j]
	}
	return a;
}


// Resize changes the length and capacity of a vector.
// If the new length is shorter than the current length, Resize discards
// trailing elements. If the new length is longer than the current length,
// Resize adds nil elements. The capacity parameter is ignored unless the
// new length or capacity is longer that the current capacity.
func (p *Vector) Resize(length, capacity int) *Vector {
	a := p.a;

	if length > cap(a) || capacity > cap(a) {
		// not enough space or larger capacity requested explicitly
		b := make([]interface{}, length, capacity);
		copy(b, a);
		a = b;
	} else if length < len(a) {
		// clear trailing elements
		for i := range a[length:] {
			a[length+i] = nil
		}
	}

	p.a = a[0:length];
	return p;
}


// Len returns the number of elements in the vector.
func (p *Vector) Len() int	{ return len(p.a) }


// Cap returns the capacity of the vector; that is, the
// maximum length the vector can grow without resizing.
func (p *Vector) Cap() int	{ return cap(p.a) }


// At returns the i'th element of the vector.
func (p *Vector) At(i int) interface{}	{ return p.a[i] }


// Set sets the i'th element of the vector to value x.
func (p *Vector) Set(i int, x interface{})	{ p.a[i] = x }


// Last returns the element in the vector of highest index.
func (p *Vector) Last() interface{}	{ return p.a[len(p.a)-1] }


// Data returns all the elements as a slice.
func (p *Vector) Data() []interface{} {
	arr := make([]interface{}, p.Len());
	for i, v := range p.a {
		arr[i] = v
	}
	return arr;
}


// Insert inserts into the vector an element of value x before
// the current element at index i.
func (p *Vector) Insert(i int, x interface{}) {
	p.a = expand(p.a, i, 1);
	p.a[i] = x;
}


// Delete deletes the i'th element of the vector.  The gap is closed so the old
// element at index i+1 has index i afterwards.
func (p *Vector) Delete(i int) {
	a := p.a;
	n := len(a);

	copy(a[i:n-1], a[i+1:n]);
	a[n-1] = nil;	// support GC, nil out entry
	p.a = a[0 : n-1];
}


// InsertVector inserts into the vector the contents of the Vector
// x such that the 0th element of x appears at index i after insertion.
func (p *Vector) InsertVector(i int, x *Vector) {
	p.a = expand(p.a, i, len(x.a));
	copy(p.a[i:i+len(x.a)], x.a);
}


// Cut deletes elements i through j-1, inclusive.
func (p *Vector) Cut(i, j int) {
	a := p.a;
	n := len(a);
	m := n - (j - i);

	copy(a[i:m], a[j:n]);
	for k := m; k < n; k++ {
		a[k] = nil	// support GC, nil out entries
	}

	p.a = a[0:m];
}


// Slice returns a new Vector by slicing the old one to extract slice [i:j].
// The elements are copied. The original vector is unchanged.
func (p *Vector) Slice(i, j int) *Vector {
	s := new(Vector).Resize(j-i, 0);	// will fail in Init() if j < i
	copy(s.a, p.a[i:j]);
	return s;
}


// Do calls function f for each element of the vector, in order.
// The function should not change the indexing of the vector underfoot.
func (p *Vector) Do(f func(elem interface{})) {
	for i := 0; i < len(p.a); i++ {
		f(p.a[i])	// not too safe if f changes the Vector
	}
}


// Convenience wrappers

// Push appends x to the end of the vector.
func (p *Vector) Push(x interface{})	{ p.Insert(len(p.a), x) }


// Pop deletes the last element of the vector.
func (p *Vector) Pop() interface{} {
	i := len(p.a) - 1;
	x := p.a[i];
	p.a[i] = nil;	// support GC, nil out entry
	p.a = p.a[0:i];
	return x;
}


// AppendVector appends the entire Vector x to the end of this vector.
func (p *Vector) AppendVector(x *Vector)	{ p.InsertVector(len(p.a), x) }


// Partial sort.Interface support

// LessInterface provides partial support of the sort.Interface.
type LessInterface interface {
	Less(y interface{}) bool;
}


// Less returns a boolean denoting whether the i'th element is less than the j'th element.
func (p *Vector) Less(i, j int) bool	{ return p.a[i].(LessInterface).Less(p.a[j]) }


// Swap exchanges the elements at indexes i and j.
func (p *Vector) Swap(i, j int) {
	a := p.a;
	a[i], a[j] = a[j], a[i];
}


// Iterate over all elements; driver for range
func (p *Vector) iterate(c chan<- interface{}) {
	for _, v := range p.a {
		c <- v
	}
	close(c);
}


// Channel iterator for range.
func (p *Vector) Iter() <-chan interface{} {
	c := make(chan interface{});
	go p.iterate(c);
	return c;
}
