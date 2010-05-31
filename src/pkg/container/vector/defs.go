// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The vector package implements containers for managing sequences
// of elements. Vectors grow and shrink dynamically as necessary.
package vector


// Vector is a container for numbered sequences of elements of type interface{}.
// A vector's length and capacity adjusts automatically as necessary.
// The zero value for Vector is an empty vector ready to use.
type Vector []interface{}


// IntVector is a container for numbered sequences of elements of type int.
// A vector's length and capacity adjusts automatically as necessary.
// The zero value for IntVector is an empty vector ready to use.
type IntVector []int


// StringVector is a container for numbered sequences of elements of type string.
// A vector's length and capacity adjusts automatically as necessary.
// The zero value for StringVector is an empty vector ready to use.
type StringVector []string


// Initial underlying array size
const initialSize = 8


// Partial sort.Interface support

// LessInterface provides partial support of the sort.Interface.
type LessInterface interface {
	Less(y interface{}) bool
}


// Less returns a boolean denoting whether the i'th element is less than the j'th element.
func (p *Vector) Less(i, j int) bool { return (*p)[i].(LessInterface).Less((*p)[j]) }


// sort.Interface support

// Less returns a boolean denoting whether the i'th element is less than the j'th element.
func (p *IntVector) Less(i, j int) bool { return (*p)[i] < (*p)[j] }


// Less returns a boolean denoting whether the i'th element is less than the j'th element.
func (p *StringVector) Less(i, j int) bool { return (*p)[i] < (*p)[j] }
