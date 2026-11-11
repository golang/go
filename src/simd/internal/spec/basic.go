// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//specgen:category Basics

package spec

import (
	"fmt"
	"strings"
)

// Len returns the number of elements in x.
func Len[E EltOrMask, W Width](x Vec[E, W]) int {
	return x.len()
}

// GetElem returns the i'th element x. If i >= {{lanes .x "x"}}, it panics.
//
//	z = x[i]
func GetElem[E Elt, W Width](x Vec[E, W], i uint8) (z E) {
	return x[i]
}

// SetElem returns x with its i'th element set to y. If i >= {{lanes .x "x"}}, it panics.
//
//	z = x
//	z[i] = y
func SetElem[E Elt, W Width](x Vec[E, W], i uint8, y E) (z Vec[E, W]) {
	z = makeVec[E, W]()
	copy(z, x)
	z[i] = y
	return z
}

// GetLo returns the lower half of x.
//
// {{if eq .xL 4}}
//
//	z = {x[0], x[1]}
//
// {{else}}
//
//	z = {x[0], x[1], ..., x[n/2-1]}, where n = {{lanes .x "x"}}
//
// {{end}}
//
// See also [Enlarge].
//
//specgen:require z={xB}{xN}x{xL/2}
func GetLo[E Elt, W SplittableWidth, zW Width](x Vec[E, W]) (z Vec[E, zW]) {
	z = makeVec[E, zW]()
	copy(z, x[:x.len()/2])
	return z
}

// GetHi returns the upper half of x.
//
// {{if eq .xL 4}}
//
//	z = {x[2], x[3]}
//
// {{else}}
//
//	z = {x[n/2], x[n/2+1], ..., x[n-1]}, where n = {{lanes .x "x"}}
//
// {{end}}
//
//specgen:require z={xB}{xN}x{xL/2}
func GetHi[E Elt, W SplittableWidth, zW Width](x Vec[E, W]) (z Vec[E, zW]) {
	z = makeVec[E, zW]()
	copy(z, x[x.len()/2:])
	return z
}

// SetLo returns x with its lower half set to y.
//
// {{if eq .xL 4}}
//
//	z = {y[0], y[1], x[2], x[3]}
//
// {{else}}
//
//	z = {y[0], y[1], ..., y[n/2-1], x[n/2], ..., x[n-1]}, where n = {{lanes .x "x"}}
//
// {{end}}
//
// See also [Enlarge].
//
//specgen:require y={xB}{xN}x{xL/2}
func SetLo[E Elt, W SplittableWidth, yW Width](x Vec[E, W], y Vec[E, yW]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	copy(z, x)
	copy(z, y)
	return z
}

// SetHi returns x with its upper half set to y.
//
// {{if eq .xL 4}}
//
//	z = {x[0], x[1], y[0], y[1]}
//
// {{else}}
//
//	z = {x[0], ..., x[n/2-1], y[0], y[1], ..., y[n/2-1]}, where n = {{lanes .x "x"}}
//
// {{end}}
//
//specgen:require y={xB}{xN}x{xL/2}
func SetHi[E Elt, W SplittableWidth, yW Width](x Vec[E, W], y Vec[E, yW]) (z Vec[E, W]) {
	z = makeVec[E, W]()
	copy(z, x)
	copy(z[z.len()/2:], y)
	return z
}

// Enlarge returns x with twice as many elements, with the upper half zeroed.
// [GetLo] reverses this: x.Enlarge().GetLo() returns x.
//
// {{if eq .xL 2}}
//
//	z = concat(x, {0, 0})
//
// {{else}}
//
//	z = concat(x, {0, ..., 0})
//
// {{end}}
//
//specgen:require z={xB}{xN}x{xL*2}
func Enlarge[E Elt, W DoublableWidth, zW Width](x Vec[E, W]) (z Vec[E, zW]) {
	z = makeVec[E, zW]()
	copy(z, x)
	return z
}

// String returns x in the form "{x[0],x[1],...}".
func String[E Nums, W Width](x Vec[E, W]) string {
	var buf strings.Builder
	buf.WriteByte('{')
	for i, val := range x {
		if i > 0 {
			buf.WriteByte(',')
		}
		fmt.Fprint(&buf, val)
	}
	buf.WriteByte('}')
	return buf.String()
}

// StringMask returns x in the form "{x[0],x[1],...}", where 0 means false and 1
// means true.
//
//specgen:name String
func StringMask[E MaskElt, W Width](x Vec[E, W]) string {
	var buf strings.Builder
	buf.WriteByte('{')
	for i, val := range x {
		if i > 0 {
			buf.WriteByte(',')
		}
		if val == 0 {
			buf.WriteByte('0')
		} else {
			buf.WriteByte('1')
		}
	}
	buf.WriteByte('}')
	return buf.String()
}
