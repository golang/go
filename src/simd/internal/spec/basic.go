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
