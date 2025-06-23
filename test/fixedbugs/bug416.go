// errorcheck

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct {
	X int
}

func (t *T) X() {} // ERROR "type T has both field and method named X|redeclares struct field name|field and method with the same name"
