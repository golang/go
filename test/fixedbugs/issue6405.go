// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 6405: spurious 'not enough arguments to return' error

package p

func Open() (int, error) {
	return OpenFile() // ERROR "undefined: OpenFile"
}
