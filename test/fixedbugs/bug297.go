// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash; issue 961.

package main

type ByteSize float64
const (
	_ = iota;   // ignore first value by assigning to blank identifier
	KB ByteSize = 1<<(10*X) // ERROR "undefined"
)
