// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1662
// iota inside var

package main

var (
	a = iota  // ERROR "undefined: iota|iota is only defined in const"
	b = iota  // ERROR "undefined: iota|iota is only defined in const"
	c = iota  // ERROR "undefined: iota|iota is only defined in const"
)
