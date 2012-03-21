// skip # used by embed1.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that embedded interface types can have local methods.

package p

type T int
func (t T) m() {}

type I interface { m() }
type J interface { I }

func main() {
	var i I
	var j J
	var t T
	i = t
	j = t
	_ = i
	_ = j
	i = j
	_ = i
	j = i
	_ = j
}
