// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=749

package main

func f() (ok bool) { return false }

func main() {
	var i interface{}
	i = f
	_ = i.(func()bool)
	_ = i.(func()(bool))
}
