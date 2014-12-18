// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR61248: Transformations to recover calls made them fail typechecking in gccgo.

package main

func main() {
	var f func(int, interface{})
	go f(0, recover())
}
