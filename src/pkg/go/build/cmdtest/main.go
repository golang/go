// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/build/pkgtest"

func main() {
	pkgtest.Foo()
	print(int(pkgtest.Sqrt(9)))
}
