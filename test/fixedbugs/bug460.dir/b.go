// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

var x a.Foo

func main() {
	x.int = 20 // ERROR "unexported field"
}

