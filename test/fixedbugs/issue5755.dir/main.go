// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	a.Test1("frumious")
	a.Test2("frumious")
	a.Test3("frumious")
	a.Test4("frumious")

	a.Test5(nil)
	a.Test6(nil)
	a.Test7(nil)
	a.Test8(nil)
	a.Test9(0)

	a.TestBar()
	a.IsBaz(nil)
}
