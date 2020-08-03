// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func a() { //@mark(funcA, "a")
	d()
}

func b() { //@mark(funcB, "b")
	d()
}

func c() { //@mark(funcC, "c")
	d()
}

func d() { //@mark(funcD, "d"),incomingcalls("d", funcA, funcB, funcC),outgoingcalls("d", funcE, funcF, funcG)
	e()
	f()
	g()
}

func e() {} //@mark(funcE, "e")

func f() {} //@mark(funcF, "f")

func g() {} //@mark(funcG, "g")

func main() {
	a()
	b()
	c()
}
