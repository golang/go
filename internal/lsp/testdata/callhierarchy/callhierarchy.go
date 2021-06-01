// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package callhierarchy

import "golang.org/x/tools/internal/lsp/callhierarchy/outgoing"

func a() { //@mark(hierarchyA, "a")
	D()
}

func b() { //@mark(hierarchyB, "b")
	D()
}

// C is an exported function
func C() { //@mark(hierarchyC, "C")
	D()
	D()
}

// To test hierarchy across function literals
var x = func() { //@mark(hierarchyLiteral, "func"),mark(hierarchyLiteralOut, "x")
	D()
}

// D is exported to test incoming/outgoing calls across packages
func D() { //@mark(hierarchyD, "D"),incomingcalls(hierarchyD, hierarchyA, hierarchyB, hierarchyC, hierarchyLiteral, incomingA),outgoingcalls(hierarchyD, hierarchyE, hierarchyF, hierarchyG, hierarchyLiteralOut, outgoingB, hierarchyFoo, hierarchyH, hierarchyI, hierarchyJ, hierarchyK)
	e()
	x()
	F()
	outgoing.B()
	foo := func() {} //@mark(hierarchyFoo, "foo"),incomingcalls(hierarchyFoo, hierarchyD),outgoingcalls(hierarchyFoo)
	foo()

	func() {
		g()
	}()

	var i Interface = impl{}
	i.H()
	i.I()

	s := Struct{}
	s.J()
	s.K()
}

func e() {} //@mark(hierarchyE, "e")

// F is an exported function
func F() {} //@mark(hierarchyF, "F")

func g() {} //@mark(hierarchyG, "g")

type Interface interface {
	H() //@mark(hierarchyH, "H")
	I() //@mark(hierarchyI, "I")
}

type impl struct{}

func (i impl) H() {}
func (i impl) I() {}

type Struct struct {
	J func() //@mark(hierarchyJ, "J")
	K func() //@mark(hierarchyK, "K")
}
