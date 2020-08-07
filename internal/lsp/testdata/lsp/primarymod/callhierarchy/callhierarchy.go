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
const x = func() { //@mark(hierarchyLiteral, "func")
	D()
}

// D is exported to test incoming/outgoing calls across packages
func D() { //@mark(hierarchyD, "D"),incomingcalls(hierarchyD, hierarchyA, hierarchyB, hierarchyC, hierarchyLiteral, incomingA),outgoingcalls(hierarchyD, hierarchyE, hierarchyF, hierarchyG, outgoingB)
	e()
	f()
	g()
	outgoing.B()
}

func e() {} //@mark(hierarchyE, "e")

func f() {} //@mark(hierarchyF, "f")

func g() {} //@mark(hierarchyG, "g")
