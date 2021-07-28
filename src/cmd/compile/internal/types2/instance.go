// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

// TODO(rfindley): move this code to named.go.

import "cmd/compile/internal/syntax"

// instance holds position information for use in lazy instantiation.
//
// TODO(rfindley): come up with a better name for this type, now that its usage
// has changed.
type instance struct {
	pos     syntax.Pos   // position of type instantiation; for error reporting only
	posList []syntax.Pos // position of each targ; for error reporting only
	verify  bool         // if set, check constraint satisfaction upon instantiation
}

// expand ensures that the underlying type of n is instantiated.
// The underlying type will be Typ[Invalid] if there was an error.
func (n *Named) expand() {
	if n.instance != nil {
		// n must be loaded before instantiation, in order to have accurate
		// tparams. This is done implicitly by the call to n.TParams, but making it
		// explicit is harmless: load is idempotent.
		n.load()
		inst := n.check.instantiate(n.instance.pos, n.orig.underlying, n.TParams(), n.targs, n.instance.posList)
		n.underlying = inst
		n.fromRHS = inst
		n.instance = nil
	}
}

// expand expands uninstantiated named types and leaves all other types alone.
// expand does not recurse.
func expand(typ Type) Type {
	if t, _ := typ.(*Named); t != nil {
		t.expand()
	}
	return typ
}
