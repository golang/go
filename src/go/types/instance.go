// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// TODO(rfindley): move this code to named.go.

import "go/token"

// instance holds a Checker along with syntactic information
// information, for use in lazy instantiation.
type instance struct {
	check   *Checker
	pos     token.Pos   // position of type instantiation; for error reporting only
	posList []token.Pos // position of each targ; for error reporting only
}

// complete ensures that the underlying type of n is instantiated.
// The underlying type will be Typ[Invalid] if there was an error.
// TODO(rfindley): expand would be a better name for this method, but conflicts
// with the existing concept of lazy expansion. Need to reconcile this.
func (n *Named) complete() {
	if n.instance != nil && len(n.targs) > 0 && n.underlying == nil {
		check := n.instance.check
		inst := check.instantiate(n.instance.pos, n.orig.underlying, n.TParams().list(), n.targs, n.instance.posList)
		n.underlying = inst
		n.fromRHS = inst
		n.methods = n.orig.methods
	}
}

// expand expands a type instance into its instantiated
// type and leaves all other types alone. expand does
// not recurse.
func expand(typ Type) Type {
	if t, _ := typ.(*Named); t != nil {
		t.complete()
	}
	return typ
}
