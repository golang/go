// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s syntax node for M4-S1 property interfaces
// (SPEC Sec 2.3, T-203.2). Upstream files reference it through hook
// lines only (G3).

// PropertyType is the type of a property element inside an interface
// type in LogosMode (SPEC Sec 5.2: PropertyElem ::= ["const"] Name Type):
//
//	interface { const Name T; HP int; Start() error }
//
// It appears only as the Type of a *Field with a non-nil Name in an
// *InterfaceType method list. Const reports whether the element had the
// `const` marker (read-only property); without it the property is
// read-write. Type is the property value type.
//
// A bare `F T` element only becomes this node for an exported F:
// T-203.2 sources every property from an exported field or method of
// the same name, and the issue49205 `m int` error must parse exactly
// like plain Go under LogosMode. The `const F T` form has no such
// collision and accepts any name.
//
// The position of a PropertyType is the position of `const` when the
// marker is present, else the position of the property type.
type PropertyType struct {
	Const bool
	Type  Expr
	expr
}
