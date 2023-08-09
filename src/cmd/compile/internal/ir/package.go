// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import "cmd/compile/internal/types"

// A Package holds information about the package being compiled.
type Package struct {
	// Imports, listed in source order.
	// See golang.org/issue/31636.
	Imports []*types.Pkg

	// InitOrder is the list of package-level initializers in the order
	// in which they must be executed.
	InitOrder []Node

	// Init functions, listed in source order.
	Inits []*Func

	// Funcs contains all (instantiated) functions, methods, and
	// function literals to be compiled.
	Funcs []*Func

	// Externs holds constants, (non-generic) types, and variables
	// declared at package scope.
	Externs []*Name

	// Assembly function declarations.
	Asms []*Name

	// Cgo directives.
	CgoPragmas [][]string

	// Variables with //go:embed lines.
	Embeds []*Name

	// Exported (or re-exported) symbols.
	Exports []*Name
}
