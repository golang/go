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

	// Init functions, listed in source order.
	Inits []*Func

	// Top-level declarations.
	Decls []Node

	// Extern (package global) declarations.
	Externs []Node

	// Assembly function declarations.
	Asms []*Name

	// Cgo directives.
	CgoPragmas [][]string

	// Variables with //go:embed lines.
	Embeds []*Name

	// Exported (or re-exported) symbols.
	Exports []*Name
}
