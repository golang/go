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

	// Funcs contains all (instantiated) functions, methods, and
	// function literals to be compiled.
	Funcs []*Func

	// Externs holds constants, (non-generic) types, and variables
	// declared at package scope.
	Externs []*Name

	// AsmHdrDecls holds declared constants and struct types that should
	// be included in -asmhdr output. It's only populated when -asmhdr
	// is set.
	AsmHdrDecls []*Name

	// Cgo directives.
	CgoPragmas [][]string

	// Variables with //go:embed lines.
	Embeds []*Name

	// PluginExports holds exported functions and variables that are
	// accessible through the package plugin API. It's only populated
	// for -buildmode=plugin (i.e., compiling package main and -dynlink
	// is set).
	PluginExports []*Name
}
