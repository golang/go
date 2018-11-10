// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package doc extracts source code documentation from a Go AST.
package doc

import (
	"go/ast"
	"go/token"
)

// Package is the documentation for an entire package.
type Package struct {
	Doc        string
	Name       string
	ImportPath string
	Imports    []string
	Filenames  []string
	Notes      map[string][]*Note

	// Deprecated: For backward compatibility Bugs is still populated,
	// but all new code should use Notes instead.
	Bugs []string

	// declarations
	Consts []*Value
	Types  []*Type
	Vars   []*Value
	Funcs  []*Func
}

// Value is the documentation for a (possibly grouped) var or const declaration.
type Value struct {
	Doc   string
	Names []string // var or const names in declaration order
	Decl  *ast.GenDecl

	order int
}

// Type is the documentation for a type declaration.
type Type struct {
	Doc  string
	Name string
	Decl *ast.GenDecl

	// associated declarations
	Consts  []*Value // sorted list of constants of (mostly) this type
	Vars    []*Value // sorted list of variables of (mostly) this type
	Funcs   []*Func  // sorted list of functions returning this type
	Methods []*Func  // sorted list of methods (including embedded ones) of this type
}

// Func is the documentation for a func declaration.
type Func struct {
	Doc  string
	Name string
	Decl *ast.FuncDecl

	// methods
	// (for functions, these fields have the respective zero value)
	Recv  string // actual   receiver "T" or "*T"
	Orig  string // original receiver "T" or "*T"
	Level int    // embedding level; 0 means not embedded
}

// A Note represents a marked comment starting with "MARKER(uid): note body".
// Any note with a marker of 2 or more upper case [A-Z] letters and a uid of
// at least one character is recognized. The ":" following the uid is optional.
// Notes are collected in the Package.Notes map indexed by the notes marker.
type Note struct {
	Pos, End token.Pos // position range of the comment containing the marker
	UID      string    // uid found with the marker
	Body     string    // note body text
}

// Mode values control the operation of New.
type Mode int

const (
	// extract documentation for all package-level declarations,
	// not just exported ones
	AllDecls Mode = 1 << iota

	// show all embedded methods, not just the ones of
	// invisible (unexported) anonymous fields
	AllMethods
)

// New computes the package documentation for the given package AST.
// New takes ownership of the AST pkg and may edit or overwrite it.
//
func New(pkg *ast.Package, importPath string, mode Mode) *Package {
	var r reader
	r.readPackage(pkg, mode)
	r.computeMethodSets()
	r.cleanupTypes()
	return &Package{
		Doc:        r.doc,
		Name:       pkg.Name,
		ImportPath: importPath,
		Imports:    sortedKeys(r.imports),
		Filenames:  r.filenames,
		Notes:      r.notes,
		Bugs:       noteBodies(r.notes["BUG"]),
		Consts:     sortedValues(r.values, token.CONST),
		Types:      sortedTypes(r.types, mode&AllMethods != 0),
		Vars:       sortedValues(r.values, token.VAR),
		Funcs:      sortedFuncs(r.funcs, true),
	}
}
