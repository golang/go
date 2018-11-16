// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"cmd/internal/obj"
	"cmd/internal/src"
	"unicode"
	"unicode/utf8"
)

// Sym represents an object name. Most commonly, this is a Go identifier naming
// an object declared within a package, but Syms are also used to name internal
// synthesized objects.
//
// As an exception, field and method names that are exported use the Sym
// associated with localpkg instead of the package that declared them. This
// allows using Sym pointer equality to test for Go identifier uniqueness when
// handling selector expressions.
type Sym struct {
	Importdef *Pkg   // where imported definition was found
	Linkname  string // link name

	Pkg  *Pkg
	Name string // object name

	// saved and restored by dcopy
	Def        *Node    // definition: ONAME OTYPE OPACK or OLITERAL
	Block      int32    // blocknumber to catch redeclaration
	Lastlineno src.XPos // last declaration for diagnostic

	flags   bitset8
	Label   *Node // corresponding label (ephemeral)
	Origpkg *Pkg  // original package for . import
}

const (
	symOnExportList = 1 << iota // added to exportlist (no need to add again)
	symUniq
	symSiggen // type symbol has been generated
	symAsm    // on asmlist, for writing to -asmhdr
	symAlgGen // algorithm table has been generated
	symFunc   // function symbol; uses internal ABI
)

func (sym *Sym) OnExportList() bool { return sym.flags&symOnExportList != 0 }
func (sym *Sym) Uniq() bool         { return sym.flags&symUniq != 0 }
func (sym *Sym) Siggen() bool       { return sym.flags&symSiggen != 0 }
func (sym *Sym) Asm() bool          { return sym.flags&symAsm != 0 }
func (sym *Sym) AlgGen() bool       { return sym.flags&symAlgGen != 0 }
func (sym *Sym) Func() bool         { return sym.flags&symFunc != 0 }

func (sym *Sym) SetOnExportList(b bool) { sym.flags.set(symOnExportList, b) }
func (sym *Sym) SetUniq(b bool)         { sym.flags.set(symUniq, b) }
func (sym *Sym) SetSiggen(b bool)       { sym.flags.set(symSiggen, b) }
func (sym *Sym) SetAsm(b bool)          { sym.flags.set(symAsm, b) }
func (sym *Sym) SetAlgGen(b bool)       { sym.flags.set(symAlgGen, b) }
func (sym *Sym) SetFunc(b bool)         { sym.flags.set(symFunc, b) }

func (sym *Sym) IsBlank() bool {
	return sym != nil && sym.Name == "_"
}

func (sym *Sym) LinksymName() string {
	if sym.IsBlank() {
		return "_"
	}
	if sym.Linkname != "" {
		return sym.Linkname
	}
	return sym.Pkg.Prefix + "." + sym.Name
}

func (sym *Sym) Linksym() *obj.LSym {
	if sym == nil {
		return nil
	}
	if sym.Func() {
		// This is a function symbol. Mark it as "internal ABI".
		return Ctxt.LookupInit(sym.LinksymName(), func(s *obj.LSym) {
			s.SetABI(obj.ABIInternal)
		})
	}
	return Ctxt.Lookup(sym.LinksymName())
}

// Less reports whether symbol a is ordered before symbol b.
//
// Symbols are ordered exported before non-exported, then by name, and
// finally (for non-exported symbols) by package height and path.
//
// Ordering by package height is necessary to establish a consistent
// ordering for non-exported names with the same spelling but from
// different packages. We don't necessarily know the path for the
// package being compiled, but by definition it will have a height
// greater than any other packages seen within the compilation unit.
// For more background, see issue #24693.
func (a *Sym) Less(b *Sym) bool {
	if a == b {
		return false
	}

	// Exported symbols before non-exported.
	ea := IsExported(a.Name)
	eb := IsExported(b.Name)
	if ea != eb {
		return ea
	}

	// Order by name and then (for non-exported names) by package
	// height and path.
	if a.Name != b.Name {
		return a.Name < b.Name
	}
	if !ea {
		if a.Pkg.Height != b.Pkg.Height {
			return a.Pkg.Height < b.Pkg.Height
		}
		return a.Pkg.Path < b.Pkg.Path
	}
	return false
}

// IsExported reports whether name is an exported Go symbol (that is,
// whether it begins with an upper-case letter).
func IsExported(name string) bool {
	if r := name[0]; r < utf8.RuneSelf {
		return 'A' <= r && r <= 'Z'
	}
	r, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(r)
}
