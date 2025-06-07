// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements support functionality for iimport.go.

package importer

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"fmt"
	"go/token"
	"internal/pkgbits"
	"sync"
)

func assert(p bool) {
	base.Assert(p)
}

func errorf(format string, args ...interface{}) {
	panic(fmt.Sprintf(format, args...))
}

// Synthesize a token.Pos
type fakeFileSet struct {
	fset  *token.FileSet
	files map[string]*token.File
}

func (s *fakeFileSet) pos(file string, line, column int) token.Pos {
	// TODO(mdempsky): Make use of column.

	// Since we don't know the set of needed file positions, we
	// reserve maxlines positions per file.
	const maxlines = 64 * 1024
	f := s.files[file]
	if f == nil {
		f = s.fset.AddFile(file, -1, maxlines)
		s.files[file] = f
		// Allocate the fake linebreak indices on first use.
		// TODO(adonovan): opt: save ~512KB using a more complex scheme?
		fakeLinesOnce.Do(func() {
			fakeLines = make([]int, maxlines)
			for i := range fakeLines {
				fakeLines[i] = i
			}
		})
		f.SetLines(fakeLines)
	}

	if line > maxlines {
		line = 1
	}

	// Treat the file as if it contained only newlines
	// and column=1: use the line number as the offset.
	return f.Pos(line - 1)
}

var (
	fakeLines     []int
	fakeLinesOnce sync.Once
)

func chanDir(d int) types2.ChanDir {
	switch types.ChanDir(d) {
	case types.Crecv:
		return types2.RecvOnly
	case types.Csend:
		return types2.SendOnly
	case types.Cboth:
		return types2.SendRecv
	default:
		errorf("unexpected channel dir %d", d)
		return 0
	}
}

var predeclared = []types2.Type{
	// basic types
	types2.Typ[types2.Bool],
	types2.Typ[types2.Int],
	types2.Typ[types2.Int8],
	types2.Typ[types2.Int16],
	types2.Typ[types2.Int32],
	types2.Typ[types2.Int64],
	types2.Typ[types2.Uint],
	types2.Typ[types2.Uint8],
	types2.Typ[types2.Uint16],
	types2.Typ[types2.Uint32],
	types2.Typ[types2.Uint64],
	types2.Typ[types2.Uintptr],
	types2.Typ[types2.Float32],
	types2.Typ[types2.Float64],
	types2.Typ[types2.Complex64],
	types2.Typ[types2.Complex128],
	types2.Typ[types2.String],

	// basic type aliases
	types2.Universe.Lookup("byte").Type(),
	types2.Universe.Lookup("rune").Type(),

	// error
	types2.Universe.Lookup("error").Type(),

	// untyped types
	types2.Typ[types2.UntypedBool],
	types2.Typ[types2.UntypedInt],
	types2.Typ[types2.UntypedRune],
	types2.Typ[types2.UntypedFloat],
	types2.Typ[types2.UntypedComplex],
	types2.Typ[types2.UntypedString],
	types2.Typ[types2.UntypedNil],

	// package unsafe
	types2.Typ[types2.UnsafePointer],

	// invalid type
	types2.Typ[types2.Invalid], // only appears in packages with errors

	// used internally by gc; never used by this package or in .a files
	// not to be confused with the universe any
	anyType{},

	// comparable
	types2.Universe.Lookup("comparable").Type(),

	// "any" has special handling: see usage of predeclared.
}

type anyType struct{}

func (t anyType) Underlying() types2.Type { return t }
func (t anyType) String() string          { return "any" }

// See cmd/compile/internal/noder.derivedInfo.
type derivedInfo struct {
	idx    pkgbits.Index
	needed bool
}

// See cmd/compile/internal/noder.typeInfo.
type typeInfo struct {
	idx     pkgbits.Index
	derived bool
}
