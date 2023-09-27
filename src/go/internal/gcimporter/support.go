// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements support functionality for iimport.go.

package gcimporter

import (
	"fmt"
	"go/token"
	"go/types"
	"internal/pkgbits"
	"sync"
)

func assert(b bool) {
	if !b {
		panic("assertion failed")
	}
}

func errorf(format string, args ...any) {
	panic(fmt.Sprintf(format, args...))
}

// deltaNewFile is a magic line delta offset indicating a new file.
// We use -64 because it is rare; see issue 20080 and CL 41619.
// -64 is the smallest int that fits in a single byte as a varint.
const deltaNewFile = -64

// Synthesize a token.Pos
type fakeFileSet struct {
	fset  *token.FileSet
	files map[string]*fileInfo
}

type fileInfo struct {
	file     *token.File
	lastline int
}

const maxlines = 64 * 1024

func (s *fakeFileSet) pos(file string, line, column int) token.Pos {
	// TODO(mdempsky): Make use of column.

	// Since we don't know the set of needed file positions, we reserve
	// maxlines positions per file. We delay calling token.File.SetLines until
	// all positions have been calculated (by way of fakeFileSet.setLines), so
	// that we can avoid setting unnecessary lines. See also golang/go#46586.
	f := s.files[file]
	if f == nil {
		f = &fileInfo{file: s.fset.AddFile(file, -1, maxlines)}
		s.files[file] = f
	}

	if line > maxlines {
		line = 1
	}
	if line > f.lastline {
		f.lastline = line
	}

	// Return a fake position assuming that f.file consists only of newlines.
	return token.Pos(f.file.Base() + line - 1)
}

func (s *fakeFileSet) setLines() {
	fakeLinesOnce.Do(func() {
		fakeLines = make([]int, maxlines)
		for i := range fakeLines {
			fakeLines[i] = i
		}
	})
	for _, f := range s.files {
		f.file.SetLines(fakeLines[:f.lastline])
	}
}

var (
	fakeLines     []int
	fakeLinesOnce sync.Once
)

func chanDir(d int) types.ChanDir {
	// tag values must match the constants in cmd/compile/internal/gc/go.go
	switch d {
	case 1 /* Crecv */ :
		return types.RecvOnly
	case 2 /* Csend */ :
		return types.SendOnly
	case 3 /* Cboth */ :
		return types.SendRecv
	default:
		errorf("unexpected channel dir %d", d)
		return 0
	}
}

var predeclared = []types.Type{
	// basic types
	types.Typ[types.Bool],
	types.Typ[types.Int],
	types.Typ[types.Int8],
	types.Typ[types.Int16],
	types.Typ[types.Int32],
	types.Typ[types.Int64],
	types.Typ[types.Uint],
	types.Typ[types.Uint8],
	types.Typ[types.Uint16],
	types.Typ[types.Uint32],
	types.Typ[types.Uint64],
	types.Typ[types.Uintptr],
	types.Typ[types.Float32],
	types.Typ[types.Float64],
	types.Typ[types.Complex64],
	types.Typ[types.Complex128],
	types.Typ[types.String],

	// basic type aliases
	types.Universe.Lookup("byte").Type(),
	types.Universe.Lookup("rune").Type(),

	// error
	types.Universe.Lookup("error").Type(),

	// untyped types
	types.Typ[types.UntypedBool],
	types.Typ[types.UntypedInt],
	types.Typ[types.UntypedRune],
	types.Typ[types.UntypedFloat],
	types.Typ[types.UntypedComplex],
	types.Typ[types.UntypedString],
	types.Typ[types.UntypedNil],

	// package unsafe
	types.Typ[types.UnsafePointer],

	// invalid type
	types.Typ[types.Invalid], // only appears in packages with errors

	// used internally by gc; never used by this package or in .a files
	// not to be confused with the universe any
	anyType{},

	// comparable
	types.Universe.Lookup("comparable").Type(),

	// any
	types.Universe.Lookup("any").Type(),
}

type anyType struct{}

func (t anyType) Underlying() types.Type { return t }
func (t anyType) String() string         { return "any" }

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

// See cmd/compile/internal/types.SplitVargenSuffix.
func splitVargenSuffix(name string) (base, suffix string) {
	i := len(name)
	for i > 0 && name[i-1] >= '0' && name[i-1] <= '9' {
		i--
	}
	const dot = "·"
	if i >= len(dot) && name[i-len(dot):i] == dot {
		i -= len(dot)
		return name[:i], name[i:]
	}
	return name, ""
}
