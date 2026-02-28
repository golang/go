// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements support functionality for ureader.go.

package gcimporter

import (
	"fmt"
	"go/token"
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
