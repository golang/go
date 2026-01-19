// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements support functionality for iimport.go.

package importer

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types2"
	"go/token"
	"internal/pkgbits"
)

func assert(p bool) {
	base.Assert(p)
}

const deltaNewFile = -64 // see cmd/compile/internal/gc/bexport.go

// Synthesize a token.Pos
type fakeFileSet struct {
	fset  *token.FileSet
	files map[string]*token.File
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
