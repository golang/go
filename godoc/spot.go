// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

// ----------------------------------------------------------------------------
// SpotInfo

// A SpotInfo value describes a particular identifier spot in a given file;
// It encodes three values: the SpotKind (declaration or use), a line or
// snippet index "lori", and whether it's a line or index.
//
// The following encoding is used:
//
//	bits    32   4    1       0
//	value    [lori|kind|isIndex]
type SpotInfo uint32

// SpotKind describes whether an identifier is declared (and what kind of
// declaration) or used.
type SpotKind uint32

const (
	PackageClause SpotKind = iota
	ImportDecl
	ConstDecl
	TypeDecl
	VarDecl
	FuncDecl
	MethodDecl
	Use
	nKinds
)

var (
	// These must match the SpotKind values above.
	name = []string{
		"Packages",
		"Imports",
		"Constants",
		"Types",
		"Variables",
		"Functions",
		"Methods",
		"Uses",
		"Unknown",
	}
)

func (x SpotKind) Name() string { return name[x] }

func init() {
	// sanity check: if nKinds is too large, the SpotInfo
	// accessor functions may need to be updated
	if nKinds > 8 {
		panic("internal error: nKinds > 8")
	}
}

// makeSpotInfo makes a SpotInfo.
func makeSpotInfo(kind SpotKind, lori int, isIndex bool) SpotInfo {
	// encode lori: bits [4..32)
	x := SpotInfo(lori) << 4
	if int(x>>4) != lori {
		// lori value doesn't fit - since snippet indices are
		// most certainly always smaller then 1<<28, this can
		// only happen for line numbers; give it no line number (= 0)
		x = 0
	}
	// encode kind: bits [1..4)
	x |= SpotInfo(kind) << 1
	// encode isIndex: bit 0
	if isIndex {
		x |= 1
	}
	return x
}

func (x SpotInfo) Kind() SpotKind { return SpotKind(x >> 1 & 7) }
func (x SpotInfo) Lori() int      { return int(x >> 4) }
func (x SpotInfo) IsIndex() bool  { return x&1 != 0 }
