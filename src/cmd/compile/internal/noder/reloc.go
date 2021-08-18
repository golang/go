// UNREVIEWED

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

// A reloc indicates a particular section within a unified IR export.
//
// TODO(mdempsky): Rename to "section" or something similar?
type reloc int

// A relocEnt (relocation entry) is an entry in an atom's local
// reference table.
//
// TODO(mdempsky): Rename this too.
type relocEnt struct {
	kind reloc
	idx  int
}

// Reserved indices within the meta relocation section.
const (
	publicRootIdx  = 0
	privateRootIdx = 1
)

const (
	relocString reloc = iota
	relocMeta
	relocPosBase
	relocPkg
	relocName
	relocType
	relocObj
	relocObjExt
	relocObjDict
	relocBody

	numRelocs = iota
)
