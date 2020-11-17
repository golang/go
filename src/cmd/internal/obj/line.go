// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"cmd/internal/goobj"
	"cmd/internal/src"
)

// AddImport adds a package to the list of imported packages.
func (ctxt *Link) AddImport(pkg string, fingerprint goobj.FingerprintType) {
	ctxt.Imports = append(ctxt.Imports, goobj.ImportedPkg{Pkg: pkg, Fingerprint: fingerprint})
}

func linkgetlineFromPos(ctxt *Link, xpos src.XPos) (f string, l int32) {
	pos := ctxt.PosTable.Pos(xpos)
	if !pos.IsKnown() {
		pos = src.Pos{}
	}
	// TODO(gri) Should this use relative or absolute line number?
	return pos.SymFilename(), int32(pos.RelLine())
}

// getFileIndexAndLine returns the file index (local to the CU), and the line number for a position.
func getFileIndexAndLine(ctxt *Link, xpos src.XPos) (int, int32) {
	f, l := linkgetlineFromPos(ctxt, xpos)
	return ctxt.PosTable.FileIndex(f), l
}
