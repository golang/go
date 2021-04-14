// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/src"
)

// numImport tracks how often a package with a given name is imported.
// It is used to provide a better error message (by using the package
// path to disambiguate) if a package that appears multiple times with
// the same name appears in an error message.
var numImport = make(map[string]int)

func npos(pos src.XPos, n *Node) *Node {
	n.Pos = pos
	return n
}

func builtinCall(op Op) *Node {
	return nod(OCALL, mkname(builtinpkg.Lookup(goopnames[op])), nil)
}
