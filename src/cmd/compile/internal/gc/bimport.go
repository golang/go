// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ir"
	"cmd/internal/src"
)

func npos(pos src.XPos, n *ir.Node) *ir.Node {
	n.Pos = pos
	return n
}

func builtinCall(op ir.Op) *ir.Node {
	return ir.Nod(ir.OCALL, mkname(ir.BuiltinPkg.Lookup(ir.OpNames[op])), nil)
}
