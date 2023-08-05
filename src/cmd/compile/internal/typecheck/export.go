// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// importfunc declares symbol s as an imported function with type t.
// ipkg is the package being imported.
func importfunc(pos src.XPos, s *types.Sym, t *types.Type) *ir.Name {
	n := importobj(pos, s, ir.ONAME, ir.PFUNC, t)
	n.Func = ir.NewFunc(pos)
	n.Func.Nname = n
	return n
}

// importobj declares symbol s as an imported object representable by op.
// ipkg is the package being imported.
func importobj(pos src.XPos, s *types.Sym, op ir.Op, ctxt ir.Class, t *types.Type) *ir.Name {
	n := importsym(pos, s, op, ctxt)
	n.SetType(t)
	if ctxt == ir.PFUNC {
		n.Sym().SetFunc(true)
	}
	return n
}

func importsym(pos src.XPos, s *types.Sym, op ir.Op, ctxt ir.Class) *ir.Name {
	if n := s.PkgDef(); n != nil {
		base.Fatalf("importsym of symbol that already exists: %v", n)
	}

	n := ir.NewDeclNameAt(pos, op, s)
	n.Class = ctxt // TODO(mdempsky): Move this into NewDeclNameAt too?
	s.SetPkgDef(n)
	return n
}

// importvar declares symbol s as an imported variable with type t.
// ipkg is the package being imported.
func importvar(pos src.XPos, s *types.Sym, t *types.Type) *ir.Name {
	return importobj(pos, s, ir.ONAME, ir.PEXTERN, t)
}
