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
func importfunc(s *types.Sym, t *types.Type) {
	fn := ir.NewFunc(src.NoXPos, src.NoXPos, s, t)
	importsym(fn.Nname)
}

// importvar declares symbol s as an imported variable with type t.
func importvar(s *types.Sym, t *types.Type) {
	n := ir.NewNameAt(src.NoXPos, s, t)
	n.Class = ir.PEXTERN
	importsym(n)
}

func importsym(name *ir.Name) {
	sym := name.Sym()
	if sym.Def != nil {
		base.Fatalf("importsym of symbol that already exists: %v", sym.Def)
	}
	sym.Def = name
}
