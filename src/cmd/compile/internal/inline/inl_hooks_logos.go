// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import "cmd/compile/internal/ir"

func logosInlImpossible(fn *ir.Func) string {
	if sym := fn.Nname.Sym(); sym != nil && sym.Pkg != nil && sym.Pkg.Path == "reflect" {
		switch sym.Name {
		case "Value.Method", "Value.MethodByName",
			"(*rtype).Method", "(*rtype).MethodByName",
			"(*interfaceType).Method", "(*interfaceType).MethodByName":
			return "reflect method lookup must stay visible to walk.usemethod"
		}
	}
	return ""
}

func logosInlStaticValue(fn ir.Node, ro *ir.ReassignOracle) ir.Node {
	if ro != nil {
		return ro.StaticValue(fn)
	}
	return ir.StaticValue(fn)
}

func logosInlRegionNoInline(callee *ir.Func) bool {
	return ir.LogosRegionsEnabled() && ir.LogosRegionNoInline(callee)
}
