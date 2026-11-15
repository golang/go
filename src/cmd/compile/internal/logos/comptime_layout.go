// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/constant"
	"go/token"
	"strconv"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"internal/buildcfg"
)

func logosLayoutName(call *syntax.CallExpr, sites *cfSites) (string, bool) {
	idx, ok := call.Fun.(*syntax.IndexExpr)
	if !ok || len(call.ArgList) != 0 || sites == nil || sites.comptimeName == "" {
		return "", false
	}
	sel, ok := idx.X.(*syntax.SelectorExpr)
	if !ok {
		return "", false
	}
	qn, ok := sel.X.(*syntax.Name)
	if !ok || qn.Value != sites.comptimeName {
		return "", false
	}
	switch sel.Sel.Value {
	case "SizeOf", "AlignOf", "HasPointers":
		return sel.Sel.Value, true
	}
	return "", false
}

func cfCallSpan(ctx cfCtx, call *syntax.CallExpr) (int, int) {
	start := offAt(ctx.starts, syntax.StartPos(call))
	end := cfExactEnd(ctx, call.Fun)
	depth := 0
	scanSegment(ctx.buf.b, end, len(ctx.buf.b), func(t tok) bool {
		switch t.kind {
		case token.LPAREN:
			depth++
		case token.RPAREN:
			depth--
			if depth < 0 {
				return false
			}
			if depth == 0 {
				end = t.end
				return false
			}
		case token.COMMENT:
		default:
			return depth > 0
		}
		return true
	})
	return start, end
}

func (fl *fileLower) logosLayoutCallValue(call *syntax.CallExpr, active cfChain, sites *cfSites) (constant.Value, bool) {
	name, ok := logosLayoutName(call, sites)
	if !ok {
		return nil, false
	}
	targs := syntax.UnpackListExpr(call.Fun.(*syntax.IndexExpr).Index)
	if len(targs) != 1 {
		return nil, false
	}
	t, ok := fl.cfResolveTypeExpr(targs[0], active, sites)
	if !ok || cfHasVarSize(t, make(map[types2.Type]bool)) {
		return nil, false
	}
	sizes := types2.SizesFor("gc", buildcfg.GOARCH)
	switch name {
	case "SizeOf":
		return constant.MakeInt64(sizes.Sizeof(t)), true
	case "AlignOf":
		return constant.MakeInt64(sizes.Alignof(t)), true
	}
	return constant.MakeBool(types2.LogosHasPointers(t)), true
}

func logosLayoutText(v constant.Value) string {
	if v.Kind() == constant.Bool {
		return strconv.FormatBool(constant.BoolVal(v))
	}
	return v.String()
}

func cfHasVarSize(t types2.Type, seen map[types2.Type]bool) bool {
	t = types2.Unalias(t)
	if seen[t] {
		return false
	}
	seen[t] = true
	switch t := t.(type) {
	case *types2.TypeParam:
		return true
	case *types2.Named:
		return cfHasVarSize(t.Underlying(), seen)
	case *types2.Pointer:
		return cfHasVarSize(t.Elem(), seen)
	case *types2.Slice:
		return cfHasVarSize(t.Elem(), seen)
	case *types2.Array:
		return cfHasVarSize(t.Elem(), seen)
	case *types2.Map:
		return cfHasVarSize(t.Key(), seen) || cfHasVarSize(t.Elem(), seen)
	case *types2.Chan:
		return cfHasVarSize(t.Elem(), seen)
	case *types2.Option:
		return cfHasVarSize(t.Elem, seen)
	case *types2.ConstView:
		return cfHasVarSize(t.Elem, seen)
	case *types2.LogosTuple:
		return cfHasVarSize(t.Underlying(), seen)
	case *types2.Struct:
		for i := range t.NumFields() {
			if cfHasVarSize(t.Field(i).Type(), seen) {
				return true
			}
		}
	}
	return false
}
