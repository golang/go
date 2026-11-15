// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"go/constant"
	"go/token"
	"strconv"
	"strings"
)

type constMatInit struct {
	pkg string
	val ctfe.Value
}

var constMatInits = map[syntax.Expr]constMatInit{}

func clearConstMat() {
	clear(constMatInits)
}

func (fl *fileLower) registerKeepCall(e syntax.Expr, v ctfe.Value) {
	constMatInits[e] = constMatInit{pkg: fl.pkg.tpkg.Path(), val: v}
	if u := syntax.Unparen(e); u != e {
		constMatInits[u] = constMatInit{pkg: fl.pkg.tpkg.Path(), val: v}
	}
}

func LookupConstMatInit(pkg string, e syntax.Expr) (ctfe.Value, bool) {
	if m, ok := constMatInits[e]; ok && m.pkg == pkg {
		return m.val, true
	}
	if u := syntax.Unparen(e); u != e {
		if m, ok := constMatInits[u]; ok && m.pkg == pkg {
			return m.val, true
		}
	}
	return ctfe.Value{}, false
}

func (fl *fileLower) importedConstTrigger(obj *types2.Const, init syntax.Expr) bool {
	if named, ok := types2.Unalias(obj.Type()).(*types2.Named); ok {
		if pkg := named.Obj().Pkg(); pkg != nil && pkg != fl.pkg.tpkg {
			return true
		}
	}
	found := false
	syntax.Inspect(init, func(n syntax.Node) bool {
		if found {
			return false
		}
		c, ok := n.(*syntax.CallExpr)
		if !ok {
			return true
		}
		sel, ok := syntax.Unparen(c.Fun).(*syntax.SelectorExpr)
		if !ok {
			return true
		}
		if s, ok := fl.pkg.info.Selections[sel]; ok {
			if fn, ok := s.Obj().(*types2.Func); ok {
				if pkg := fn.Pkg(); pkg != nil && pkg != fl.pkg.tpkg {
					found = true
					return false
				}
			}
		}
		if obj, ok := fl.pkg.info.Uses[sel.Sel]; ok {
			if fn, ok := obj.(*types2.Func); ok {
				if pkg := fn.Pkg(); pkg != nil && pkg != fl.pkg.tpkg {
					found = true
					return false
				}
			}
		}
		return true
	})
	return found
}

func (fl *fileLower) materializeImportedConst(cd *syntax.ConstDecl, obj *types2.Const, v ctfe.Value, init syntax.Expr) (ctfeResult, bool) {
	var zero ctfeResult
	t := v.Type()
	if t == nil {
		return zero, false
	}
	if t.Kind == ctfe.KMap {
		if _, ok := fl.pkg.constMaps[obj]; ok {
			return ctfeResult{obj: obj, val: v, init: init, lit: init, text: renderExpr(init)}, true
		}
		return zero, false
	}
	if !fl.importedConstTrigger(obj, init) {
		return zero, false
	}
	pos := syntax.StartPos(init)
	if v.IsComposite() {
		return fl.materializeComposite(obj, v, init, pos)
	}
	return fl.materializeScalar(obj, v, init, pos)
}

func (fl *fileLower) materializeScalar(obj *types2.Const, v ctfe.Value, init syntax.Expr, pos syntax.Pos) (ctfeResult, bool) {
	var zero ctfeResult
	if b, ok := types2.Unalias(obj.Type()).(*types2.Basic); ok && b.Info()&types2.IsUntyped != 0 {
		return zero, false
	}
	lit0, err := v.Expr(pos)
	if err != nil {
		return zero, false
	}
	text0 := renderExpr(lit0)
	if !spellableMatType(obj.Type(), fl.pkg.tpkg) {
		if r, ok := fl.ownedScalarBridge(obj, v, init, pos); ok {
			return r, true
		}
		return zero, false
	}
	spell, ok := fl.spellLowered(obj.Type())
	if !ok {
		return zero, false
	}
	te, ok := reparseTypeAt(spell, pos)
	if !ok {
		return zero, false
	}
	conv := at(&syntax.CallExpr{Fun: te, ArgList: []syntax.Expr{lit0}}, pos)
	return ctfeResult{obj: obj, val: v, init: init, lit: conv, text: spell + "(" + text0 + ")"}, true
}

func (fl *fileLower) materializeComposite(obj *types2.Const, v ctfe.Value, init syntax.Expr, pos syntax.Pos) (ctfeResult, bool) {
	if lit, text, ok := fl.buildMatValue(v, obj.Type(), pos); ok {
		return ctfeResult{obj: obj, val: v, init: init, lit: lit, text: text}, true
	}
	if r, ok := fl.ownedCompositeBridge(obj, v, init, pos); ok {
		return r, true
	}
	return fl.keepCallResult(obj, v, init)
}

func (fl *fileLower) keepCallResult(obj *types2.Const, v ctfe.Value, init syntax.Expr) (ctfeResult, bool) {
	fl.registerKeepCall(init, v)
	return ctfeResult{obj: obj, val: v, init: init, lit: init, text: renderExpr(init), keepCall: true}, true
}

func reparseTypeAt(text string, pos syntax.Pos) (syntax.Expr, bool) {
	te, ok := reparseType(text)
	if !ok {
		return nil, false
	}
	syntax.Inspect(te, func(n syntax.Node) bool {
		if n != nil {
			n.SetPos(pos)
		}
		return true
	})
	return te, true
}

func spellableMatType(t types2.Type, self *types2.Package) bool {
	if cv, ok := t.(*types2.ConstView); ok {
		t = cv.Elem
	}
	if !spellableType(t, self) {
		return false
	}
	return matFieldsAssignable(t, self)
}

func matFieldsAssignable(t types2.Type, self *types2.Package) bool {
	seen := make(map[types2.Type]bool)
	var walk func(t types2.Type) bool
	walk = func(t types2.Type) bool {
		if t == nil || seen[t] {
			return true
		}
		seen[t] = true
		u := types2.Unalias(t)
		if n, ok := u.(*types2.Named); ok {
			u = n.Underlying()
		}
		switch u := u.(type) {
		case *types2.Struct:
			for i := range u.NumFields() {
				f := u.Field(i)
				if !f.Exported() && f.Pkg() != self {
					return false
				}
				if !walk(f.Type()) {
					return false
				}
			}
			return true
		case *types2.Slice:
			return walk(u.Elem())
		case *types2.Array:
			return walk(u.Elem())
		case *types2.Map:
			return walk(u.Key()) && walk(u.Elem())
		case *types2.Pointer:
			return walk(u.Elem())
		}
		return true
	}
	return walk(t)
}

func (fl *fileLower) buildMatValue(v ctfe.Value, t types2.Type, pos syntax.Pos) (syntax.Expr, string, bool) {
	if !spellableMatType(t, fl.pkg.tpkg) {
		return nil, "", false
	}
	if v.IsNilValue() {
		spell, ok := fl.spellLowered(t)
		if !ok {
			return nil, "", false
		}
		te, ok := reparseTypeAt(spell, pos)
		if !ok {
			return nil, "", false
		}
		return at(&syntax.CallExpr{Fun: te, ArgList: []syntax.Expr{nilOf(pos)}}, pos), spell + "(nil)", true
	}
	spell, ok := fl.spellLowered(t)
	if !ok {
		return nil, "", false
	}
	te, ok := reparseTypeAt(spell, pos)
	if !ok {
		return nil, "", false
	}
	ut := types2.Unalias(t)
	if n, ok := ut.(*types2.Named); ok {
		ut = n.Underlying()
	}
	_, isNamed := t.(*types2.Named)
	if _, isAlias := t.(*types2.Alias); isAlias {
		isNamed = true
	}
	switch u := ut.(type) {
	case *types2.Basic:
		leaf, err := v.Expr(pos)
		if err != nil {
			return nil, "", false
		}
		if !isNamed {
			return leaf, renderExpr(leaf), true
		}
		return at(&syntax.CallExpr{Fun: te, ArgList: []syntax.Expr{leaf}}, pos), spell + "(" + renderExpr(leaf) + ")", true
	case *types2.Struct:
		fields := v.Fields()
		if len(fields) != u.NumFields() {
			return nil, "", false
		}
		elems := make([]syntax.Expr, len(fields))
		texts := make([]string, len(fields))
		for i := range fields {
			e, s, ok := fl.buildMatValue(fields[i], u.Field(i).Type(), pos)
			if !ok {
				return nil, "", false
			}
			elems[i], texts[i] = e, s
		}
		lit := at(&syntax.CompositeLit{Type: te, ElemList: elems, Rbrace: pos}, pos)
		setRbrace(lit, pos)
		return lit, spell + "{" + strings.Join(texts, ", ") + "}", true
	case *types2.Array:
		array := v.Elems()
		if int64(len(array)) != u.Len() {
			return nil, "", false
		}
		elems := make([]syntax.Expr, len(array))
		texts := make([]string, len(array))
		for i := range array {
			e, s, ok := fl.buildMatValue(array[i], u.Elem(), pos)
			if !ok {
				return nil, "", false
			}
			elems[i], texts[i] = e, s
		}
		lit := at(&syntax.CompositeLit{Type: te, ElemList: elems, Rbrace: pos}, pos)
		setRbrace(lit, pos)
		return lit, spell + "{" + strings.Join(texts, ", ") + "}", true
	case *types2.Slice:
		backing, lo, length, cap_, ok := v.SliceHeader()
		if !ok {
			return nil, "", false
		}
		if lo < 0 || length < 0 || lo > cap_ || lo+length > cap_ || cap_ > len(backing) {
			return nil, "", false
		}
		if !v.SliceNeedsBacking() {
			vis := backing[lo : lo+length]
			elems := make([]syntax.Expr, len(vis))
			texts := make([]string, len(vis))
			for i := range vis {
				e, s, ok := fl.buildMatValue(vis[i], u.Elem(), pos)
				if !ok {
					return nil, "", false
				}
				elems[i], texts[i] = e, s
			}
			lit := at(&syntax.CompositeLit{Type: te, ElemList: elems, Rbrace: pos}, pos)
			setRbrace(lit, pos)
			return lit, spell + "{" + strings.Join(texts, ", ") + "}", true
		}
		acc := backing[lo:cap_]
		elems := make([]syntax.Expr, len(acc))
		texts := make([]string, len(acc))
		for i := range acc {
			e, s, ok := fl.buildMatValue(acc[i], u.Elem(), pos)
			if !ok {
				return nil, "", false
			}
			elems[i], texts[i] = e, s
		}
		backingLit := at(&syntax.CompositeLit{Type: te, ElemList: elems, Rbrace: pos}, pos)
		setRbrace(backingLit, pos)
		loT, hiT, mxT := "0", strconv.Itoa(length), strconv.Itoa(cap_-lo)
		sl := at(&syntax.SliceExpr{
			X:     backingLit,
			Index: [3]syntax.Expr{litOf(pos, loT, syntax.IntLit), litOf(pos, hiT, syntax.IntLit), litOf(pos, mxT, syntax.IntLit)},
			Full:  true,
		}, pos)
		btext := spell + "{" + strings.Join(texts, ", ") + "}"
		return sl, btext + "[" + loT + ":" + hiT + ":" + mxT + "]", true
	case *types2.Map:
		keys, vals := v.MapEntries()
		elems := make([]syntax.Expr, len(keys))
		texts := make([]string, len(keys))
		for i := range keys {
			kx, ks, ok := fl.buildMatValue(keys[i], u.Key(), pos)
			if !ok {
				return nil, "", false
			}
			vx, vs, ok := fl.buildMatValue(vals[i], u.Elem(), pos)
			if !ok {
				return nil, "", false
			}
			elems[i] = at(&syntax.KeyValueExpr{Key: kx, Value: vx}, pos)
			texts[i] = ks + ": " + vs
		}
		lit := at(&syntax.CompositeLit{Type: te, ElemList: elems, NKeys: len(elems), Rbrace: pos}, pos)
		setRbrace(lit, pos)
		return lit, spell + "{" + strings.Join(texts, ", ") + "}", true
	}
	return nil, "", false
}

func matConstant(v ctfe.Value) (constant.Value, bool) {
	t := v.Type()
	if t == nil {
		return nil, false
	}
	switch t.Kind {
	case ctfe.KBool, ctfe.KUntypedBool:
		return constant.MakeBool(v.Bool()), true
	case ctfe.KString, ctfe.KUntypedString:
		return constant.MakeString(v.Str()), true
	case ctfe.KInt, ctfe.KInt8, ctfe.KInt16, ctfe.KInt32, ctfe.KInt64,
		ctfe.KUint, ctfe.KUint8, ctfe.KUint16, ctfe.KUint32, ctfe.KUint64, ctfe.KUintptr,
		ctfe.KUntypedInt, ctfe.KUntypedRune:
		if v.BigInt() == nil {
			return nil, false
		}
		return constant.MakeFromLiteral(v.BigInt().String(), token.INT, 0), true
	case ctfe.KFloat32, ctfe.KFloat64, ctfe.KUntypedFloat:
		return constant.MakeFromLiteral(strconv.FormatFloat(v.Float64(), 'g', -1, 64), token.FLOAT, 0), true
	}
	return nil, false
}

func stampMatValue(n syntax.Expr, t types2.Type, c constant.Value) {
	tv := syntax.TypeAndValue{Type: t, Value: c}
	tv.SetIsValue()
	n.SetTypeInfo(tv)
}

func matNil(t types2.Type, pos syntax.Pos) syntax.Expr {
	n := nilOf(pos)
	tv := syntax.TypeAndValue{Type: t}
	tv.SetIsValue()
	tv.SetIsNil()
	n.SetTypeInfo(tv)
	return n
}

func BuildMatLiteral(v ctfe.Value, t types2.Type, pos syntax.Pos) syntax.Expr {
	if v.IsNilValue() {
		return nil
	}
	return buildMatLiteral(v, t, pos)
}

func buildMatLiteral(v ctfe.Value, t types2.Type, pos syntax.Pos) syntax.Expr {
	if v.IsNilValue() || v.Type() != nil && v.Type().Impure != "" {
		return matNil(t, pos)
	}
	ut := types2.Unalias(t)
	if n, ok := ut.(*types2.Named); ok {
		ut = n.Underlying()
	}
	switch u := ut.(type) {
	case *types2.Basic:
		return buildMatScalar(v, t, pos)
	case *types2.Struct:
		fields := v.Fields()
		if len(fields) != u.NumFields() {
			return nil
		}
		elems := make([]syntax.Expr, len(fields))
		for i := range fields {
			e := buildMatLiteral(fields[i], u.Field(i).Type(), pos)
			if e == nil {
				return nil
			}
			elems[i] = e
		}
		lit := &syntax.CompositeLit{ElemList: elems, Rbrace: pos}
		lit.SetPos(pos)
		stampMatValue(lit, t, nil)
		return lit
	case *types2.Array:
		array := v.Elems()
		if int64(len(array)) != u.Len() {
			return nil
		}
		elems := make([]syntax.Expr, len(array))
		for i := range array {
			e := buildMatLiteral(array[i], u.Elem(), pos)
			if e == nil {
				return nil
			}
			elems[i] = e
		}
		lit := &syntax.CompositeLit{ElemList: elems, Rbrace: pos}
		lit.SetPos(pos)
		stampMatValue(lit, t, nil)
		return lit
	case *types2.Slice:
		backing, lo, length, cap_, ok := v.SliceHeader()
		if !ok {
			return nil
		}
		if lo < 0 || length < 0 || lo > cap_ || lo+length > cap_ || cap_ > len(backing) {
			return nil
		}
		elemT := u.Elem()
		if !v.SliceNeedsBacking() {
			vis := backing[lo : lo+length]
			elems := make([]syntax.Expr, len(vis))
			for i := range vis {
				e := buildMatLiteral(vis[i], elemT, pos)
				if e == nil {
					return nil
				}
				elems[i] = e
			}
			lit := &syntax.CompositeLit{ElemList: elems, Rbrace: pos}
			lit.SetPos(pos)
			stampMatValue(lit, t, nil)
			return lit
		}
		acc := backing[lo:cap_]
		elems := make([]syntax.Expr, len(acc))
		for i := range acc {
			e := buildMatLiteral(acc[i], elemT, pos)
			if e == nil {
				return nil
			}
			elems[i] = e
		}
		backingLit := &syntax.CompositeLit{ElemList: elems, Rbrace: pos}
		backingLit.SetPos(pos)
		stampMatValue(backingLit, t, nil)
		bound := func(n int) syntax.Expr {
			s := strconv.Itoa(n)
			b := &syntax.BasicLit{Value: s, Kind: syntax.IntLit}
			b.SetPos(pos)
			stampMatValue(b, types2.Typ[types2.UntypedInt], constant.MakeFromLiteral(s, token.INT, 0))
			return b
		}
		sl := &syntax.SliceExpr{X: backingLit, Index: [3]syntax.Expr{bound(0), bound(length), bound(cap_ - lo)}, Full: true}
		sl.SetPos(pos)
		stampMatValue(sl, t, nil)
		return sl
	case *types2.Map:
		keys, vals := v.MapEntries()
		elems := make([]syntax.Expr, len(keys))
		for i := range keys {
			kx := buildMatLiteral(keys[i], u.Key(), pos)
			if kx == nil {
				return nil
			}
			vx := buildMatLiteral(vals[i], u.Elem(), pos)
			if vx == nil {
				return nil
			}
			kv := &syntax.KeyValueExpr{Key: kx, Value: vx}
			kv.SetPos(pos)
			elems[i] = kv
		}
		lit := &syntax.CompositeLit{ElemList: elems, NKeys: len(elems), Rbrace: pos}
		lit.SetPos(pos)
		stampMatValue(lit, t, nil)
		return lit
	}
	return nil
}

func buildMatScalar(v ctfe.Value, t types2.Type, pos syntax.Pos) syntax.Expr {
	c, ok := matConstant(v)
	if !ok {
		return nil
	}
	if _, ok := types2.Unalias(t).Underlying().(*types2.Basic); !ok {
		return nil
	}
	leaf, err := v.Expr(pos)
	if err != nil {
		return nil
	}
	stampMatValue(leaf, t, c)
	if _, ok := leaf.(*syntax.BasicLit); ok {
		return leaf
	}
	if op, ok := leaf.(*syntax.Operation); ok {
		if mag, ok := op.X.(*syntax.BasicLit); ok {
			stampMatValue(mag, t, nil)
		}
		stampMatValue(op, t, c)
		return op
	}
	return nil
}
