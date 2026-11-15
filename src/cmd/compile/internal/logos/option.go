// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"reflect"
	"sort"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const (
	optTemp     = "_logos_v"
	optOk       = "_logos_ok"
	optWrapTemp = "_logos_w%d"
)

var exprType = reflect.TypeFor[syntax.Expr]()

func (fl *fileLower) lowerOptions() {
	fl.optMemo = make(map[*syntax.OptionType]syntax.Expr)
	fl.narrowDone = make(map[*syntax.Name]bool)
	fl.optBodyType = make(map[syntax.Expr]types2.Type)
	fl.collectOptReturns()
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		fl.lowerOptNode(n)
		return true
	})
	fl.rejectCompositeSurvivors()
}

func (fl *fileLower) lowerOptNode(n syntax.Node) {
	switch n := n.(type) {
	case *syntax.CallStmt:
		if fl.markOptDeferGo(n) {
			return
		}
	case *syntax.CallExpr:
		if _, ok := fl.optDeferGoPend[n]; ok {
			return
		}
	case *syntax.AssignStmt:
		fl.lowerOptAssign(n)
		fl.lowerOptFields(n, "Lhs")
		return
	case *syntax.RangeClause:
		fl.lowerOptRange(n)
		fl.lowerOptFields(n, "Lhs")
		return
	case *syntax.ReturnStmt:
		fl.lowerOptReturn(n)
	case *syntax.VarDecl:
		n.Values = fl.lowerOptVarValues(n.NameList, n.Type, n.Values)
	case *syntax.ConstDecl:
		n.Values = fl.lowerOptVarValues(n.NameList, n.Type, n.Values)
	case *syntax.SendStmt:
		fl.lowerOptSend(n)
	case *syntax.LambdaExpr:
		fl.lowerOptLambdaBody(n)
	}
	fl.lowerOptFields(n, "")
}

func (fl *fileLower) lowerOptLambdaBody(lam *syntax.LambdaExpr) {
	body, ok := lam.Body.(syntax.Expr)
	if !ok {
		return
	}
	switch syntax.Unparen(body).(type) {
	case *syntax.CoalesceExpr, *syntax.OptSelectExpr:
	case *syntax.Name, *syntax.SelectorExpr, *syntax.IndexExpr:
		if _, ok := fl.compositeRefOf(body); !ok {
			return
		}
	default:
		return
	}
	orig := fl.pkg.info.TypeOf(body)
	lowered := fl.lowerOptExpr(body)
	lam.Body = lowered
	if lowered != body && orig != nil {
		fl.optBodyType[lowered] = orig
	}
}

func (fl *fileLower) lowerOptFields(n syntax.Node, except string) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	t := v.Type()
	for i := range v.NumField() {
		if !t.Field(i).IsExported() {
			continue
		}
		if except != "" && t.Field(i).Name == except {
			continue
		}
		fl.lowerOptField(v.Field(i), t.Field(i).Type)
	}
}

func (fl *fileLower) lowerOptField(v reflect.Value, ft reflect.Type) {
	if ft == exprType {
		if v.IsNil() {
			return
		}
		v.Set(reflect.ValueOf(fl.lowerOptExpr(v.Interface().(syntax.Expr))))
		return
	}
	k := ft.Kind()
	if (k == reflect.Slice || k == reflect.Array) && ft.Elem() == exprType {
		for i := range v.Len() {
			ev := v.Index(i)
			if ev.IsNil() {
				continue
			}
			ev.Set(reflect.ValueOf(fl.lowerOptExpr(ev.Interface().(syntax.Expr))))
		}
	}
}

func (fl *fileLower) lowerOptExpr(e syntax.Expr) (out syntax.Expr) {
	defer func() {
		if out != nil && e != nil {
			fl.carryType(out, e)
		}
	}()
	switch e := e.(type) {
	case nil:
		return nil
	case *syntax.Name:
		if out, ok := fl.lowerConstMapName(e); ok {
			return out
		}
		if out, ok := fl.compositeRewriteRoot(e); ok {
			return out
		}
		return fl.narrowAccess(e)
	case *syntax.BasicLit:
		return e
	case *syntax.OptionType:
		return fl.lowerOptionType(e)
	case *syntax.CoalesceExpr:
		return fl.lowerCoalesce(e)
	case *syntax.OptSelectExpr:
		return fl.lowerOptSelect(e)
	case *syntax.CallExpr:
		if out, ok := fl.lowerConstMapLen(e); ok {
			return out
		}
		fl.compositeShapeArg(e)
		return fl.lowerOptCallExpr(e)
	case *syntax.IndexExpr:
		if out, ok := fl.compositeRewriteRoot(e); ok {
			return out
		}
		return fl.lowerMapIndex(e)
	case *syntax.SelectorExpr:
		if out, ok := fl.compositeRewriteRoot(e); ok {
			return out
		}
		if g, ok := fl.constMapUse(e); ok {
			return fl.lowerConstMapSelector(e, g)
		}
		e.X = fl.lowerOptExpr(e.X)
		return e
	case *syntax.Operation:
		if out, ok := fl.lowerNilCompare(e); ok {
			return out
		}
		e.X = fl.lowerOptExpr(e.X)
		if e.Y != nil {
			e.Y = fl.lowerOptExpr(e.Y)
		}
		return e
	case *syntax.CompositeLit:
		return fl.lowerOptComposite(e)
	case *syntax.ParenExpr:
		e.X = fl.lowerOptExpr(e.X)
		return e
	case *syntax.ListExpr:
		for i, x := range e.ElemList {
			e.ElemList[i] = fl.lowerOptExpr(x)
		}
		return e
	case *syntax.SliceExpr:
		e.X = fl.lowerOptExpr(e.X)
		for i, x := range e.Index {
			if x != nil {
				e.Index[i] = fl.lowerOptExpr(x)
			}
		}
		return e
	case *syntax.AssertExpr:
		e.X = fl.lowerOptExpr(e.X)
		if e.Type != nil {
			e.Type = fl.lowerTypeExpr(e.Type)
		}
		return e
	case *syntax.TypeSwitchGuard:
		e.X = fl.lowerOptExpr(e.X)
		return e
	case *syntax.KeyValueExpr:
		e.Key = fl.lowerOptExpr(e.Key)
		e.Value = fl.lowerOptExpr(e.Value)
		return e
	case *syntax.FuncLit:
		return e
	default:
		return e
	}
}

func (fl *fileLower) nextWrapID() int {
	fl.wrapSeq++
	return fl.wrapSeq
}

func (fl *fileLower) optType(e syntax.Expr) types2.Type {
	if e == nil {
		return nil
	}
	return fl.pkg.info.TypeOf(e)
}

func (fl *fileLower) asOptionThroughConst(t types2.Type) *types2.Option {
	if c := constAsView(t); c != nil {
		return fl.asOptionThroughConst(c.Elem)
	}
	return fl.asOption(t)
}

func (fl *fileLower) asOption(t types2.Type) *types2.Option {
	if o, ok := types2.Unalias(t).(*types2.Option); ok {
		return o
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if u := n.Underlying(); u != nil {
			if o, ok := u.(*types2.Option); ok {
				return o
			}
		}
	}
	return nil
}

func (fl *fileLower) lowerOptionType(e *syntax.OptionType) syntax.Expr {
	if prev, ok := fl.optMemo[e]; ok {
		return cloneType(prev, e.Pos())
	}
	pos := e.Pos()
	var elemT types2.Type
	if ot := fl.asOption(fl.optType(e)); ot != nil {
		elemT = ot.Elem
	} else if et := fl.optType(e.Elem); et != nil {
		elemT = et
	}
	qoff := fl.off(pos)
	var out syntax.Expr
	if elemT != nil && types2.LogosOptionIsBare(elemT) {
		out = fl.lowerTypeExpr(e.Elem)
		fl.add(qoff, qoff+1, "")
	} else {
		es := fl.optTypeHead(e.Elem)
		le := fl.lowerTypeExpr(e.Elem)
		fl.needRT = true
		fl.add(es, es, rtAlias+".Option[")
		fl.add(qoff, qoff+1, "]")
		out = at(&syntax.IndexExpr{X: selOf(pos, nameOf(pos, rtAlias), "Option"), Index: le}, pos)
		fl.deleteGroupingParens(es, qoff)
	}
	fl.optMemo[e] = out
	return out
}

func (fl *fileLower) optTypeHead(e syntax.Expr) int {
	es := fl.off(fl.optLeftmost(e))
	if _, ok := syntax.Unparen(e).(*syntax.FuncType); !ok {
		return es
	}
	if fl.funcHeadOffs == nil {
		fl.funcHeadOffs = funcHeadOffsets(fl.src)
	}
	i := sort.SearchInts(fl.funcHeadOffs, es)
	if i == 0 {
		base.Fatalf("logos: function type at offset %d has no func keyword", es)
	}
	return fl.funcHeadOffs[i-1]
}

func funcHeadOffsets(src []byte) []int {
	var out []int
	scanSegment(src, 0, len(src), func(t tok) bool {
		if t.kind == token.FUNC {
			out = append(out, t.start)
		}
		return true
	})
	return out
}

func (fl *fileLower) deleteGroupingParens(es, qoff int) {
	depth, n := 0, 0
	scanSegment(fl.src, es, qoff, func(t tok) bool {
		switch {
		case t.comment || t.auto:
		case t.kind == token.LPAREN:
			depth++
		case t.kind == token.RPAREN:
			depth--
			if depth < 0 {
				depth = 0
				if fl.parenClaims[t.start] {
					return true
				}
				fl.add(t.start, t.end, "")
				fl.parenClaims[t.start] = true
				n++
			}
		}
		return true
	})
	off := skipBlankBack(fl.src, es)
	for n > 0 && off > 0 {
		switch {
		case fl.src[off-1] == '(':
			if !fl.parenClaims[off-1] {
				fl.add(off-1, off, "")
				fl.parenClaims[off-1] = true
				n--
			}
			off = skipBlankBack(fl.src, off-1)
		case off >= len("const") && string(fl.src[off-len("const"):off]) == "const":
			off = skipBlankBack(fl.src, off-len("const"))
		default:
			return
		}
	}
}

func (fl *fileLower) optLeftmost(e syntax.Expr) syntax.Pos {
	for {
		switch n := syntax.Unparen(e).(type) {
		case *syntax.CompositeLit:
			if n.Type == nil {
				return e.Pos()
			}
			e = n.Type
		case *syntax.OptSelectExpr:
			e = n.X
		case *syntax.CoalesceExpr:
			e = n.X
		case *syntax.OptionType:
			e = n.Elem
		case *syntax.SelectorExpr:
			e = n.X
		case *syntax.IndexExpr:
			e = n.X
		case *syntax.SliceExpr:
			e = n.X
		case *syntax.AssertExpr:
			e = n.X
		case *syntax.CallExpr:
			e = n.Fun
		case *syntax.Operation:
			if n.Y == nil {
				return e.Pos()
			}
			e = n.X
		default:
			return e.Pos()
		}
	}
}

func (fl *fileLower) optHeadStart(e syntax.Expr) syntax.Pos {
	for {
		switch n := e.(type) {
		case *syntax.ParenExpr:
			return e.Pos()
		case *syntax.CompositeLit:
			if n.Type == nil {
				return e.Pos()
			}
			e = n.Type
		case *syntax.OptSelectExpr:
			e = n.X
		case *syntax.CoalesceExpr:
			e = n.X
		case *syntax.OptionType:
			e = n.Elem
		case *syntax.SelectorExpr:
			e = n.X
		case *syntax.IndexExpr:
			e = n.X
		case *syntax.SliceExpr:
			e = n.X
		case *syntax.AssertExpr:
			e = n.X
		case *syntax.CallExpr:
			e = n.Fun
		case *syntax.Operation:
			if n.Y == nil {
				return e.Pos()
			}
			e = n.X
		default:
			return e.Pos()
		}
	}
}

func (fl *fileLower) lowerTypeExpr(e syntax.Expr) syntax.Expr {
	switch t := e.(type) {
	case *syntax.OptionType:
		return fl.lowerOptionType(t)
	case *syntax.ParenExpr:
		t.X = fl.lowerTypeExpr(t.X)
		return e
	case *syntax.Operation:
		if t.Op == syntax.Mul && t.Y == nil {
			t.X = fl.lowerTypeExpr(t.X)
		}
		return e
	case *syntax.SliceType:
		t.Elem = fl.lowerTypeExpr(t.Elem)
		return e
	case *syntax.ArrayType:
		t.Elem = fl.lowerTypeExpr(t.Elem)
		return e
	case *syntax.DotsType:
		t.Elem = fl.lowerTypeExpr(t.Elem)
		return e
	case *syntax.MapType:
		t.Key = fl.lowerTypeExpr(t.Key)
		t.Value = fl.lowerTypeExpr(t.Value)
		return e
	case *syntax.ChanType:
		t.Elem = fl.lowerTypeExpr(t.Elem)
		return e
	case *syntax.FuncType:
		for _, f := range t.ParamList {
			if f.Type != nil {
				f.Type = fl.lowerTypeExpr(f.Type)
			}
		}
		for _, f := range t.ResultList {
			if f.Type != nil {
				f.Type = fl.lowerTypeExpr(f.Type)
			}
		}
		return e
	case *syntax.StructType:
		for _, f := range t.FieldList {
			if f.Type != nil {
				f.Type = fl.lowerTypeExpr(f.Type)
			}
		}
		return e
	case *syntax.InterfaceType:
		for _, f := range t.MethodList {
			if f.Type != nil {
				f.Type = fl.lowerTypeExpr(f.Type)
			}
		}
		return e
	default:
		return e
	}
}

func (fl *fileLower) spellLoweredBoth(t types2.Type, pos syntax.Pos) (native syntax.Expr, eject string, ok bool) {
	if ownedTypeBridge != nil {
		if n, e, ok := ownedTypeBridge(fl, t, pos); ok {
			return n, e, true
		}
	}
	s, ok := fl.spellLowered(t)
	if !ok {
		return nil, "", false
	}
	n, ok := typeExprAt(s, pos)
	if !ok {
		return nil, "", false
	}
	return n, s, true
}

func (fl *fileLower) optSelectName(e *syntax.OptSelectExpr) string {
	if o := fl.coeType(e.X); o != nil {
		return tupleSelectionName(o.Elem, e.Sel.Value)
	}
	return e.Sel.Value
}

func (fl *fileLower) selectAccessBare(recv syntax.Expr, e *syntax.OptSelectExpr) (syntax.Expr, string) {
	pos := e.Pos()
	sel := fl.optSelectName(e)
	return fl.selectAccess(e, selOf(pos, recv, sel), "."+sel)
}

func (fl *fileLower) selectAccessPresent(recv syntax.Expr, e *syntax.OptSelectExpr) (syntax.Expr, string) {
	pos := e.Pos()
	sel := fl.optSelectName(e)
	return fl.selectAccess(e, selOf(pos, selOf(pos, recv, "Value"), sel), ".Value."+sel)
}

func (fl *fileLower) selectAccess(e *syntax.OptSelectExpr, native syntax.Expr, suffix string) (syntax.Expr, string) {
	pos := e.Pos()
	ot := fl.coeType(e.X)
	if ot == nil {
		return native, suffix
	}
	obj, _, _ := types2.LookupFieldOrMethod(ot.Elem, true, fl.pkg.tpkg, e.Sel.Value)
	if obj == nil {
		return native, suffix
	}
	if pkg := obj.Pkg(); pkg != nil && pkg != fl.pkg.tpkg && !obj.Exported() {
		if ownedFieldBridge != nil {
			if nacc, esuf, ok := ownedFieldBridge(fl, e, native, suffix); ok {
				return nacc, esuf
			}
		}
		base.Fatalf("logos: cannot access unexported selection %s of %s at %s without a registered private view", e.Sel.Value, ot.Elem, pos.String())
	}
	return native, suffix
}

func (fl *fileLower) spellLowered(t types2.Type) (string, bool) {
	if o := fl.asOption(t); o != nil {
		if types2.LogosOptionIsBare(o.Elem) {
			return fl.spellLowered(o.Elem)
		}
		inner, ok := fl.spellLowered(o.Elem)
		if !ok {
			return "", false
		}
		fl.needRT = true
		return rtAlias + ".Option[" + inner + "]", true
	}
	if t == nil {
		return "", false
	}
	switch c := types2.Unalias(t).(type) {
	case *types2.ConstView:
		return fl.spellLowered(c.Elem)
	case *types2.ReadonlyDescendants:
		return fl.spellLowered(c.Elem)
	case *types2.Named, *types2.Alias, *types2.Basic, *types2.TypeParam:
		return fl.spellType(t)
	case *types2.Pointer:
		e, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		return "*" + e, true
	case *types2.Slice:
		e, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		return "[]" + e, true
	case *types2.Array:
		e, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		return fmt.Sprintf("[%d]", c.Len()) + e, true
	case *types2.Map:
		k, ok := fl.spellLowered(c.Key())
		if !ok {
			return "", false
		}
		v, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		return "map[" + k + "]" + v, true
	case *types2.Chan:
		e, ok := fl.spellLowered(c.Elem())
		if !ok {
			return "", false
		}
		switch c.Dir() {
		case types2.SendOnly:
			return "chan<- " + e, true
		case types2.RecvOnly:
			return "<-chan " + e, true
		default:
			return "chan " + e, true
		}
	case *types2.Signature:
		return fl.spellLoweredSig(c)
	case *types2.LogosTuple:
		if st, ok := c.Underlying().(*types2.Struct); ok {
			return fl.spellLoweredStruct(st)
		}
		return "", false
	case *types2.Struct:
		return fl.spellLoweredStruct(c)
	case *types2.Interface:
		return fl.spellLoweredInterface(c)
	default:
		return "", false
	}
}

func (fl *fileLower) spellLoweredSig(sig *types2.Signature) (string, bool) {
	var b strings.Builder
	b.WriteString("func(")
	for i := range sig.Params().Len() {
		if i > 0 {
			b.WriteString(", ")
		}
		p, ok := fl.spellLowered(sig.Params().At(i).Type())
		if !ok {
			return "", false
		}
		b.WriteString(p)
	}
	b.WriteString(")")
	switch n := sig.Results().Len(); n {
	case 0:
	case 1:
		r, ok := fl.spellLowered(sig.Results().At(0).Type())
		if !ok {
			return "", false
		}
		b.WriteByte(' ')
		b.WriteString(r)
	default:
		b.WriteString(" (")
		for i := range n {
			if i > 0 {
				b.WriteString(", ")
			}
			r, ok := fl.spellLowered(sig.Results().At(i).Type())
			if !ok {
				return "", false
			}
			b.WriteString(r)
		}
		b.WriteString(")")
	}
	return b.String(), true
}

func (fl *fileLower) spellLoweredStruct(st *types2.Struct) (string, bool) {
	var b strings.Builder
	b.WriteString("struct{ ")
	for i := range st.NumFields() {
		if i > 0 {
			b.WriteString("; ")
		}
		f := st.Field(i)
		if !f.Exported() && f.Pkg() != fl.pkg.tpkg {
			return "", false
		}
		if f.Embedded() {
			t, ok := fl.spellLowered(f.Type())
			if !ok {
				return "", false
			}
			b.WriteString(t)
		} else {
			t, ok := fl.spellLowered(f.Type())
			if !ok {
				return "", false
			}
			b.WriteString(f.Name())
			b.WriteByte(' ')
			b.WriteString(t)
		}
		if tag := st.Tag(i); tag != "" {
			b.WriteString(" `")
			b.WriteString(tag)
			b.WriteByte('`')
		}
	}
	b.WriteString(" }")
	return b.String(), true
}

func (fl *fileLower) spellLoweredInterface(it *types2.Interface) (string, bool) {
	if it.NumMethods() == 0 && it.NumEmbeddeds() == 0 {
		return fl.spellType(it)
	}
	var b strings.Builder
	b.WriteString("interface{ ")
	sep := ""
	for i := range it.NumMethods() {
		m := it.Method(i)
		if !m.Exported() && m.Pkg() != fl.pkg.tpkg {
			return "", false
		}
		sig, ok := types2.Unalias(m.Type()).(*types2.Signature)
		if !ok {
			return "", false
		}
		s, ok := fl.spellLoweredSig(sig)
		if !ok {
			return "", false
		}
		b.WriteString(sep)
		b.WriteString(m.Name())
		b.WriteString(s)
		sep = "; "
	}
	for i := range it.NumEmbeddeds() {
		e, ok := fl.spellLowered(it.EmbeddedType(i))
		if !ok {
			return "", false
		}
		b.WriteString(sep)
		b.WriteString(e)
		sep = "; "
	}
	b.WriteString(" }")
	return b.String(), true
}

func isNilName(e syntax.Expr) bool {
	n, ok := syntax.Unparen(e).(*syntax.Name)
	return ok && n.Value == "nil"
}

func (fl *fileLower) narrowTarget(e *syntax.Name) *types2.Option {
	obj, ok := fl.pkg.info.Uses[e].(*types2.Var)
	if !ok {
		return nil
	}
	dOpt := fl.asOption(obj.Type())
	if dOpt == nil {
		return nil
	}
	if rec := fl.optType(e); rec == nil || !types2.Identical(rec, dOpt.Elem) {
		return nil
	}
	return dOpt
}

func (fl *fileLower) narrowAccess(e *syntax.Name) syntax.Expr {
	if fl.narrowDone[e] {
		return e
	}
	fl.narrowDone[e] = true
	dOpt := fl.narrowTarget(e)
	if dOpt == nil || types2.LogosOptionIsBare(dOpt.Elem) {
		return e
	}
	pos := e.Pos()
	off := fl.off(pos) + len(e.Value)
	acc := selOf(pos, e, "Value")
	fl.wrapOwner[acc] = fl.nextWrapID()
	fl.addOwned(fl.wrapOwner[acc], off, off, ".Value")
	return acc
}

func (fl *fileLower) coeType(e syntax.Expr) *types2.Option {
	if name, ok := syntax.Unparen(e).(*syntax.Name); ok {
		if o := fl.narrowTarget(name); o != nil {
			return o
		}
	}
	return fl.asOptionThroughConst(fl.optType(e))
}

func (fl *fileLower) lowerOptReceiver(x syntax.Expr) syntax.Expr {
	if name, ok := syntax.Unparen(x).(*syntax.Name); ok {
		if fl.narrowTarget(name) != nil {
			fl.narrowDone[name] = true
			return x
		}
	}
	return fl.lowerOptExpr(x)
}

func (fl *fileLower) needsWrap(v syntax.Expr, rec, target types2.Type) (wrap, zero bool) {
	if target == nil {
		return false, false
	}
	if _, multipleResults := rec.(*types2.Tuple); multipleResults {
		return false, false
	}
	tOpt := fl.asOption(target)
	if tOpt == nil {
		return false, false
	}
	if isNilName(v) {
		return !types2.LogosOptionIsBare(tOpt.Elem), !types2.LogosOptionIsBare(tOpt.Elem)
	}
	if rec == nil || types2.Identical(rec, target) {
		return false, false
	}
	if types2.LogosOptionIsBare(tOpt.Elem) {
		return false, false
	}
	return true, false
}

func (fl *fileLower) wrapKnown(v syntax.Expr, rec, target types2.Type) syntax.Expr {
	wrap, zero := fl.needsWrap(v, rec, target)
	if !wrap {
		return v
	}
	pos := v.Pos()
	nativeTop, ejectTop, ok := fl.spellLoweredBoth(target, pos)
	if !ok {
		base.Fatalf("logos: cannot spell type %s of the option wrap at %s", target, pos.String())
	}
	if zero {
		n := syntax.Unparen(v).(*syntax.Name)
		off := fl.off(n.Pos())
		fl.add(off, off+3, ejectTop+"{}")
		typ := nativeTop
		out := at(&syntax.CompositeLit{Type: typ, Rbrace: pos}, pos)
		fl.wrapSpan[out] = span{start: off, end: off + 3}
		fl.wrapOwner[out] = fl.nextWrapID()
		return out
	}
	nativeNodes := []syntax.Expr{nativeTop}
	ejectTexts := []string{ejectTop}
	for cur := target; ; {
		elem := fl.asOption(cur).Elem
		if types2.Identical(rec, elem) {
			break
		}
		inner := fl.asOption(elem)
		if inner == nil || types2.LogosOptionIsBare(inner.Elem) {
			break
		}
		nt, et, ok := fl.spellLoweredBoth(elem, pos)
		if !ok {
			base.Fatalf("logos: cannot spell type %s of the option wrap at %s", elem, pos.String())
		}
		nativeNodes = append(nativeNodes, nt)
		ejectTexts = append(ejectTexts, et)
		cur = elem
	}
	vs := fl.wrapSpanOf(v)
	text := fl.lazyText(vs)
	id, ok := fl.wrapOwner[v]
	if !ok {
		id = fl.nextWrapID()
	} else {
		head, tail := fl.drainOwned(vs, id)
		text = cat(head, text, tail)
	}
	fused := text
	for i := len(ejectTexts) - 1; i >= 0; i-- {
		fused = cat(gen(ejectTexts[i]+"{Value: "), fused, gen(", Present: true}"))
	}
	fl.addT(vs.start, vs.end, fused)
	native := v
	for i := len(nativeNodes) - 1; i >= 0; i-- {
		typ := nativeNodes[i]
		native = at(&syntax.CompositeLit{
			Type: typ,
			ElemList: []syntax.Expr{
				at(&syntax.KeyValueExpr{Key: nameOf(pos, "Value"), Value: native}, pos),
				at(&syntax.KeyValueExpr{Key: nameOf(pos, "Present"), Value: nameOf(pos, "true")}, pos),
			},
			NKeys:  2,
			Rbrace: pos,
		}, pos)
	}
	fl.wrapSpan[native] = vs
	fl.wrapOwner[native] = id
	return native
}

func (fl *fileLower) wrapSpanOf(v syntax.Expr) span {
	if s, ok := fl.wrapSpan[v]; ok {
		return s
	}
	from := fl.off(leftmostOf(v))
	return span{start: from, end: fl.exprEndNested(from)}
}

func (fl *fileLower) drainOwned(vs span, id int) (head, tail otext) {
	ls := fl.ownerOfInsertion(vs.start)
	if le := fl.ownerOfInsertion(vs.end); le != ls {
		head = drainOwnedAt(ls, vs.start, id)
		tail = drainOwnedAt(le, vs.end, id)
		return head, tail
	}
	var rest []splice
	for _, sp := range *ls {
		if sp.owner != id || sp.start != sp.end || (sp.start != vs.start && sp.start != vs.end) {
			rest = append(rest, sp)
			continue
		}
		if sp.start == vs.start {
			head = cat(head, otext{s: sp.text, chunks: sp.chunks})
		} else {
			tail = cat(tail, otext{s: sp.text, chunks: sp.chunks})
		}
	}
	*ls = rest
	return head, tail
}

func drainOwnedAt(list *[]splice, off, id int) otext {
	var text otext
	var rest []splice
	for _, sp := range *list {
		if sp.owner != id || sp.start != sp.end || sp.start != off {
			rest = append(rest, sp)
			continue
		}
		text = cat(text, otext{s: sp.text, chunks: sp.chunks})
	}
	*list = rest
	return text
}

func (fl *fileLower) drainBoundaries(vs span) (head, tail otext) {
	return fl.drainBoundariesSince(vs, 1)
}

func (fl *fileLower) drainBoundariesSince(vs span, since int) (head, tail otext) {
	take := func(off int) otext {
		list := fl.ownerOfInsertion(off)
		var text otext
		var rest []splice
		for _, sp := range *list {
			if sp.owner < since || sp.start != sp.end || sp.start != off {
				rest = append(rest, sp)
				continue
			}
			text = cat(text, otext{s: sp.text, chunks: sp.chunks})
		}
		*list = rest
		return text
	}
	return take(vs.start), take(vs.end)
}

func (fl *fileLower) drainBodyWrapper(body syntax.Node, text otext) otext {
	expr, ok := body.(syntax.Expr)
	if !ok {
		return text
	}
	top, ok := syntax.Unparen(expr).(*syntax.CallExpr)
	if !ok {
		return text
	}
	_, ok = fl.wrapOwner[top]
	if !ok {
		return text
	}
	vs, ok := fl.wrapSpan[top]
	if !ok {
		return text
	}
	head, tail := fl.drainBoundaries(vs)
	return cat(head, text, tail)
}

func (fl *fileLower) wrapperTail(e syntax.Expr) otext {
	top, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok {
		return otext{}
	}
	id, ok := fl.wrapOwner[top]
	if !ok {
		return otext{}
	}
	vs, ok := fl.wrapSpan[top]
	if !ok {
		return otext{}
	}
	return drainOwnedAt(fl.ownerOfInsertion(vs.end), vs.end, id)
}

func (fl *fileLower) isMapRead(ix *syntax.IndexExpr) bool {
	if m := fl.pkg.info.Types; m != nil {
		return m[ix].HasOk()
	}
	return ix.GetTypeInfo().HasOk()
}

func (fl *fileLower) lowerMapIndex(e *syntax.IndexExpr) syntax.Expr {
	if g, ok := fl.constMapReadUse(e.X); ok {
		return fl.lowerConstMapRead(e, g)
	}
	opt := fl.asOption(fl.optType(e))
	if !fl.isMapRead(e) || opt == nil || types2.LogosOptionIsBare(opt.Elem) || fl.isTypeExpr(e) {
		e.X = fl.lowerOptExpr(e.X)
		e.Index = fl.lowerOptExpr(e.Index)
		return e
	}
	id := fl.nextWrapID()
	mStart, lbrack, rbrack := fl.lookupSpans(e)
	fl.addOwned(id, mStart, mStart, rtAlias+".Lookup(")
	rawIdx := e.Index
	idxT := fl.optType(rawIdx)
	e.X = fl.lowerOptExpr(e.X)
	e.Index = fl.lowerOptExpr(e.Index)
	fl.convertIndexKey(e, rawIdx, e.Index, idxT)
	out := fl.lookupFinish(e, lbrack, rbrack, id)
	fl.wrapOwner[out] = id
	return out
}

func logosMapKey(t types2.Type) (types2.Type, bool) {
	if t == nil {
		return nil, false
	}
	mt, ok := t.Underlying().(*types2.Map)
	if !ok {
		return nil, false
	}
	return mt.Key(), true
}

func (fl *fileLower) convertIndexKey(ix *syntax.IndexExpr, raw, lowered syntax.Expr, idxT types2.Type) {
	keyT, ok := logosMapKey(fl.optType(ix.X))
	if !ok {
		return
	}
	if idxT == nil || types2.Identical(idxT, keyT) || !types2.AssignableTo(idxT, keyT) {
		return
	}
	kn, kt, ok := fl.spellLoweredBoth(keyT, raw.Pos())
	if !ok {
		base.Fatalf("logos: cannot spell map key type %s at %s", keyT, raw.Pos().String())
	}
	kStart := fl.off(leftmostOf(raw))
	if cl, ok := syntax.Unparen(raw).(*syntax.CompositeLit); ok && cl.Type != nil {
		kStart = fl.off(cl.Type.Pos())
	}
	ix.Index = callOf(raw.Pos(), kn, []syntax.Expr{lowered})
	kEnd := fl.exprEndKey(kStart)
	fl.add(kStart, kStart, kt+"(")
	fl.add(kEnd, kEnd, ")")
}

func (fl *fileLower) exprEndKey(from int) int {
	end := len(fl.src)
	depth := 0
	seen := false
	scanSegment(fl.src, from, len(fl.src), func(t tok) bool {
		if t.comment {
			return true
		}
		if t.auto {
			if depth == 0 && seen {
				end = t.start
				return false
			}
			return true
		}
		seen = true
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				end = t.start
				return false
			}
			depth--
		case token.COMMA, token.SEMICOLON, token.COLON, token.ASSIGN, token.DEFINE:
			if depth == 0 {
				end = t.start
				return false
			}
		case token.EOF:
			end = len(fl.src)
			return false
		}
		return true
	})
	return end
}

func (fl *fileLower) lookupSpans(e *syntax.IndexExpr) (mStart, lbrack, rbrack int) {
	mStart = fl.off(fl.optLeftmost(e.X))
	kStart := fl.off(leftmostOf(e.Index))
	lbrack = fl.findLbrack(mStart, kStart)
	rbrack = fl.findRbrack(kStart)
	return mStart, lbrack, rbrack
}

func (fl *fileLower) lookupFinish(e *syntax.IndexExpr, lbrack, rbrack int, id int) syntax.Expr {
	pos := e.Pos()
	fl.needRT = true
	call := callOf(pos, selOf(pos, nameOf(pos, rtAlias), "Lookup"), []syntax.Expr{e.X, e.Index})
	fl.add(lbrack, lbrack+1, ", ")
	fl.add(rbrack, rbrack+1, ")")
	mStart := fl.off(fl.optLeftmost(e.X))
	if s, ok := fl.wrapSpan[e.X]; ok {
		mStart = s.start
	}
	fl.wrapSpan[call] = span{start: mStart, end: rbrack + 1}
	return call
}

func (fl *fileLower) lowerCoalesce(e *syntax.CoalesceExpr) syntax.Expr {
	opt := fl.coeType(e.X)
	ux, isIndex := syntax.Unparen(e.X).(*syntax.IndexExpr)
	mapRead := isIndex && fl.isMapRead(ux)
	if mapRead {
		if _, ok := fl.constMapReadUse(ux.X); ok {
			mapRead = false
		}
	}
	if opt == nil && mapRead {
		opt = fl.asOption(fl.optType(ux))
	}
	if opt == nil {
		e.X = fl.lowerOptExpr(e.X)
		e.Y = fl.lowerOptExpr(e.Y)
		return e
	}
	closure := fl.newGroupClosure(e)
	pos := e.Pos()
	id := fl.nextWrapID()
	resultType, etext, ok := fl.spellLoweredBoth(opt.Elem, pos)
	if !ok {
		base.Fatalf("logos: cannot spell type %s of the ?? expression at %s", opt.Elem, pos.String())
	}
	signature := at(&syntax.FuncType{
		ParamList:  []*syntax.Field{},
		ResultList: []*syntax.Field{at(&syntax.Field{Type: resultType}, pos)},
	}, pos)
	qoff := fl.off(pos)
	yStart := fl.off(leftmostOf(e.Y))
	through := yStart
	if rbrace, ok := lastBraceOf(e.Y); ok {
		through = fl.off(rbrace) + 1
	}
	yEnd := fl.exprEndThrough(yStart, through)
	fl.add(qoff, qoff+2, "")
	if mapRead {
		mStart, lbrack, rbrack := fl.lookupSpans(ux)
		fl.deleteGroupingParens(mStart, qoff)
		xStart := fl.off(fl.optHeadStart(e.X))
		xEnd := rbrack + 1
		wy := fl.wrapKnown(fl.lowerOptExpr(e.Y), fl.optType(e.Y), opt.Elem)
		rawIdx := ux.Index
		idxT := fl.optType(rawIdx)
		ux.X = fl.lowerOptExpr(ux.X)
		ux.Index = fl.lowerOptExpr(ux.Index)
		fl.convertIndexKey(ux, rawIdx, ux.Index, idxT)
		if types2.LogosOptionIsBare(opt.Elem) {
			fl.addOwned(id, xStart, xStart, closure.head("func() "+etext, []string{etext})+" { "+closure.prefix(nil)+"if "+optTemp+", "+optOk+" := ")
			init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp), nameOf(pos, optOk)}, []syntax.Expr{ux})
			body := []syntax.Stmt{
				ifOf(pos, init, nameOf(pos, optOk), []syntax.Stmt{returnOf(pos, []syntax.Expr{nameOf(pos, optTemp)})}),
				returnOf(pos, []syntax.Expr{wy}),
			}
			lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
			fl.addOwned(id, xEnd, xEnd, "; "+optOk+" { return "+optTemp+closure.success()+" }; "+closure.prefix(body[1])+"return ")
			out := callOf(pos, lit, nil)
			fl.wrapSpan[out] = span{start: xStart, end: yEnd}
			fl.wrapOwner[out] = id
			fl.addOwned(id, yEnd, yEnd, closure.success()+" }()")
			closure.bind(out, fl.wrapSpan[out], id)
			return out
		}
		fl.addOwned(id, xStart, xStart, closure.head("func() "+etext, []string{etext})+" { "+closure.prefix(nil)+"if "+optTemp+" := "+rtAlias+".Lookup(")
		call := fl.lookupFinish(ux, lbrack, rbrack, id)
		init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{call})
		cond := selOf(pos, nameOf(pos, optTemp), "Present")
		then := []syntax.Stmt{returnOf(pos, []syntax.Expr{selOf(pos, nameOf(pos, optTemp), "Value")})}
		body := []syntax.Stmt{
			ifOf(pos, init, cond, then),
			returnOf(pos, []syntax.Expr{wy}),
		}
		lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
		fl.addOwned(id, xEnd, xEnd, "; "+optTemp+".Present { return "+optTemp+".Value"+closure.success()+" }; "+closure.prefix(body[1])+"return ")
		out := callOf(pos, lit, nil)
		fl.wrapSpan[out] = span{start: xStart, end: yEnd}
		fl.wrapOwner[out] = id
		fl.addOwned(id, yEnd, yEnd, closure.success()+" }()")
		closure.bind(out, fl.wrapSpan[out], id)
		return out
	}
	bare := types2.LogosOptionIsBare(opt.Elem)
	xStart := fl.off(fl.optHeadStart(e.X))
	openParen, closeParen := headerParens(e.X)
	fl.addOwned(id, xStart, xStart, closure.head("func() "+etext, []string{etext})+" { "+closure.prefix(nil)+"if "+optTemp+" := "+openParen)
	nx := fl.lowerOptReceiver(e.X)
	wy := fl.wrapKnown(fl.lowerOptExpr(e.Y), fl.optType(e.Y), opt.Elem)
	xEnd := lastTokenEnd(fl.src, xStart, qoff)
	init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{nx})
	var cond syntax.Expr
	var taken syntax.Expr
	var mid string
	if bare {
		cond = notNil(pos, nameOf(pos, optTemp))
		taken = nameOf(pos, optTemp)
		mid = "; " + optTemp + " != nil { return " + optTemp + closure.success() + " }; "
	} else {
		cond = selOf(pos, nameOf(pos, optTemp), "Present")
		taken = selOf(pos, nameOf(pos, optTemp), "Value")
		mid = "; " + optTemp + ".Present { return " + optTemp + ".Value" + closure.success() + " }; "
	}
	body := []syntax.Stmt{
		ifOf(pos, init, cond, []syntax.Stmt{returnOf(pos, []syntax.Expr{taken})}),
		returnOf(pos, []syntax.Expr{wy}),
	}
	lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
	fl.addOwned(id, xEnd, xEnd, closeParen+mid+closure.prefix(body[1])+"return ")
	out := callOf(pos, lit, nil)
	fl.wrapSpan[out] = span{start: xStart, end: yEnd}
	fl.wrapOwner[out] = id
	fl.addOwned(id, yEnd, yEnd, closure.success()+" }()")
	closure.bind(out, fl.wrapSpan[out], id)
	return out
}

func headerParens(e syntax.Expr) (string, string) {
	found := false
	syntax.Inspect(e, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			return false
		case *syntax.CompositeLit:
			if n.Type != nil {
				found = true
			}
		}
		return !found
	})
	if found {
		return "(", ")"
	}
	return "", ""
}

func lastBraceOf(e syntax.Expr) (syntax.Pos, bool) {
	switch n := e.(type) {
	case *syntax.CompositeLit:
		return n.Rbrace, true
	case *syntax.FuncLit:
		return n.Body.Rbrace, true
	case *syntax.SelectorExpr:
		return lastBraceOf(n.X)
	case *syntax.IndexExpr:
		return lastBraceOf(n.X)
	case *syntax.SliceExpr:
		return lastBraceOf(n.X)
	case *syntax.AssertExpr:
		return lastBraceOf(n.X)
	case *syntax.CallExpr:
		return lastBraceOf(n.Fun)
	case *syntax.SwitchExpr:
		return n.Rbrace, true
	case *syntax.LambdaExpr:
		if block, ok := n.Body.(*syntax.BlockStmt); ok {
			return block.Rbrace, true
		}
		if body, ok := n.Body.(syntax.Expr); ok {
			return lastBraceOf(body)
		}
	case *syntax.CoalesceExpr:
		if pos, ok := lastBraceOf(n.Y); ok {
			return pos, true
		}
		return lastBraceOf(n.X)
	case *syntax.OptSelectExpr:
		return lastBraceOf(n.X)
	case *syntax.Operation:
		if n.Y != nil {
			if pos, ok := lastBraceOf(n.Y); ok {
				return pos, true
			}
		}
		return lastBraceOf(n.X)
	}
	return syntax.Pos{}, false
}

func (fl *fileLower) lowerOptSelect(e *syntax.OptSelectExpr) syntax.Expr {
	pos := e.Pos()
	id := fl.nextWrapID()
	ox := fl.coeType(e.X)
	ro := fl.asOption(fl.optType(e))
	if ox == nil || ro == nil {
		e.X = fl.lowerOptExpr(e.X)
		return e
	}
	closure := fl.newGroupClosure(e)
	xStart := fl.off(fl.optHeadStart(e.X))
	nativeR, ejectR, ok := fl.spellLoweredBoth(ro, pos)
	if !ok {
		base.Fatalf("logos: cannot spell type %s of the ?. expression at %s", ro, pos.String())
	}
	rText := ejectR
	openParen, closeParen := headerParens(e.X)
	fl.addOwned(id, xStart, xStart, closure.head("func() "+ejectR, []string{ejectR})+" { "+closure.prefix(nil)+"if "+optTemp+" := "+openParen)
	nx := fl.lowerOptReceiver(e.X)
	bareRecv := types2.LogosOptionIsBare(ox.Elem)
	accessRec := ro.Elem
	obj, _, _ := types2.LookupFieldOrMethod(ox.Elem, true, fl.pkg.tpkg, fl.optSelectName(e))
	if obj == nil {
		obj, _, _ = types2.LookupFieldOrMethod(tupleUnderlying(ox.Elem).Underlying(), true, fl.pkg.tpkg, fl.optSelectName(e))
	}
	if obj != nil && fl.asOption(obj.Type()) != nil {
		accessRec = ro
	}
	recv := nameOf(pos, optTemp)
	var acc syntax.Expr
	var accSuffix string
	if bareRecv {
		acc, accSuffix = fl.selectAccessBare(recv, e)
	} else {
		acc, accSuffix = fl.selectAccessPresent(recv, e)
	}
	some, someHead, someTail := fl.wrapParts(acc, accessRec, ro, pos)
	var none syntax.Expr
	var noneT string
	if types2.LogosOptionIsBare(ro.Elem) {
		none = nilOf(pos)
		noneT = "nil"
	} else {
		noneT = rText + "{}"
		none = at(&syntax.CompositeLit{Type: nativeR, Rbrace: pos}, pos)
	}
	var cond syntax.Expr
	var condT string
	if bareRecv {
		cond = notNil(pos, nameOf(pos, optTemp))
		condT = optTemp + " != nil"
	} else {
		cond = selOf(pos, nameOf(pos, optTemp), "Present")
		condT = optTemp + ".Present"
	}
	resultType := nativeR
	signature := at(&syntax.FuncType{
		ParamList:  []*syntax.Field{},
		ResultList: []*syntax.Field{at(&syntax.Field{Type: resultType}, pos)},
	}, pos)
	init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{nx})
	body := []syntax.Stmt{
		ifOf(pos, init, cond, []syntax.Stmt{returnOf(pos, []syntax.Expr{some})}),
		returnOf(pos, []syntax.Expr{none}),
	}
	lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
	xEnd := lastTokenEnd(fl.src, xStart, fl.off(pos))
	selEnd := fl.off(e.Sel.Pos()) + len(e.Sel.Value)
	fl.add(xEnd, selEnd, closeParen+"; "+condT+" { return "+someHead+optTemp+accSuffix+someTail+closure.success()+" }; return "+noneT+closure.success()+" }()")
	out := callOf(pos, lit, nil)
	fl.wrapSpan[out] = span{start: xStart, end: selEnd}
	fl.wrapOwner[out] = id
	closure.bind(out, fl.wrapSpan[out], id)
	return out
}

func (fl *fileLower) wrapParts(acc syntax.Expr, rec, target types2.Type, pos syntax.Pos) (some syntax.Expr, head, tail string) {
	wrap, _ := fl.needsWrap(acc, rec, target)
	if !wrap {
		return acc, "", ""
	}
	typ, tText, ok := fl.spellLoweredBoth(target, pos)
	if !ok {
		base.Fatalf("logos: cannot spell type %s of the ?. value at %s", target, pos.String())
	}
	return at(&syntax.CompositeLit{
		Type: typ,
		ElemList: []syntax.Expr{
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Value"), Value: acc}, pos),
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Present"), Value: nameOf(pos, "true")}, pos),
		},
		NKeys:  2,
		Rbrace: pos,
	}, pos), tText + "{Value: ", ", Present: true}"
}

func (fl *fileLower) lowerOptCallExpr(e *syntax.CallExpr) syntax.Expr {
	if osel, ok := syntax.Unparen(e.Fun).(*syntax.OptSelectExpr); ok {
		return fl.lowerOptCall(e, osel)
	}
	e.Fun = fl.lowerOptExpr(e.Fun)
	var targets []types2.Type
	if sig := fl.callSig(e); sig != nil {
		targets = fl.callTargets(e, sig)
	}
	for i, a := range e.ArgList {
		la := fl.lowerOptExpr(a)
		if targets != nil {
			la = fl.wrapKnown(la, fl.optType(a), targets[i])
		}
		e.ArgList[i] = la
	}
	return e
}

func (fl *fileLower) callSig(e *syntax.CallExpr) *types2.Signature {
	switch fun := syntax.Unparen(e.Fun).(type) {
	case *syntax.Name:
		if obj := fl.pkg.info.Uses[fun]; obj != nil {
			if fn, ok := obj.(*types2.Func); ok {
				if sig, ok := types2.Unalias(fn.Type()).(*types2.Signature); ok {
					return sig
				}
			}
		}
		return nil
	case *syntax.SelectorExpr:
		if sel := fl.pkg.info.Selections[fun]; sel != nil {
			if fn, ok := sel.Obj().(*types2.Func); ok {
				if sig, ok := types2.Unalias(fn.Type()).(*types2.Signature); ok {
					return sig
				}
			}
		}
		if sig, ok := types2.Unalias(fl.optType(fun)).(*types2.Signature); ok {
			return sig
		}
		return nil
	default:
		if sig, ok := types2.Unalias(fl.optType(fun)).(*types2.Signature); ok {
			return sig
		}
		return nil
	}
}

func (fl *fileLower) callTargets(e *syntax.CallExpr, sig *types2.Signature) []types2.Type {
	params := sig.Params()
	n := len(e.ArgList)
	targets := make([]types2.Type, n)
	shift := 0
	if fun, ok := syntax.Unparen(e.Fun).(*syntax.SelectorExpr); ok {
		if fl.isTypeExpr(fun.X) {
			if recv := sig.Recv(); recv != nil && n > 0 {
				targets[0] = recv.Type()
				shift = 1
			}
		}
	}
	for i := range targets {
		targets[i] = fl.paramTarget(params, sig.Variadic(), e.HasDots, n, i-shift)
	}
	return targets
}

func (fl *fileLower) isTypeExpr(e syntax.Expr) bool {
	if tv, ok := fl.pkg.info.Types[e]; ok {
		return tv.IsType()
	}
	return e.GetTypeInfo().IsType()
}

func (fl *fileLower) paramTarget(params *types2.Tuple, variadic, dots bool, nargs, i int) types2.Type {
	if i < 0 {
		return nil
	}
	np := params.Len()
	if !variadic {
		if i < np {
			return params.At(i).Type()
		}
		return nil
	}
	if np == 0 {
		return nil
	}
	if dots {
		if i < np {
			return params.At(i).Type()
		}
		return nil
	}
	if i < np-1 {
		return params.At(i).Type()
	}
	if s, ok := types2.Unalias(params.At(np - 1).Type()).(*types2.Slice); ok {
		return s.Elem()
	}
	return nil
}

func (fl *fileLower) lowerOptCall(e *syntax.CallExpr, osel *syntax.OptSelectExpr) syntax.Expr {
	pos := osel.Pos()
	id := fl.nextWrapID()
	ox := fl.coeType(osel.X)
	if ox == nil {
		e.Fun = fl.lowerOptExpr(e.Fun)
		for i, a := range e.ArgList {
			e.ArgList[i] = fl.lowerOptExpr(a)
		}
		return e
	}
	sig := fl.optCallSig(ox, osel.Sel.Value)
	if sig == nil {
		osel.X = fl.lowerOptReceiver(osel.X)
		e.Fun = osel
		for i, a := range e.ArgList {
			e.ArgList[i] = fl.lowerOptExpr(a)
		}
		return e
	}
	closure := fl.newGroupClosure(e)
	xStart := fl.off(fl.optLeftmost(osel.X))
	rs := sig.Results()
	n := rs.Len()
	rTexts := make([]string, n)
	nrNodes := make([]syntax.Expr, n)
	roList := make([]types2.Type, n)
	recList := make([]types2.Type, n)
	for i := range n {
		r := rs.At(i).Type()
		ro := fl.resultOption(r)
		recList[i] = r
		roList[i] = ro
		nt, t, ok := fl.spellLoweredBoth(ro, pos)
		if !ok {
			base.Fatalf("logos: cannot spell type %s of the ?. call at %s", ro, pos.String())
		}
		nrNodes[i] = nt
		rTexts[i] = t
	}
	var headText string
	switch n {
	case 0:
		headText = "func()"
	case 1:
		headText = "func() " + rTexts[0]
	default:
		headText = "func() (" + strings.Join(rTexts, ", ") + ")"
	}
	fl.addOwned(id, xStart, xStart, closure.head(headText, rTexts)+" { "+closure.prefix(nil)+"if "+optTemp+" := ")
	nx := fl.lowerOptReceiver(osel.X)
	params := sig.Params()
	for i, a := range e.ArgList {
		la := fl.lowerOptExpr(a)
		e.ArgList[i] = fl.wrapKnown(la, fl.optType(a), fl.paramTarget(params, sig.Variadic(), e.HasDots, len(e.ArgList), i))
	}
	bareRecv := types2.LogosOptionIsBare(ox.Elem)
	recv := nameOf(pos, optTemp)
	var racc syntax.Expr
	var accSuffix string
	if bareRecv {
		racc, accSuffix = fl.selectAccessBare(recv, osel)
	} else {
		racc, accSuffix = fl.selectAccessPresent(recv, osel)
	}
	var cond syntax.Expr
	var condT string
	if bareRecv {
		cond = notNil(pos, nameOf(pos, optTemp))
		condT = optTemp + " != nil"
	} else {
		cond = selOf(pos, nameOf(pos, optTemp), "Present")
		condT = optTemp + ".Present"
	}
	xEnd := lastTokenEnd(fl.src, xStart, fl.off(pos))
	selEnd := fl.off(osel.Sel.Pos()) + len(osel.Sel.Value)
	lparen, rparen := fl.findCallParens(selEnd)
	argsText := fl.lazyText(span{start: lparen, end: rparen + 1})
	init := defineOf(pos, []syntax.Expr{nameOf(pos, optTemp)}, []syntax.Expr{nx})
	call := callOf(pos, racc, e.ArgList)
	switch rs.Len() {
	case 0:
		invoke := exprStmtOf(pos, call)
		lit := at(&syntax.FuncLit{
			Type: at(&syntax.FuncType{}, pos),
			Body: blockOf(pos, []syntax.Stmt{
				ifOf(pos, init, cond, []syntax.Stmt{invoke}),
			}),
		}, pos)
		fl.addT(xEnd, rparen+1, cat(gen("; "+condT+" { "+closure.prefix(invoke)+optTemp+accSuffix), argsText, gen(" }; "+closure.voidReturn()+"}()")))
		out := callOf(pos, lit, nil)
		fl.carryType(out, e)
		fl.wrapSpan[out] = span{start: xStart, end: rparen + 1}
		fl.wrapOwner[out] = id
		closure.bind(out, fl.wrapSpan[out], id)
		return out
	case 1:
		some, someHead, someTail := fl.wrapParts(call, recList[0], roList[0], pos)
		var none syntax.Expr
		var noneT string
		if fl.bareResult(roList[0]) {
			none = nilOf(pos)
			noneT = "nil"
		} else {
			noneT = rTexts[0] + "{}"
			none = at(&syntax.CompositeLit{Type: nrNodes[0], Rbrace: pos}, pos)
		}
		resultType := nrNodes[0]
		signature := at(&syntax.FuncType{
			ParamList:  []*syntax.Field{},
			ResultList: []*syntax.Field{at(&syntax.Field{Type: resultType}, pos)},
		}, pos)
		body := []syntax.Stmt{
			ifOf(pos, init, cond, []syntax.Stmt{returnOf(pos, []syntax.Expr{some})}),
			returnOf(pos, []syntax.Expr{none}),
		}
		lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
		fl.addT(xEnd, rparen+1, cat(
			gen("; "+condT+" { "+closure.prefix(body[0].(*syntax.IfStmt).Then.List[0])+"return "+someHead+optTemp+accSuffix),
			argsText,
			gen(someTail+closure.success()+" }; return "+noneT+closure.success()+" }()"),
		))
		out := callOf(pos, lit, nil)
		fl.carryType(out, e)
		fl.wrapSpan[out] = span{start: xStart, end: rparen + 1}
		fl.wrapOwner[out] = id
		closure.bind(out, fl.wrapSpan[out], id)
		return out
	default:
		fields := make([]*syntax.Field, n)
		for i := range n {
			fields[i] = at(&syntax.Field{Type: nrNodes[i]}, pos)
		}
		signature := at(&syntax.FuncType{
			ParamList:  []*syntax.Field{},
			ResultList: fields,
		}, pos)
		tmps := make([]syntax.Expr, n)
		rets := make([]syntax.Expr, n)
		noneNodes := make([]syntax.Expr, n)
		noneList := make([]string, n)
		var headNames, retWraps []string
		for i := range n {
			tmp := fmt.Sprintf(optWrapTemp, i)
			tmps[i] = nameOf(pos, tmp)
			some, sh, st := fl.wrapParts(tmps[i], recList[i], roList[i], pos)
			rets[i] = some
			headNames = append(headNames, tmp)
			retWraps = append(retWraps, sh+tmp+st)
			if fl.bareResult(roList[i]) {
				noneNodes[i] = nilOf(pos)
				noneList[i] = "nil"
			} else {
				noneNodes[i] = at(&syntax.CompositeLit{Type: nrNodes[i], Rbrace: pos}, pos)
				noneList[i] = rTexts[i] + "{}"
			}
		}
		then := []syntax.Stmt{
			defineOf(pos, tmps, []syntax.Expr{call}),
			returnOf(pos, rets),
		}
		body := []syntax.Stmt{
			ifOf(pos, init, cond, then),
			returnOf(pos, noneNodes),
		}
		lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, body)}, pos)
		fl.addT(xEnd, rparen+1, cat(
			gen("; "+condT+" { "+closure.prefix(then[0])+strings.Join(headNames, ", ")+" := "+optTemp+accSuffix),
			argsText,
			gen("; return "+strings.Join(retWraps, ", ")+closure.success()+" }; return "+strings.Join(noneList, ", ")+closure.success()+" }()"),
		))
		out := callOf(pos, lit, nil)
		fl.carryType(out, e)
		fl.wrapSpan[out] = span{start: xStart, end: rparen + 1}
		fl.wrapOwner[out] = id
		closure.bind(out, fl.wrapSpan[out], id)
		return out
	}
}

type optDeferGoWork struct {
	osel *syntax.OptSelectExpr
	ox   *types2.Option
	nx   syntax.Expr
	lv   string
	rpos syntax.Pos
}

func (fl *fileLower) markOptDeferGo(s *syntax.CallStmt) bool {
	e, ok := syntax.Unparen(s.Call).(*syntax.CallExpr)
	if !ok {
		return false
	}
	osel, ok := syntax.Unparen(e.Fun).(*syntax.OptSelectExpr)
	if !ok {
		return false
	}
	ox := fl.coeType(osel.X)
	if ox == nil {
		return false
	}
	sig := fl.optCallSig(ox, osel.Sel.Value)
	if sig == nil {
		return false
	}
	lv := fmt.Sprintf("_logos_d%d", fl.nextWrapID())
	rpos := fl.optLeftmost(osel.X)
	osel.X = fl.lowerOptReceiver(osel.X)
	for i, a := range e.ArgList {
		target := fl.paramTarget(sig.Params(), sig.Variadic(), e.HasDots, len(e.ArgList), i)
		e.ArgList[i] = fl.wrapKnown(fl.lowerOptExpr(a), fl.optType(a), target)
	}
	if fl.optDeferGoPend == nil {
		fl.optDeferGoPend = make(map[*syntax.CallExpr]optDeferGoWork)
	}
	fl.optDeferGoPend[e] = optDeferGoWork{osel: osel, ox: ox, nx: osel.X, lv: lv, rpos: rpos}
	return true
}

func (fl *fileLower) optCallSig(ox *types2.Option, name string) *types2.Signature {
	obj, _, _ := types2.LookupFieldOrMethod(ox.Elem, true, fl.pkg.tpkg, name)
	if obj == nil {
		return nil
	}
	sig, _ := obj.Type().Underlying().(*types2.Signature)
	return sig
}

func (fl *fileLower) lowerOptDeferGo(s *syntax.CallStmt, lowerCall func(syntax.Stmt) []syntax.Stmt) ([]syntax.Stmt, bool) {
	return fl.lowerOptDeferGoWith(s, lowerCall, fl.moveOptReceiver)
}

func (fl *fileLower) lowerOptDeferGoCopied(s *syntax.CallStmt) ([]syntax.Stmt, bool) {
	keep := func(c syntax.Stmt) []syntax.Stmt { return []syntax.Stmt{c} }
	return fl.lowerOptDeferGoWith(s, keep, fl.copyOptReceiver)
}

func (fl *fileLower) moveOptReceiver(r span, stub string) otext {
	return gen(fl.moveRegion(r, stub))
}

func (fl *fileLower) copyOptReceiver(r span, stub string) otext {
	text := fl.textOf(r)
	fl.add(r.start, r.end, stub)
	return text
}

func (fl *fileLower) lowerOptDeferGoWith(s *syntax.CallStmt, lowerCall func(syntax.Stmt) []syntax.Stmt, captureRecv func(span, string) otext) ([]syntax.Stmt, bool) {
	e, ok := syntax.Unparen(s.Call).(*syntax.CallExpr)
	if !ok {
		return nil, false
	}
	w, ok := fl.optDeferGoPend[e]
	if !ok {
		return nil, false
	}
	delete(fl.optDeferGoPend, e)
	pos := w.osel.Pos()
	rpos := w.rpos
	recvName := nameOf(rpos, w.lv)
	var cond, callee syntax.Expr
	var condT, accSuffix string
	if types2.LogosOptionIsBare(w.ox.Elem) {
		cond, condT = notNil(pos, nameOf(pos, w.lv)), w.lv+" != nil"
		callee, accSuffix = fl.selectAccessBare(recvName, w.osel)
	} else {
		cond, condT = selOf(pos, nameOf(pos, w.lv), "Present"), w.lv+".Present"
		callee, accSuffix = fl.selectAccessPresent(recvName, w.osel)
	}
	xStart := fl.off(rpos)
	xEnd := lastTokenEnd(fl.src, xStart, fl.off(pos))
	selEnd := fl.off(w.osel.Sel.Pos()) + len(w.osel.Sel.Value)
	_, rparen := fl.findCallParens(selEnd)
	recv := span{start: xStart, end: xEnd}
	if vs, ok := fl.wrapSpan[w.nx]; ok {
		recv = vs
	}
	var head, tail otext
	if id, ok := fl.wrapOwner[w.nx]; ok {
		head, tail = fl.drainOwned(recv, id)
	}
	captured := captureRecv(recv, w.lv)
	fl.add(xEnd, selEnd, accSuffix)
	stmtStart := fl.off(s.Pos())
	fl.addT(stmtStart, stmtStart, cat(gen(w.lv+" := "), head, captured, tail, gen("; if "+condT+" { ")))
	define := defineOf(pos, []syntax.Expr{nameOf(pos, w.lv)}, []syntax.Expr{w.nx})
	e.Fun = callee
	guarded := lowerCall(s)
	fl.add(rparen+1, rparen+1, " }")
	scanForLiterals(fl, define)
	return []syntax.Stmt{define, ifOf(s.Pos(), nil, cond, guarded)}, true
}

func (fl *fileLower) lowerOptDeferGoIn(block *syntax.BlockStmt) {
	lower := func(list []syntax.Stmt) []syntax.Stmt {
		var out []syntax.Stmt
		for _, st := range list {
			if cs, ok := st.(*syntax.CallStmt); ok {
				if lowered, ok := fl.lowerOptDeferGoCopied(cs); ok {
					out = append(out, lowered...)
					continue
				}
			}
			out = append(out, st)
		}
		return out
	}
	syntax.Inspect(block, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		case *syntax.BlockStmt:
			n.List = lower(n.List)
		case *syntax.CaseClause:
			n.Body = lower(n.Body)
		case *syntax.CommClause:
			n.Body = lower(n.Body)
		}
		return true
	})
}

func (fl *fileLower) resultOption(r types2.Type) types2.Type {
	if fl.asOption(r) != nil {
		return r
	}
	return &types2.Option{Elem: r}
}

func (fl *fileLower) bareResult(ro types2.Type) bool {
	return types2.LogosOptionIsBare(fl.asOption(ro).Elem)
}

func (fl *fileLower) lowerNilCompare(e *syntax.Operation) (syntax.Expr, bool) {
	if e.Op != syntax.Eql && e.Op != syntax.Neq {
		return nil, false
	}
	if e.X == nil || e.Y == nil {
		return nil, false
	}
	lx, rx := syntax.Unparen(e.X), syntax.Unparen(e.Y)
	lnil := isNilName(lx)
	rnil := isNilName(rx)
	if lnil == rnil {
		return nil, false
	}
	var xside, nilside syntax.Expr
	if rnil {
		xside, nilside = e.X, rx
	} else {
		xside, nilside = e.Y, lx
	}
	ox := fl.asOption(fl.optType(xside))
	if ox == nil || types2.LogosOptionIsBare(ox.Elem) {
		return nil, false
	}
	pos := e.Pos()
	present := selOf(pos, xside, "Present")
	var out syntax.Expr = present
	if e.Op == syntax.Eql {
		out = at(&syntax.Operation{Op: syntax.Not, X: present}, pos)
	}
	bang := ""
	if e.Op == syntax.Eql {
		bang = "!"
	}
	xs := fl.off(syntax.StartPos(xside))
	if rnil {
		op := fl.findCmpOp(xs, fl.off(leftmostOf(nilside)))
		fl.add(xs, xs, bang)
		fl.add(op, fl.nilEnd(op, fl.off(syntax.EndPos(nilside))), ".Present")
	} else {
		fl.add(fl.off(e.X.Pos()), xs, bang)
		fl.add(fl.operandEnd(xs), fl.operandEnd(xs), ".Present")
	}
	return out, true
}

func (fl *fileLower) nilEnd(op, inner int) int {
	end := inner
	depth := 0
	scanSegment(fl.src, op, end, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind == token.LPAREN {
			depth++
		}
		return true
	})
	for range depth {
		matched := false
		scanSegment(fl.src, end, len(fl.src), func(t tok) bool {
			if t.comment || t.auto {
				return true
			}
			if t.kind == token.RPAREN {
				end = t.span.end
				matched = true
			}
			return false
		})
		if !matched {
			base.Fatalf("logos: parenthesized nil at offset %d has no )", op)
		}
	}
	return end
}

func (fl *fileLower) findCmpOp(from, to int) int {
	op := -1
	depth := 0
	scanSegment(fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth > 0 {
				depth--
			}
		case token.EQL, token.NEQ:
			if depth == 0 {
				op = t.start
				return false
			}
		}
		return true
	})
	if op < 0 {
		base.Fatalf("logos: nil comparison at offset %d has no == or !=", from)
	}
	return op
}

func unlist(e syntax.Expr) ([]syntax.Expr, bool) {
	if l, ok := syntax.Unparen(e).(*syntax.ListExpr); ok {
		return l.ElemList, true
	}
	return []syntax.Expr{e}, false
}

func (fl *fileLower) lowerOptAssign(s *syntax.AssignStmt) {
	if s.Op == syntax.Def {
		fl.lowerOptLHS(s.Lhs)
		if s.Rhs != nil {
			s.Rhs = fl.lowerOptExpr(s.Rhs)
		}
		return
	}
	if s.Rhs == nil || s.Op != 0 {
		if name, ok := syntax.Unparen(s.Lhs).(*syntax.Name); ok {
			s.Lhs = fl.narrowAccess(name)
		} else {
			fl.lowerOptLHS(s.Lhs)
		}
		if s.Rhs != nil {
			s.Rhs = fl.lowerOptExpr(s.Rhs)
		}
		return
	}
	lhs, _ := unlist(s.Lhs)
	rhs, rhsList := unlist(s.Rhs)
	if len(lhs) > 1 && len(rhs) == 1 {
		fl.lowerOptLHS(s.Lhs)
		s.Rhs = fl.lowerOptExpr(s.Rhs)
		return
	}
	if len(lhs) == len(rhs) {
		for i := range lhs {
			nr := fl.lowerOptExpr(rhs[i])
			nr = fl.wrapKnown(nr, fl.optType(rhs[i]), fl.targetOfLHS(lhs[i]))
			if rhsList {
				s.Rhs.(*syntax.ListExpr).ElemList[i] = nr
			} else {
				s.Rhs = nr
			}
		}
		fl.lowerOptLHS(s.Lhs)
		return
	}
	fl.lowerOptLHS(s.Lhs)
	s.Rhs = fl.lowerOptExpr(s.Rhs)
}

func (fl *fileLower) lowerOptLHS(e syntax.Expr) {
	switch e := e.(type) {
	case *syntax.Name:
	case *syntax.ListExpr:
		for _, x := range e.ElemList {
			fl.lowerOptLHS(x)
		}
	case *syntax.ParenExpr:
		fl.lowerOptLHS(e.X)
	case *syntax.IndexExpr:
		e.X = fl.lowerOptExpr(e.X)
		e.Index = fl.lowerOptExpr(e.Index)
	case *syntax.SelectorExpr:
		e.X = fl.lowerOptExpr(e.X)
	case *syntax.Operation:
		if e.Op == syntax.Mul && e.Y == nil {
			e.X = fl.lowerOptExpr(e.X)
		}
	}
}

func (fl *fileLower) targetOfLHS(e syntax.Expr) types2.Type {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Name:
		if obj, ok := fl.pkg.info.Uses[e].(*types2.Var); ok {
			return obj.Type()
		}
		return fl.optType(e)
	case *syntax.SelectorExpr:
		if sel := fl.pkg.info.Selections[e]; sel != nil {
			return sel.Obj().Type()
		}
		return fl.optType(e)
	case *syntax.IndexExpr:
		switch c := types2.Unalias(fl.optType(e.X)).(type) {
		case *types2.Slice:
			return c.Elem()
		case *types2.Array:
			return c.Elem()
		case *types2.Map:
			return c.Elem()
		}
		return nil
	case *syntax.Operation:
		if e.Op == syntax.Mul && e.Y == nil {
			if p, ok := types2.Unalias(fl.optType(e.X)).(*types2.Pointer); ok {
				return p.Elem()
			}
		}
		return nil
	}
	return nil
}

func (fl *fileLower) collectOptReturns() {
	fl.retResults = make(map[*syntax.ReturnStmt][]types2.Type)
	type fb struct {
		body    *syntax.BlockStmt
		results []types2.Type
	}
	var list []fb
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncDecl:
			if n.Body != nil {
				list = append(list, fb{n.Body, fl.funcDeclResults(n)})
			}
		case *syntax.FuncLit:
			if sig, ok := types2.Unalias(fl.optType(n)).(*types2.Signature); ok {
				list = append(list, fb{n.Body, tupleTypes(sig.Results())})
			}
		}
		return true
	})
	for _, f := range list {
		if f.body == nil {
			continue
		}
		results := f.results
		syntax.Inspect(f.body, func(n syntax.Node) bool {
			if _, ok := n.(*syntax.FuncLit); ok {
				return false
			}
			if rs, ok := n.(*syntax.ReturnStmt); ok {
				fl.retResults[rs] = results
			}
			return true
		})
	}
	for rs, typ := range fl.armResult {
		fl.retResults[rs] = []types2.Type{typ}
	}
}

func (fl *fileLower) funcDeclResults(fn *syntax.FuncDecl) []types2.Type {
	if obj := fl.pkg.info.Defs[fn.Name]; obj != nil {
		if f, ok := obj.(*types2.Func); ok {
			if sig, ok := types2.Unalias(f.Type()).(*types2.Signature); ok {
				return tupleTypes(sig.Results())
			}
		}
	}
	return nil
}

func tupleTypes(t *types2.Tuple) []types2.Type {
	if t == nil {
		return nil
	}
	out := make([]types2.Type, t.Len())
	for i := range t.Len() {
		out[i] = t.At(i).Type()
	}
	return out
}

func (fl *fileLower) lowerOptReturn(s *syntax.ReturnStmt) {
	if s.Results == nil {
		return
	}
	orig := s.Results
	low := fl.lowerOptExpr(orig)
	s.Results = low
	res := fl.retResults[s]
	if len(res) == 0 {
		return
	}
	ll, lowList := unlist(low)
	ol, _ := unlist(orig)
	if len(ll) > 1 && len(ol) == 1 {
		return
	}
	if lowList && len(ll) == len(res) && len(ol) == len(res) {
		for i := range ll {
			ll[i] = fl.wrapKnown(ll[i], fl.optType(ol[i]), res[i])
		}
		return
	}
	if len(ll) == 1 && len(res) == 1 && len(ol) == 1 {
		s.Results = fl.wrapKnown(low, fl.optType(orig), res[0])
	}
}

func (fl *fileLower) lowerOptVarValues(names []*syntax.Name, typ, vals syntax.Expr) syntax.Expr {
	if vals == nil || typ == nil {
		return vals
	}
	vl, isList := unlist(vals)
	if len(vl) != len(names) {
		return fl.lowerOptExpr(vals)
	}
	targets := make([]types2.Type, len(names))
	for i, name := range names {
		if obj := fl.pkg.info.Defs[name]; obj != nil {
			switch obj.(type) {
			case *types2.Var, *types2.Const:
				targets[i] = obj.Type()
			}
		}
	}
	for i := range vl {
		nv := fl.lowerOptExpr(vl[i])
		nv = fl.wrapKnown(nv, fl.optType(vl[i]), targets[i])
		if isList {
			vals.(*syntax.ListExpr).ElemList[i] = nv
		} else {
			vals = nv
		}
	}
	return vals
}

func (fl *fileLower) lowerOptSend(s *syntax.SendStmt) {
	s.Chan = fl.lowerOptExpr(s.Chan)
	var elem types2.Type
	if ch, ok := types2.Unalias(fl.optType(s.Chan)).(*types2.Chan); ok {
		elem = ch.Elem()
	}
	s.Value = fl.wrapKnown(fl.lowerOptExpr(s.Value), fl.optType(s.Value), elem)
}

func structFieldType(st *types2.Struct, key syntax.Expr) types2.Type {
	n, ok := syntax.Unparen(key).(*syntax.Name)
	if !ok {
		return nil
	}
	for i := range st.NumFields() {
		if st.Field(i).Name() == n.Value {
			return st.Field(i).Type()
		}
	}
	return nil
}

func (fl *fileLower) lowerOptComposite(e *syntax.CompositeLit) syntax.Expr {
	t := fl.optType(e)
	if t == nil {
		for i, x := range e.ElemList {
			e.ElemList[i] = fl.lowerOptExpr(x)
		}
		return e
	}
	u := types2.Unalias(t)
	if p, ok := u.(*types2.Pointer); ok {
		u = types2.Unalias(p.Elem())
	}
	if n, ok := u.(*types2.Named); ok {
		u = n.Underlying()
	}
	switch c := u.(type) {
	case *types2.Struct:
		for i, x := range e.ElemList {
			if kv, ok := syntax.Unparen(x).(*syntax.KeyValueExpr); ok {
				if _, isName := syntax.Unparen(kv.Key).(*syntax.Name); isName {
					nv := fl.lowerOptExpr(kv.Value)
					kv.Value = fl.wrapKnown(nv, fl.optType(kv.Value), structFieldType(c, kv.Key))
					continue
				}
			}
			nv := fl.lowerOptExpr(x)
			var ft types2.Type
			if i < c.NumFields() {
				ft = c.Field(i).Type()
			}
			e.ElemList[i] = fl.wrapKnown(nv, fl.optType(x), ft)
		}
	case *types2.Map:
		for i, x := range e.ElemList {
			if kv, ok := syntax.Unparen(x).(*syntax.KeyValueExpr); ok {
				nk := fl.lowerOptExpr(kv.Key)
				nv := fl.lowerOptExpr(kv.Value)
				kv.Key = fl.wrapKnown(nk, fl.optType(kv.Key), c.Key())
				kv.Value = fl.wrapKnown(nv, fl.optType(kv.Value), c.Elem())
				continue
			}
			e.ElemList[i] = fl.lowerOptExpr(x)
		}
	case *types2.Slice, *types2.Array:
		var elem types2.Type
		switch c := c.(type) {
		case *types2.Slice:
			elem = c.Elem()
		case *types2.Array:
			elem = c.Elem()
		}
		for i, x := range e.ElemList {
			nv := fl.lowerOptExpr(x)
			e.ElemList[i] = fl.wrapKnown(nv, fl.optType(x), elem)
		}
	default:
		for i, x := range e.ElemList {
			e.ElemList[i] = fl.lowerOptExpr(x)
		}
	}
	return e
}

func (fl *fileLower) lowerOptRange(s *syntax.RangeClause) {
	s.X = fl.lowerOptExpr(s.X)
	if s.Lhs != nil {
		fl.lowerOptLHS(s.Lhs)
	}
}

func (fl *fileLower) findCallParens(selEnd int) (lparen, rparen int) {
	lparen = -1
	scanSegment(fl.src, selEnd, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind != token.LPAREN {
			base.Fatalf("logos: optional call at offset %d has no (", selEnd)
		}
		lparen = t.start
		return false
	})
	if lparen < 0 {
		base.Fatalf("logos: optional call at offset %d has no (", selEnd)
	}
	depth := 0
	scanSegment(fl.src, lparen, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
			if depth == 0 {
				rparen = t.start
				return false
			}
		case token.SEMICOLON:
			if depth == 0 {
				base.Fatalf("logos: optional call at offset %d has no )", lparen)
			}
		}
		return true
	})
	if rparen <= lparen {
		base.Fatalf("logos: optional call at offset %d has no )", lparen)
	}
	return lparen, rparen
}

func (fl *fileLower) findLbrack(from, kStart int) int {
	lbrack := -1
	scanSegment(fl.src, from, kStart, func(t tok) bool {
		if !t.comment && t.kind == token.LBRACK {
			lbrack = t.start
		}
		return true
	})
	if lbrack < 0 {
		base.Fatalf("logos: index expression at offset %d has no [", from)
	}
	return lbrack
}

func (fl *fileLower) findRbrack(kStart int) int {
	rbrack := -1
	depth := 0
	scanSegment(fl.src, kStart, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				if t.kind != token.RBRACK {
					base.Fatalf("logos: index expression at offset %d ends in %s", kStart, fl.src[t.start:t.end])
				}
				rbrack = t.start
				return false
			}
			depth--
		case token.SEMICOLON, token.EOF:
			base.Fatalf("logos: index expression at offset %d has no ]", kStart)
		}
		return true
	})
	if rbrack < 0 {
		base.Fatalf("logos: index expression at offset %d has no ]", kStart)
	}
	return rbrack
}

func (fl *fileLower) exprEnd(from int) int {
	return fl.exprEndThrough(from, from)
}

func (fl *fileLower) exprEndNested(from int) int {
	return fl.exprEndThrough(from, len(fl.src))
}

func (fl *fileLower) exprEndThrough(from, through int) int {
	last := from
	depth := 0
	scanSegment(fl.src, from, len(fl.src), func(t tok) bool {
		if t.comment {
			return true
		}
		if t.auto {
			return depth > 0
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK:
			depth++
		case token.LBRACE:
			if depth == 0 && t.start >= through {
				return false
			}
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				return false
			}
			depth--
		case token.COMMA, token.SEMICOLON, token.COLON, token.ASSIGN, token.DEFINE:
			if depth == 0 && !fl.fatArrowAt(t) {
				return false
			}
		}
		last = t.end
		return true
	})
	return last
}

func (fl *fileLower) fatArrowAt(t tok) bool {
	return t.kind == token.ASSIGN && t.end < len(fl.src) && fl.src[t.end] == '>'
}
