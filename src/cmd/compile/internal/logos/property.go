// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"go/token"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func (fl *fileLower) lowerProperties() {
	fl.lowerPropertyInterfaces()
	fl.lowerPropertyDecls()
	orig := len(fl.file.DeclList)
	for _, decl := range fl.file.DeclList[:orig] {
		switch decl := decl.(type) {
		case *syntax.FuncDecl:
			if decl.Body != nil {
				decl.Body.List = fl.lowerPropBlock(decl.Body.List)
			}
		case *syntax.VarDecl:
			if decl.Values != nil {
				c := &propCollect{}
				decl.Values = fl.propReadIn(decl.Values, c)
				fl.emitPropReads(c)
			}
		}
	}
	fl.flushTypeImports()
}

func (fl *fileLower) lowerPropertyInterfaces() {
	for _, decl := range fl.file.DeclList {
		td, ok := decl.(*syntax.TypeDecl)
		if !ok {
			continue
		}
		it, ok := td.Type.(*syntax.InterfaceType)
		if !ok {
			continue
		}
		fl.lowerPropertyInterface(it)
	}
}

func (fl *fileLower) lowerPropertyInterface(it *syntax.InterfaceType) {
	var out []*syntax.Field
	changed := false
	rw := make(map[string]bool)
	for _, f := range it.MethodList {
		pt, ok := f.Type.(*syntax.PropertyType)
		if !ok || f.Name == nil {
			continue
		}
		if !pt.Const {
			rw[f.Name.Value] = true
		}
	}
	elemStart := func(f *syntax.Field) int {
		start := fl.off(syntax.StartPos(f))
		if pt, ok := f.Type.(*syntax.PropertyType); ok {
			start = min(start, fl.off(pt.Pos()))
		}
		return start
	}
	seen := make(map[string]bool)
	for i, f := range it.MethodList {
		pt, ok := f.Type.(*syntax.PropertyType)
		if !ok || f.Name == nil {
			out = append(out, f)
			continue
		}
		changed = true
		pos := f.Pos()
		name := f.Name.Value
		typeSp, ok := fl.platformSpan(pt.Type)
		if !ok {
			base.Fatalf("logos: cannot span property type at %s", pos.String())
		}
		start := elemStart(f)
		limit := typeSp.end
		if i+1 < len(it.MethodList) {
			limit = min(limit, elemStart(it.MethodList[i+1]))
		}
		afterName := span{start: fl.off(f.Name.Pos()) + len(name), end: lastTokenEnd(fl.src, typeSp.start, limit)}
		typeText := fl.textOf(afterName).trim()
		if seen[name] {
			fl.addT(start, afterName.end, gen(""))
			continue
		}
		seen[name] = true
		getResult := cloneType(pt.Type, pos)
		out = append(out, at(&syntax.Field{
			Name: nameOf(pos, "Logos_Get"+name),
			Type: at(&syntax.FuncType{ResultList: []*syntax.Field{at(&syntax.Field{Type: getResult}, pos)}}, pos),
		}, pos))
		parts := []otext{gen("Logos_Get" + name + "() "), typeText}
		if !pt.Const || rw[name] {
			setParam := cloneType(pt.Type, pos)
			out = append(out, at(&syntax.Field{
				Name: nameOf(pos, "Logos_Set"+name),
				Type: at(&syntax.FuncType{ParamList: []*syntax.Field{at(&syntax.Field{Type: setParam}, pos)}}, pos),
			}, pos))
			parts = append(parts, gen("; Logos_Set"+name+"("), typeText, gen(")"))
		}
		fl.addT(start, afterName.end, cat(parts...))
	}
	if changed {
		it.MethodList = out
	}
}

func (fl *fileLower) lowerPropertyDecls() {
	info := fl.pkg.info
	if len(info.LogosPropDecls) == 0 {
		return
	}
	byDef := make(map[*types2.TypeName][]types2.Type)
	for _, d := range info.LogosPropDecls {
		if d == nil || d.Def == nil {
			continue
		}
		byDef[d.Def] = append(byDef[d.Def], d.Ifaces...)
	}
	for _, decl := range fl.file.DeclList {
		td, ok := decl.(*syntax.TypeDecl)
		if !ok || td.Name == nil || td.Alias {
			continue
		}
		obj, ok := info.Defs[td.Name].(*types2.TypeName)
		if !ok {
			continue
		}
		ifaces, ok := byDef[obj]
		if !ok {
			continue
		}
		fl.emitAccessors(td, obj, ifaces)
	}
}

func (fl *fileLower) emitAccessors(td *syntax.TypeDecl, obj *types2.TypeName, ifaces []types2.Type) {
	named, ok := obj.Type().(*types2.Named)
	if !ok {
		base.Fatalf("logos: %s declares a property interface but is not a named type", td.Name.Value)
	}
	seen := make(map[string]bool)
	var props []types2.LogosProp
	for _, iface := range ifaces {
		set, ok := types2.LogosPropsOf(iface)
		if !ok {
			continue
		}
		for _, pr := range set {
			if seen[pr.Name] {
				continue
			}
			seen[pr.Name] = true
			props = append(props, pr)
		}
	}
	slices.SortFunc(props, func(a, b types2.LogosProp) int { return strings.Compare(a.Name, b.Name) })
	pos := td.Pos()
	typeName := td.Name.Value
	recvShape := selfShape(td, pos)
	var added []*syntax.FuncDecl
	var text bytes.Buffer
	for _, pr := range props {
		if _, promoted, _ := types2.LogosPropertySelection(fl.pkg.tpkg, types2.NewPointer(types2.LogosSelfInstance(named)), pr.Name); promoted {
			continue
		}
		getter, setter, ambig, getFn, setFn := propSourceOf(fl.pkg.tpkg, types2.NewPointer(types2.LogosSelfInstance(named)), pr.Name, pr.Typ)
		if ambig || !getter {
			base.Fatalf("logos: property %s of %s has no source at lowering", pr.Name, typeName)
		}
		getResult, ok := fl.spellPropType(pr.Typ, pos)
		if !ok {
			base.Fatalf("logos: property %s of %s has a type that cannot be spelled in this file", pr.Name, typeName)
		}
		getterRecv := recvTypeValue(recvShape, pos)
		if getFn != nil && isPointerRecv(getFn) {
			getterRecv = recvTypePointer(recvShape, pos)
		}
		var getBody syntax.Expr
		if getFn != nil {
			getBody = callOf(pos, selOf(pos, nameOf(pos, "m"), pr.Name), nil)
		} else {
			getBody = selOf(pos, nameOf(pos, "m"), pr.Name)
		}
		added = append(added, at(&syntax.FuncDecl{
			Recv: at(&syntax.Field{Name: nameOf(pos, "m"), Type: getterRecv}, pos),
			Name: nameOf(pos, "Logos_Get"+pr.Name),
			Type: at(&syntax.FuncType{ResultList: []*syntax.Field{at(&syntax.Field{Type: getResult}, pos)}}, pos),
			Body: blockOf(pos, []syntax.Stmt{returnOf(pos, []syntax.Expr{getBody})}),
		}, pos))
		if pr.ReadOnly {
			continue
		}
		if !setter {
			base.Fatalf("logos: mutable property %s of %s has no setter source at lowering", pr.Name, typeName)
		}
		setResult, ok := fl.spellPropType(pr.Typ, pos)
		if !ok {
			base.Fatalf("logos: property %s of %s has a type that cannot be spelled in this file", pr.Name, typeName)
		}
		setterRecv := recvTypePointer(recvShape, pos)
		if getFn != nil && setFn != nil && !isPointerRecv(setFn) {
			setterRecv = recvTypeValue(recvShape, pos)
		}
		setParam := nameOf(pos, "v")
		var setBody syntax.Stmt
		if getFn != nil {
			setBody = exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, "m"), "Set"+pr.Name), []syntax.Expr{setParam}))
		} else {
			lhs := selOf(pos, nameOf(pos, "m"), pr.Name)
			setBody = at(&syntax.AssignStmt{Lhs: lhs, Rhs: setParam}, pos)
		}
		added = append(added, at(&syntax.FuncDecl{
			Recv: at(&syntax.Field{Name: nameOf(pos, "m"), Type: setterRecv}, pos),
			Name: nameOf(pos, "Logos_Set"+pr.Name),
			Type: at(&syntax.FuncType{ParamList: []*syntax.Field{at(&syntax.Field{
				Name: setParam,
				Type: setResult,
			}, pos)}}, pos),
			Body: blockOf(pos, []syntax.Stmt{setBody}),
		}, pos))
	}
	if len(added) == 0 {
		return
	}
	for _, d := range added {
		var buf bytes.Buffer
		_, _ = syntax.Fprint(&buf, d, 0)
		text.WriteString("\n\n")
		text.Write(buf.Bytes())
	}
	for _, d := range added {
		fl.file.DeclList = append(fl.file.DeclList, d)
	}
	fl.add(fl.orig, fl.orig, text.String())
}

func selfShape(td *syntax.TypeDecl, pos syntax.Pos) syntax.Expr {
	var shape syntax.Expr = nameOf(pos, td.Name.Value)
	if len(td.TParamList) == 0 {
		return shape
	}
	args := make([]syntax.Expr, len(td.TParamList))
	for i, p := range td.TParamList {
		args[i] = nameOf(pos, p.Name.Value)
	}
	return at(&syntax.IndexExpr{X: shape, Index: listOf(pos, args)}, pos)
}

func recvTypeValue(base syntax.Expr, pos syntax.Pos) syntax.Expr {
	return respellRecv(base, pos, false)
}

func recvTypePointer(base syntax.Expr, pos syntax.Pos) syntax.Expr {
	return respellRecv(base, pos, true)
}

func respellRecv(shape syntax.Expr, pos syntax.Pos, pointer bool) syntax.Expr {
	text, ok := respellType(shape, pos)
	if !ok {
		base.Fatalf("logos: cannot respell receiver type %q", renderExpr(shape))
	}
	if !pointer {
		return text
	}
	return at(&syntax.Operation{Op: syntax.Mul, X: text}, pos)
}

func isPointerRecv(fn *types2.Func) bool {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return false
	}
	_, ok = types2.Unalias(sig.Recv().Type()).(*types2.Pointer)
	return ok
}

func (fl *fileLower) spellPropType(t types2.Type, pos syntax.Pos) (syntax.Expr, bool) {
	text, ok := fl.spellLowered(t)
	if !ok {
		return nil, false
	}
	return typeExprAt(text, pos)
}

func propSourceOf(pkg *types2.Package, v types2.Type, name string, want types2.Type) (getter, setter, ambig bool, getFn, setFn *types2.Func) {
	obj, index, _ := types2.LookupFieldOrMethod(v, false, pkg, name)
	if obj == nil {
		if index != nil {
			return false, false, true, nil, nil
		}
	} else if fld, ok := obj.(*types2.Var); ok {
		if types2.Identical(fld.Type(), want) {
			return true, true, false, nil, nil
		}
		return false, false, false, nil, nil
	} else if m, ok := obj.(*types2.Func); ok {
		if sig, ok := m.Type().(*types2.Signature); ok && sig.Params().Len() == 0 && sig.Results().Len() == 1 && types2.Identical(sig.Results().At(0).Type(), want) {
			getter = true
			getFn = m
		}
	}
	sobj, sindex, _ := types2.LookupFieldOrMethod(v, false, pkg, "Set"+name)
	if sobj == nil {
		return getter, false, sindex != nil, getFn, nil
	}
	if sm, ok := sobj.(*types2.Func); ok {
		if sig, ok := sm.Type().(*types2.Signature); ok && sig.Params().Len() == 1 && sig.Results().Len() == 0 && types2.Identical(sig.Params().At(0).Type(), want) {
			return getter, true, false, getFn, sm
		}
	}
	return getter, false, false, getFn, nil
}

type propNeedle struct {
	start, end int
	name       string
	recv       span
	temp       string
}

type propHoist struct {
	pres  []syntax.Stmt
	posts []syntax.Stmt
}

type propCollect struct {
	needles []propNeedle
	hoist   *propHoist
}

func (fl *fileLower) propSelOf(e syntax.Expr) (recv syntax.Expr, pr types2.LogosProp, sel *syntax.SelectorExpr, ok bool) {
	var is bool
	if sel, is = syntax.Unparen(e).(*syntax.SelectorExpr); !is || sel.Sel == nil {
		return nil, types2.LogosProp{}, nil, false
	}
	recv = syntax.Unparen(sel.X)
	t := fl.pkg.info.TypeOf(recv)
	if t == nil {
		return nil, types2.LogosProp{}, nil, false
	}
	pr, _, found := types2.LogosPropertySelection(fl.pkg.tpkg, t, sel.Sel.Value)
	if !found {
		return nil, types2.LogosProp{}, nil, false
	}
	return recv, pr, sel, true
}

func (fl *fileLower) propSelSpan(sel *syntax.SelectorExpr) (whole, recv span, ok bool) {
	selOff := fl.off(sel.Sel.Pos())
	start := fl.off(leftmostOf(sel))
	if selOff <= start {
		return span{}, span{}, false
	}
	dot := -1
	scanSegment(fl.src, start, selOff, func(t tok) bool {
		if t.kind == token.PERIOD {
			dot = t.start
		}
		return true
	})
	if dot < 0 {
		return span{}, span{}, false
	}
	return span{start: start, end: selOff + len(sel.Sel.Value)}, span{start: start, end: dot}, true
}

func propTempName(selOff int) string {
	return "_logos_pt" + strconv.Itoa(selOff)
}

func propPinName(selOff int) string {
	return "_logos_pr" + strconv.Itoa(selOff)
}

func propTrivialRecv(recv syntax.Expr) bool {
	_, ok := syntax.Unparen(recv).(*syntax.Name)
	return ok
}

func (fl *fileLower) propReadIn(e syntax.Expr, c *propCollect) syntax.Expr {
	c.hoist = nil
	return fl.propWalk(e, c)
}

func (fl *fileLower) propWalk(e syntax.Expr, c *propCollect) syntax.Expr {
	switch e := e.(type) {
	case nil:
		return nil
	case *syntax.Name, *syntax.BasicLit:
		return e
	case *syntax.ParenExpr:
		e.X = fl.propWalk(e.X, c)
		return e
	case *syntax.TryExpr:
		e.X = fl.propWalk(e.X, c)
		return e
	case *syntax.SelectorExpr:
		if _, pr, sel, ok := fl.propSelOf(e); ok {
			fl.recordPropRead(c, pr, sel)
			e.X = fl.propWalk(e.X, c)
			pos := sel.Pos()
			sel.Sel.Value = "Logos_Get" + pr.Name
			return callOf(pos, sel, nil)
		}
		e.X = fl.propWalk(e.X, c)
		return e
	case *syntax.CallExpr:
		return fl.propWalkCall(e, c)
	case *syntax.IndexExpr:
		e.X = fl.propWalk(e.X, c)
		e.Index = fl.propWalk(e.Index, c)
		return e
	case *syntax.SliceExpr:
		e.X = fl.propWalk(e.X, c)
		for i, x := range e.Index {
			if x != nil {
				e.Index[i] = fl.propWalk(x, c)
			}
		}
		return e
	case *syntax.AssertExpr:
		e.X = fl.propWalk(e.X, c)
		return e
	case *syntax.Operation:
		e.X = fl.propWalk(e.X, c)
		if e.Y != nil {
			e.Y = fl.propWalk(e.Y, c)
		}
		return e
	case *syntax.CompositeLit:
		for i, x := range e.ElemList {
			e.ElemList[i] = fl.propWalk(x, c)
		}
		return e
	case *syntax.KeyValueExpr:
		e.Key = fl.propWalk(e.Key, c)
		e.Value = fl.propWalk(e.Value, c)
		return e
	case *syntax.ListExpr:
		for i, x := range e.ElemList {
			e.ElemList[i] = fl.propWalk(x, c)
		}
		return e
	case *syntax.InterpExpr:
		for i, x := range e.Exprs {
			e.Exprs[i] = fl.propWalk(x, c)
		}
		return e
	case *syntax.FuncLit:
		if e.Body != nil {
			e.Body.List = fl.lowerPropBlock(e.Body.List)
		}
		return e
	default:
		return e
	}
}

func (fl *fileLower) recordPropRead(c *propCollect, pr types2.LogosProp, sel *syntax.SelectorExpr) {
	whole, recvSp, ok := fl.propSelSpan(sel)
	if !ok {
		base.Fatalf("logos: cannot span property selection at %s", sel.Pos().String())
	}
	c.needles = append(c.needles, propNeedle{
		start: whole.start,
		end:   whole.end,
		name:  pr.Name,
		recv:  recvSp,
	})
}

func (fl *fileLower) chainPropGet(c *propCollect, pr types2.LogosProp, sel, inner *syntax.SelectorExpr) {
	fl.recordPropRead(c, pr, inner)
	pos := inner.Pos()
	inner.Sel.Value = "Logos_Get" + pr.Name
	sel.X = callOf(pos, inner, nil)
}

func (fl *fileLower) propWalkCall(e *syntax.CallExpr, c *propCollect) syntax.Expr {
	sel, targ, ok := propCallSel(e.Fun)
	if !ok {
		e.Fun = fl.propWalk(e.Fun, c)
		fl.propWalkArgs(e.ArgList, c)
		return e
	}
	recv, pr, inner, ok := fl.propSelOf(sel.X)
	if !ok {
		e.Fun = fl.propWalk(e.Fun, c)
		fl.propWalkArgs(e.ArgList, c)
		return e
	}
	if propSelIsField(fl.pkg.info, sel) || targ || c.hoist == nil {
		fl.chainPropGet(c, pr, sel, inner)
		fl.propWalkArgs(e.ArgList, c)
		return e
	}
	fl.hoistPropCall(c, propSite{sel: sel, recv: recv, pr: pr, inner: inner})
	fl.propWalkArgs(e.ArgList, c)
	return e
}

type propSite struct {
	sel   *syntax.SelectorExpr
	recv  syntax.Expr
	pr    types2.LogosProp
	inner *syntax.SelectorExpr
}

type propTemp struct {
	pos    syntax.Pos
	pin    string
	temp   string
	needle propNeedle
	recv   syntax.Expr
	pr     types2.LogosProp
}

func propCallSel(fun syntax.Expr) (sel *syntax.SelectorExpr, targ bool, ok bool) {
	for {
		switch inner := syntax.Unparen(fun).(type) {
		case *syntax.SelectorExpr:
			return inner, targ, true
		case *syntax.IndexExpr:
			targ = true
			fun = inner.X
		default:
			return nil, false, false
		}
	}
}

func propSelIsField(info *types2.Info, sel *syntax.SelectorExpr) bool {
	if info == nil || info.Selections == nil {
		return false
	}
	s, ok := info.Selections[sel]
	if !ok || s == nil {
		return false
	}
	_, ok = s.Obj().(*types2.Var)
	return ok
}

func (fl *fileLower) propWalkArgs(args []syntax.Expr, c *propCollect) {
	for i, a := range args {
		args[i] = fl.propWalk(a, c)
	}
}

func (fl *fileLower) hoistPropCall(c *propCollect, site propSite) string {
	whole, recvSp, ok := fl.propSelSpan(site.inner)
	if !ok {
		base.Fatalf("logos: cannot span property selection at %s", site.inner.Pos().String())
	}
	selOff := fl.off(site.inner.Sel.Pos())
	pos := site.inner.Pos()
	t := propTemp{
		pos:    pos,
		temp:   propTempName(selOff),
		needle: propNeedle{start: whole.start, end: whole.end, name: site.pr.Name, recv: recvSp, temp: propTempName(selOff)},
		recv:   site.recv,
		pr:     site.pr,
	}
	if !propTrivialRecv(site.recv) {
		t.pin = propPinName(selOff)
	}
	site.inner.Sel.Value = t.temp
	site.sel.X = nameOf(leftmostOf(site.recv), t.temp)
	c.needles = append(c.needles, t.needle)
	c.hoist.pres = append(c.hoist.pres, t.prelude()...)
	if !site.pr.ReadOnly && propNeedsWriteBack(fl.pkg.info, site.sel) {
		c.hoist.posts = append(c.hoist.posts, t.writeBack())
	}
	return t.temp
}

func propNeedsWriteBack(info *types2.Info, sel *syntax.SelectorExpr) bool {
	if info == nil || info.Selections == nil {
		return true
	}
	s, ok := info.Selections[sel]
	if !ok || s == nil {
		return true
	}
	fn, ok := s.Obj().(*types2.Func)
	if !ok {
		return true
	}
	return isPointerRecv(fn)
}

func (t propTemp) prelude() []syntax.Stmt {
	if t.pin == "" {
		return []syntax.Stmt{
			defineOf(t.pos, []syntax.Expr{nameOf(t.pos, t.temp)}, []syntax.Expr{t.getCall(t.recv)}),
		}
	}
	return []syntax.Stmt{
		defineOf(t.pos, []syntax.Expr{nameOf(t.pos, t.pin)}, []syntax.Expr{t.recv}),
		defineOf(t.pos, []syntax.Expr{nameOf(t.pos, t.temp)}, []syntax.Expr{t.getCall(nameOf(t.pos, t.pin))}),
	}
}

func (t propTemp) getCall(recv syntax.Expr) syntax.Expr {
	return callOf(t.pos, selOf(t.pos, recv, "Logos_Get"+t.pr.Name), nil)
}

func (t propTemp) writeBack() syntax.Stmt {
	base := t.recv
	if t.pin != "" {
		base = nameOf(t.pos, t.pin)
	}
	return exprStmtOf(t.pos, callOf(t.pos, selOf(t.pos, base, "Logos_Set"+t.pr.Name), []syntax.Expr{nameOf(t.pos, t.temp)}))
}

func (fl *fileLower) lowerPropBlock(list []syntax.Stmt) []syntax.Stmt {
	var out []syntax.Stmt
	for _, s := range list {
		out = append(out, fl.lowerPropStmt(s)...)
	}
	return out
}

func (fl *fileLower) lowerPropStmt(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		s.List = fl.lowerPropBlock(s.List)
		return []syntax.Stmt{s}
	case *syntax.IfStmt:
		s.Init = fl.propClause(s.Init, -1)
		if s.Cond != nil {
			s.Cond = fl.propReadTop(s.Cond)
		}
		s.Then.List = fl.lowerPropBlock(s.Then.List)
		switch els := s.Else.(type) {
		case *syntax.BlockStmt:
			els.List = fl.lowerPropBlock(els.List)
		case *syntax.IfStmt:
			s.Else = fl.lowerPropStmt(els)[0]
		}
		return []syntax.Stmt{s}
	case *syntax.ForStmt:
		s.Init = fl.propClause(s.Init, -1)
		if s.Cond != nil {
			s.Cond = fl.propReadTop(s.Cond)
		}
		s.Post = fl.propClause(s.Post, fl.off(s.Body.Pos()))
		s.Body.List = fl.lowerPropBlock(s.Body.List)
		return []syntax.Stmt{s}
	case *syntax.SwitchStmt:
		s.Init = fl.propClause(s.Init, -1)
		if s.Tag != nil {
			if tg, ok := s.Tag.(*syntax.TypeSwitchGuard); ok {
				tg.X = fl.propReadTop(tg.X)
			} else {
				s.Tag = fl.propReadTop(s.Tag)
			}
		}
		for _, cc := range s.Body {
			if cc.Cases != nil {
				cc.Cases = fl.propReadTop(cc.Cases)
			}
			cc.Body = fl.lowerPropBlock(cc.Body)
		}
		return []syntax.Stmt{s}
	case *syntax.SelectStmt:
		for _, cc := range s.Body {
			fl.propReadSimple(cc.Comm)
			cc.Body = fl.lowerPropBlock(cc.Body)
		}
		return []syntax.Stmt{s}
	case *syntax.RangeClause:
		if s.Lhs != nil {
			s.Lhs = fl.propReadTop(s.Lhs)
		}
		s.X = fl.propReadTop(s.X)
		return []syntax.Stmt{s}
	case *syntax.LabeledStmt:
		inner := fl.lowerPropStmt(s.Stmt)
		s.Stmt = inner[0]
		return append([]syntax.Stmt{s}, inner[1:]...)
	case *syntax.DeclStmt:
		return fl.lowerPropDeclStmt(s)
	case *syntax.SendStmt:
		return fl.lowerPropSend(s)
	case *syntax.AssignStmt:
		return fl.lowerPropAssign(s)
	case *syntax.ExprStmt:
		return fl.lowerPropExprStmt(s)
	case *syntax.CallStmt:
		fl.propReadSimple(s)
		return []syntax.Stmt{s}
	case *syntax.ReturnStmt:
		fl.propReadSimple(s)
		return []syntax.Stmt{s}
	case *syntax.ErrdeferStmt:
		fl.propReadSimple(s)
		return []syntax.Stmt{s}
	default:
		return []syntax.Stmt{s}
	}
}

func (fl *fileLower) propReadTop(e syntax.Expr) syntax.Expr {
	if e == nil {
		return nil
	}
	c := &propCollect{}
	e = fl.propReadIn(e, c)
	fl.emitPropReads(c)
	return e
}

func (fl *fileLower) propReadSimple(s syntax.Stmt) {
	if s == nil {
		return
	}
	c := &propCollect{}
	switch s := s.(type) {
	case *syntax.AssignStmt:
		if s.Lhs != nil {
			s.Lhs = fl.propReadIn(s.Lhs, c)
		}
		if s.Rhs != nil {
			s.Rhs = fl.propReadIn(s.Rhs, c)
		}
	case *syntax.ExprStmt:
		s.X = fl.propReadIn(s.X, c)
	case *syntax.SendStmt:
		s.Chan = fl.propReadIn(s.Chan, c)
		s.Value = fl.propReadIn(s.Value, c)
	case *syntax.CallStmt:
		s.Call = fl.propReadIn(s.Call, c)
	case *syntax.ReturnStmt:
		if s.Results != nil {
			s.Results = fl.propReadIn(s.Results, c)
		}
	case *syntax.ErrdeferStmt:
		s.Call = fl.propReadIn(s.Call, c)
	case *syntax.DeclStmt:
		for _, d := range s.DeclList {
			if vd, ok := d.(*syntax.VarDecl); ok && vd.Values != nil {
				vd.Values = fl.propReadIn(vd.Values, c)
			}
		}
	case *syntax.RangeClause:
		if s.Lhs != nil {
			s.Lhs = fl.propReadIn(s.Lhs, c)
		}
		s.X = fl.propReadIn(s.X, c)
	}
	fl.emitPropReads(c)
}

func (fl *fileLower) emitPropReads(c *propCollect) {
	for _, n := range c.needles {
		if n.temp != "" || fl.propEmitted[n.end] {
			continue
		}
		if fl.propEmitted == nil {
			fl.propEmitted = make(map[int]bool)
		}
		fl.propEmitted[n.end] = true
		off := n.end - len(n.name)
		fl.add(off, n.end, "Logos_Get"+n.name+"()")
	}
}

func (fl *fileLower) lowerPropAssign(s *syntax.AssignStmt) []syntax.Stmt {
	lhs := syntax.UnpackListExpr(s.Lhs)
	if _, _, isTry := tryAssignOf(s); isTry && s.Op == 0 {
		for _, l := range lhs {
			if _, _, _, ok := fl.propSelOf(l); ok {
				c := &propCollect{}
				s.Rhs = fl.propWalk(s.Rhs, c)
				fl.emitPropReads(c)
				return []syntax.Stmt{s}
			}
		}
	}
	if s.Rhs == nil {
		if len(lhs) != 1 {
			return []syntax.Stmt{s}
		}
		if recv, pr, _, ok := fl.propSelOf(lhs[0]); ok {
			if pr.ReadOnly {
				return fl.lowerPropGeneric(s)
			}
			return fl.lowerPropCompound(s, recv, pr, nil)
		}
		return fl.lowerPropGeneric(s)
	}
	if s.Op != 0 && s.Op != syntax.Def {
		if len(lhs) != 1 {
			return []syntax.Stmt{s}
		}
		if recv, pr, _, ok := fl.propSelOf(lhs[0]); ok {
			if pr.ReadOnly {
				return fl.lowerPropGeneric(s)
			}
			rhs := syntax.UnpackListExpr(s.Rhs)
			if len(rhs) != 1 {
				return []syntax.Stmt{s}
			}
			return fl.lowerPropCompound(s, recv, pr, rhs[0])
		}
		return fl.lowerPropGeneric(s)
	}
	if len(lhs) == 1 {
		if recv, pr, _, ok := fl.propSelOf(lhs[0]); ok {
			if s.Op != 0 || pr.ReadOnly {
				return fl.lowerPropGeneric(s)
			}
			return fl.lowerPropWrite(s, recv, pr)
		}
	}
	if len(lhs) > 1 {
		roots := make([]int, 0, len(lhs))
		for i, l := range lhs {
			if _, _, _, ok := fl.propSelOf(l); ok {
				roots = append(roots, i)
			}
		}
		if len(roots) > 0 {
			if s.Op != 0 {
				return fl.lowerPropGeneric(s)
			}
			return fl.lowerPropMulti(s, lhs, roots)
		}
	}
	return fl.lowerPropGeneric(s)
}

func (fl *fileLower) propStmtSpan(s syntax.Stmt, first syntax.Expr) (span, bool) {
	sp, ok := fl.platformSpan(first)
	if !ok {
		return span{}, false
	}
	end := stmtEnd(fl.src, sp.start)
	if limit, ok := fl.stmtLimit[s]; ok {
		end = lastTokenEnd(fl.src, sp.start, min(end, limit))
	}
	if end <= sp.start {
		return span{}, false
	}
	return span{start: sp.start, end: end}, true
}

func (fl *fileLower) propClause(clause syntax.SimpleStmt, limit int) syntax.SimpleStmt {
	if clause == nil {
		return nil
	}
	if as, ok := clause.(*syntax.AssignStmt); ok && as.Op == syntax.Def {
		fl.propReadSimple(clause)
		return clause
	}
	if limit >= 0 {
		if fl.stmtLimit == nil {
			fl.stmtLimit = make(map[syntax.Stmt]int)
		}
		fl.stmtLimit[clause] = limit
	}
	var first syntax.Expr
	switch c := clause.(type) {
	case *syntax.AssignStmt:
		first = syntax.UnpackListExpr(c.Lhs)[0]
	case *syntax.ExprStmt:
		first = c.X
	case *syntax.SendStmt:
		first = c.Chan
	}
	var sp span
	spOK := false
	if first != nil {
		sp, spOK = fl.propStmtSpan(clause, first)
	}
	out := fl.lowerPropStmt(clause)
	if len(out) == 1 && out[0] == clause {
		return clause
	}
	if !spOK {
		base.Fatalf("logos: cannot span clause statement at %s", clause.Pos().String())
	}
	pos := clause.Pos()
	lit := at(&syntax.FuncLit{Type: at(&syntax.FuncType{}, pos), Body: blockOf(pos, out)}, pos)
	fl.addT(sp.start, sp.end, cat(gen("func() { "), fl.textOf(sp), gen(" }()")))
	return exprStmtOf(pos, callOf(pos, lit, nil))
}

func propNeedlesNest(needles []propNeedle) bool {
	for i := range needles {
		for j := range needles {
			if i != j && needles[i].start <= needles[j].start && needles[i].end >= needles[j].end &&
				(needles[i].start != needles[j].start || needles[i].end != needles[j].end) {
				return true
			}
		}
	}
	return false
}

type propSubst struct {
	needle otext
	repl   otext
	len    int
	start  int
	end    int
}

func (fl *fileLower) propSubstitutions(needles []propNeedle) []propSubst {
	subs := make([]propSubst, 0, len(needles))
	for _, n := range needles {
		if n.temp != "" {
			subs = append(subs, propSubst{needle: fl.textOf(span{start: n.start, end: n.end}), repl: gen(n.temp), len: n.end - n.start, start: n.start, end: n.end})
			continue
		}
		subs = append(subs, propSubst{
			needle: fl.textOf(span{start: n.start, end: n.end}),
			repl:   cat(fl.textOf(n.recv), gen(".Logos_Get"+n.name+"()")),
			len:    n.end - n.start,
			start:  n.start,
			end:    n.end,
		})
	}
	slices.SortFunc(subs, func(a, b propSubst) int {
		if d := b.len - a.len; d != 0 {
			return d
		}
		if a.start != b.start {
			if a.start < b.start {
				return -1
			}
			return 1
		}
		return 0
	})
	return subs
}

func propSubstText(text otext, subs []propSubst) otext {
	for _, s := range subs {
		if s.needle.s == "" {
			continue
		}
		text = propSubstOne(text, s)
	}
	return text
}

func propSubstOne(t otext, s propSubst) otext {
	needle := s.needle.s
	for off := 0; off+len(needle) <= len(t.s); {
		idx := strings.Index(t.s[off:], needle)
		if idx < 0 {
			return t
		}
		i := off + idx
		j := i + len(needle)
		if len(t.chunks) == 0 || propSliceMatches(t.slice(i, j), s.start, s.end) {
			return cat(t.slice(0, i), s.repl, t.slice(j, len(t.s)))
		}
		off = i + 1
	}
	return t
}

func propSliceMatches(sl otext, start, end int) bool {
	if len(sl.chunks) == 0 {
		return false
	}
	if sl.chunks[0].out != 0 || sl.chunks[0].orig != start {
		return false
	}
	last := sl.chunks[len(sl.chunks)-1]
	if last.out+last.n != len(sl.s) {
		return false
	}
	if last.orig+last.n != end {
		return false
	}
	off := 0
	for _, c := range sl.chunks {
		if c.out != off {
			return false
		}
		off += c.n
	}
	return off == len(sl.s)
}

func (fl *fileLower) lowerPropGeneric(s *syntax.AssignStmt) []syntax.Stmt {
	c := &propCollect{hoist: &propHoist{}}
	s.Lhs = fl.propWalk(s.Lhs, c)
	if s.Rhs != nil {
		s.Rhs = fl.propWalk(s.Rhs, c)
	}
	if len(c.hoist.pres) == 0 && len(c.hoist.posts) == 0 && !propNeedlesNest(c.needles) {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	sp, ok := fl.propStmtSpan(s, syntax.UnpackListExpr(s.Lhs)[0])
	if !ok {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	lhs := syntax.UnpackListExpr(s.Lhs)
	lhsSpans, ok := fl.propLhsSpans(s, lhs)
	if !ok {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	lhsTexts := make([]otext, 0, len(lhs))
	for _, lsp := range lhsSpans {
		lhsTexts = append(lhsTexts, fl.textOf(lsp))
	}
	rhsText := otext{}
	if s.Rhs != nil {
		rhsText = fl.propOperandText(s.Rhs, c)
	}
	subs := fl.propSubstitutions(c.needles)
	for i := range lhsTexts {
		lhsTexts[i] = propSubstText(lhsTexts[i], subs)
	}
	rhsText = propSubstText(rhsText, subs)
	op := "="
	if s.Op == syntax.Def {
		op = ":="
	} else if s.Op != 0 {
		op = s.Op.String()
	}
	main := cat(joinO(lhsTexts, ", "), gen(" "+op+" "), rhsText)
	if s.Rhs == nil {
		main = cat(joinO(lhsTexts, ", "), gen(op+op))
	}
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []otext{main})
}

func propMustSpan(fl *fileLower, e syntax.Expr) span {
	if list, ok := e.(*syntax.ListExpr); ok && len(list.ElemList) > 0 {
		first := propMustSpan(fl, list.ElemList[0])
		last := propMustSpan(fl, list.ElemList[len(list.ElemList)-1])
		return span{start: first.start, end: last.end}
	}
	sp, ok := fl.platformSpan(e)
	if !ok {
		base.Fatalf("logos: cannot span expression at %s", e.Pos().String())
	}
	return sp
}

func (fl *fileLower) propOperandText(e syntax.Expr, c *propCollect) otext {
	sp := propMustSpan(fl, e)
	if propNeedleWithin(c.needles, sp) {
		return fl.drainBodyWrapper(e, fl.textOf(sp))
	}
	return fl.drainBodyWrapper(e, fl.lazyText(sp))
}

func propNeedleWithin(needles []propNeedle, sp span) bool {
	for _, n := range needles {
		if n.start < sp.end && n.end > sp.start {
			return true
		}
	}
	return false
}

func (fl *fileLower) propRecvText(sel *syntax.SelectorExpr) otext {
	_, recvSp, ok := fl.propSelSpan(sel)
	if !ok {
		base.Fatalf("logos: cannot span property selection at %s", sel.Pos().String())
	}
	return fl.textOf(recvSp)
}

func (fl *fileLower) propLhsSpans(s *syntax.AssignStmt, lhs []syntax.Expr) ([]span, bool) {
	out := make([]span, 0, len(lhs))
	for i, l := range lhs {
		sp, ok := fl.platformSpan(l)
		if !ok {
			return nil, false
		}
		if i == len(lhs)-1 && s.Rhs != nil {
			sp.end = fl.off(s.Pos())
			if sp.end <= sp.start {
				return nil, false
			}
		}
		out = append(out, sp)
	}
	return out, true
}

func (fl *fileLower) finishPropReplace(c *propCollect, sp span, mains []syntax.Stmt, mainTexts []otext) []syntax.Stmt {
	_ = fl.textOf(sp)
	subs := fl.propSubstitutions(c.needles)
	var texts []otext
	var out []syntax.Stmt
	for _, st := range c.hoist.pres {
		texts = append(texts, propSubstText(gen(fl.propStmtText(st)), subs))
		out = append(out, st)
	}
	for i, m := range mains {
		texts = append(texts, propSubstText(mainTexts[i], subs))
		out = append(out, m)
	}
	for _, st := range c.hoist.posts {
		texts = append(texts, propSubstText(gen(fl.propStmtText(st)), subs))
		out = append(out, st)
	}
	fl.addT(sp.start, sp.end, joinO(texts, "; "))
	return out
}

func (fl *fileLower) propStmtText(st syntax.Stmt) string {
	var buf bytes.Buffer
	_, _ = syntax.Fprint(&buf, st, 0)
	return buf.String()
}

func (fl *fileLower) lowerPropWrite(s *syntax.AssignStmt, recv syntax.Expr, pr types2.LogosProp) []syntax.Stmt {
	sp, ok := fl.propStmtSpan(s, s.Lhs)
	if !ok {
		c := &propCollect{}
		s.Lhs = fl.propReadIn(s.Lhs, c)
		s.Rhs = fl.propReadIn(s.Rhs, c)
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	c := &propCollect{hoist: &propHoist{}}
	recv = fl.propWalk(recv, c)
	rhs := fl.propWalk(s.Rhs, c)
	sel := mustPropSel(fl, s.Lhs)
	recvText := fl.propRecvText(sel)
	rhsText := fl.propOperandText(s.Rhs, c)
	set := exprStmtOf(s.Pos(), callOf(s.Pos(), selOf(s.Pos(), recv, "Logos_Set"+pr.Name), []syntax.Expr{rhs}))
	subs := fl.propSubstitutions(c.needles)
	main := cat(propSubstText(recvText, subs), gen(".Logos_Set"+pr.Name+"("), propSubstText(rhsText, subs), gen(")"))
	return fl.finishPropReplace(c, sp, []syntax.Stmt{set}, []otext{main})
}

func (fl *fileLower) lowerPropCompound(s *syntax.AssignStmt, recv syntax.Expr, pr types2.LogosProp, rhs syntax.Expr) []syntax.Stmt {
	sp, ok := fl.propStmtSpan(s, s.Lhs)
	if !ok {
		c := &propCollect{}
		s.Lhs = fl.propReadIn(s.Lhs, c)
		if s.Rhs != nil {
			s.Rhs = fl.propReadIn(s.Rhs, c)
		}
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	c := &propCollect{hoist: &propHoist{}}
	recv = fl.propWalk(recv, c)
	if rhs != nil {
		rhs = fl.propWalk(rhs, c)
	}
	pos := s.Pos()
	recvText := fl.propRecvText(mustPropSel(fl, s.Lhs))
	pin := ""
	if !propTrivialRecv(recv) {
		pin = propPinName(fl.off(mustPropSel(fl, s.Lhs).Sel.Pos()))
		c.hoist.pres = append(c.hoist.pres, defineOf(pos, []syntax.Expr{nameOf(pos, pin)}, []syntax.Expr{recv}))
	}
	valueText := gen("1")
	if rhs != nil {
		valueText = fl.propOperandText(rhs, c)
	}
	bop := s.Op.String()
	getBase, setBase := recv, recv
	if pin != "" {
		getBase = nameOf(pos, pin)
		setBase = nameOf(pos, pin)
	} else if n, ok := syntax.Unparen(recv).(*syntax.Name); ok {
		setBase = nameOf(pos, n.Value)
	}
	var valueNode syntax.Expr = litOf(pos, "1", syntax.IntLit)
	if rhs != nil {
		valueNode = rhs
	}
	set := exprStmtOf(pos, callOf(pos, selOf(pos, setBase, "Logos_Set"+pr.Name),
		[]syntax.Expr{at(&syntax.Operation{Op: s.Op, X: callOf(pos, selOf(pos, getBase, "Logos_Get"+pr.Name), nil), Y: valueNode}, pos)}))
	ref := recvText
	if pin != "" {
		ref = gen(pin)
	}
	subs := fl.propSubstitutions(c.needles)
	main := cat(propSubstText(ref, subs), gen(".Logos_Set"+pr.Name+"("), propSubstText(ref, subs), gen(".Logos_Get"+pr.Name+"() "+bop+" "), propSubstText(valueText, subs), gen(")"))
	return fl.finishPropReplace(c, sp, []syntax.Stmt{set}, []otext{main})
}

func mustPropSel(fl *fileLower, e syntax.Expr) *syntax.SelectorExpr {
	_, _, sel, ok := fl.propSelOf(e)
	if !ok {
		base.Fatalf("logos: property target lost at %s", e.Pos().String())
	}
	return sel
}

func (fl *fileLower) lowerPropMulti(s *syntax.AssignStmt, lhs []syntax.Expr, roots []int) []syntax.Stmt {
	sp, spOK := fl.propStmtSpan(s, lhs[0])
	c := &propCollect{hoist: &propHoist{}}
	for i, l := range lhs {
		if slices.Contains(roots, i) {
			continue
		}
		fl.propWalk(l, c)
	}
	s.Rhs = fl.propWalk(s.Rhs, c)
	if !spOK {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	pos := s.Pos()
	lhsSpans, ok := fl.propLhsSpans(s, lhs)
	if !ok {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	lhsTexts := make([]otext, len(lhs))
	for _, i := range roots {
		temp, ok := fl.propRootHoist(c, pos, lhs[i])
		if !ok {
			fl.emitPropReads(c)
			return []syntax.Stmt{s}
		}
		fl.propRetarget(s, i, temp)
		lhsTexts[i] = gen(temp)
	}
	for i := range lhs {
		if slices.Contains(roots, i) {
			continue
		}
		lhsTexts[i] = fl.textOf(lhsSpans[i])
	}
	rhsText := fl.propOperandText(s.Rhs, c)
	subs := fl.propSubstitutions(c.needles)
	for i := range lhsTexts {
		lhsTexts[i] = propSubstText(lhsTexts[i], subs)
	}
	main := cat(joinO(lhsTexts, ", "), gen(" = "), propSubstText(rhsText, subs))
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []otext{main})
}

func (fl *fileLower) propRootHoist(c *propCollect, pos syntax.Pos, l syntax.Expr) (string, bool) {
	recv, pr, sel, ok := fl.propSelOf(l)
	if !ok || pr.ReadOnly {
		return "", false
	}
	fl.propWalk(recv, c)
	_, recvSp, ok := fl.propSelSpan(sel)
	if !ok {
		return "", false
	}
	_ = fl.textOf(recvSp)
	temp := propTempName(fl.off(sel.Sel.Pos()))
	typ, ok := fl.spellPropType(pr.Typ, pos)
	if !ok {
		base.Fatalf("logos: property %s has a type that cannot be spelled in this file", pr.Name)
	}
	setBase := recv
	if !propTrivialRecv(recv) {
		pin := temp + "r"
		c.hoist.pres = append(c.hoist.pres, defineOf(pos, []syntax.Expr{nameOf(pos, pin)}, []syntax.Expr{recv}))
		setBase = nameOf(pos, pin)
	}
	c.hoist.pres = append(c.hoist.pres, at(&syntax.DeclStmt{DeclList: []syntax.Decl{
		at(&syntax.VarDecl{NameList: []*syntax.Name{nameOf(pos, temp)}, Type: typ}, pos),
	}}, pos))
	c.hoist.posts = append(c.hoist.posts, exprStmtOf(pos, callOf(pos, selOf(pos, setBase, "Logos_Set"+pr.Name), []syntax.Expr{nameOf(pos, temp)})))
	return temp, true
}

func (fl *fileLower) propRetarget(s *syntax.AssignStmt, i int, temp string) {
	le, ok := s.Lhs.(*syntax.ListExpr)
	if !ok {
		base.Fatalf("logos: multi-assign target without a list at %s", s.Pos().String())
	}
	le.ElemList[i] = nameOf(s.Pos(), temp)
}

func (fl *fileLower) lowerPropExprStmt(s *syntax.ExprStmt) []syntax.Stmt {
	call, ok := syntax.Unparen(s.X).(*syntax.CallExpr)
	if !ok {
		c := &propCollect{}
		s.X = fl.propReadIn(s.X, c)
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	sel, targ, ok := propCallSel(call.Fun)
	if !ok {
		return fl.lowerPropCallArgs(s, call)
	}
	recv, pr, inner, ok := fl.propSelOf(sel.X)
	if !ok {
		return fl.lowerPropCallArgs(s, call)
	}
	if propSelIsField(fl.pkg.info, sel) || targ {
		return fl.lowerPropCallArgs(s, call)
	}
	sp, spOK := fl.propStmtSpan(s, s.X)
	c := &propCollect{hoist: &propHoist{}}
	temp := fl.hoistPropCall(c, propSite{sel: sel, recv: recv, pr: pr, inner: inner})
	fl.propWalkArgs(call.ArgList, c)
	if !spOK {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	args := make([]otext, 0, len(call.ArgList))
	for _, a := range call.ArgList {
		args = append(args, fl.propOperandText(a, c))
	}
	subs := fl.propSubstitutions(c.needles)
	for i := range args {
		args[i] = propSubstText(args[i], subs)
	}
	main := cat(gen(temp+"."+sel.Sel.Value+"("), joinO(args, ", "), gen(")"))
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []otext{main})
}

func (fl *fileLower) lowerPropCallArgs(s *syntax.ExprStmt, call *syntax.CallExpr) []syntax.Stmt {
	sp, spOK := fl.propStmtSpan(s, s.X)
	c := &propCollect{hoist: &propHoist{}}
	s.X = fl.propWalk(s.X, c)
	if len(c.hoist.pres) == 0 && len(c.hoist.posts) == 0 && !propNeedlesNest(c.needles) {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	if !spOK {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	main := fl.textOf(sp)
	subs := fl.propSubstitutions(c.needles)
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []otext{propSubstText(main, subs)})
}

func (fl *fileLower) lowerPropDeclStmt(s *syntax.DeclStmt) []syntax.Stmt {
	v0 := propDeclValue(s)
	var sp span
	spOK := v0 != nil
	if spOK {
		sp, spOK = fl.propStmtSpan(s, v0)
	}
	c := &propCollect{hoist: &propHoist{}}
	for _, d := range s.DeclList {
		if vd, ok := d.(*syntax.VarDecl); ok && vd.Values != nil {
			vd.Values = fl.propWalk(vd.Values, c)
		}
	}
	if len(c.hoist.pres) == 0 && len(c.hoist.posts) == 0 && !propNeedlesNest(c.needles) {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	if v0 == nil || !spOK {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	main := fl.textOf(sp)
	subs := fl.propSubstitutions(c.needles)
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []otext{propSubstText(main, subs)})
}

func propDeclValue(s *syntax.DeclStmt) syntax.Expr {
	for _, d := range s.DeclList {
		if vd, ok := d.(*syntax.VarDecl); ok && vd.Values != nil {
			return vd.Values
		}
	}
	return nil
}

func (fl *fileLower) lowerPropSend(s *syntax.SendStmt) []syntax.Stmt {
	sp, spOK := fl.propStmtSpan(s, s.Chan)
	c := &propCollect{hoist: &propHoist{}}
	s.Chan = fl.propWalk(s.Chan, c)
	s.Value = fl.propWalk(s.Value, c)
	if len(c.hoist.pres) == 0 && len(c.hoist.posts) == 0 && !propNeedlesNest(c.needles) {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	if !spOK {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	main := fl.textOf(sp)
	subs := fl.propSubstitutions(c.needles)
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []otext{propSubstText(main, subs)})
}
