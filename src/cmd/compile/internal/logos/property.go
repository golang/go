// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers M4-S2 interface properties (SPEC Sec 2.3, T-203.2).
// The M4-S1 check leaves property selections (`c.HP` on a
// property-interface type) and `: I` declarations in the tree; the
// recheck would reject both, because the lowered interface holds only
// the Logos_GetF/Logos_SetF accessor methods. This pass runs last in
// lowerFile, after every splice-creating pass, so its textOf calls fold
// their splices into the composed text instead of overlapping them:
//
//   - for each `type S struct : I` of this file it generates the
//     accessor methods of T-203.2 (value-receiver getter and
//     pointer-receiver setter from an exported field, or forwarding to
//     F()/SetF(T) with the forwarded method's receiver kind), appended
//     as native declarations with a matching end-of-file eject splice;
//   - property reads through an interface become Get calls, writes
//     become Set calls, compound updates pin the receiver once and
//     write back, and method calls on a property value use
//     read-call-write-back around the call.
//
// Every rewrite builds the native tree and the eject text together. A
// whole-statement replacement starts by discarding textOf of its span,
// which drains a covering splice of an earlier pass (a direct try in the
// same statement); the replacement is then composed from textOf of the
// value regions, which carry the earlier passes' lowered text.
//
// Limits, stated once: a direct try as the right side of a property
// write (`c.HP = f()?`) is left for the recheck to reject loudly; the
// try lowering owns that statement's text and this pass must not fork
// it. Property reads inside errdefer statements, hasErrdefer returns,
// and defer-try calls likewise keep small splices only, so eject
// reports overlapping edits loudly there instead of forking owned text.

// lowerProperties lowers every property declaration and use of one file.
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
			// Package-level initializers hold no statements, so only
			// pure reads lower here; anything needing a temporary is
			// left for the recheck to reject loudly.
			if decl.Values != nil {
				c := &propCollect{}
				decl.Values = fl.propReadIn(decl.Values, c)
				fl.emitPropReads(c)
			}
		}
	}
	fl.flushTypeImports()
	MergePropFacts(fl.pkg.tpkg, fl.pkg.info)
}

// lowerPropertyInterfaces rewrites every property element of every
// interface of this file to its accessor methods: `const F T` becomes
// `Logos_GetF() T`, and `F T` gains `Logos_SetF(T)` beside it. The
// native tree and the eject text change together, so the recheck and
// the ejected package both see ordinary Go methods.
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

// lowerPropertyInterface rewrites the property elements of one
// interface type to accessor method fields.
func (fl *fileLower) lowerPropertyInterface(it *syntax.InterfaceType) {
	var out []*syntax.Field
	changed := false
	for _, f := range it.MethodList {
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
		typeText := fl.textOf(typeSp)
		start := fl.off(pos)
		if cpos := fl.off(pt.Pos()); cpos < start {
			start = cpos
		}
		getResult := cloneType(pt.Type, pos)
		out = append(out, at(&syntax.Field{
			Name: nameOf(pos, "Logos_Get"+name),
			Type: at(&syntax.FuncType{ResultList: []*syntax.Field{at(&syntax.Field{Type: getResult}, pos)}}, pos),
		}, pos))
		text := "Logos_Get" + name + "() " + typeText
		if !pt.Const {
			setParam := cloneType(pt.Type, pos)
			out = append(out, at(&syntax.Field{
				Name: nameOf(pos, "Logos_Set"+name),
				Type: at(&syntax.FuncType{ParamList: []*syntax.Field{at(&syntax.Field{Type: setParam}, pos)}}, pos),
			}, pos))
			text += "; Logos_Set" + name + "(" + typeText + ")"
		}
		fl.add(start, typeSp.end, text)
	}
	if changed {
		it.MethodList = out
	}
}

// lowerPropertyDecls generates the accessor methods of every `: I`
// struct declared at the top level of this file. Info.LogosPropDecls
// (recorded during the check) names the declaring type and the asserted
// property interfaces; the struct syntax of this file supplies the type
// parameters. Setter and getter sources resolve by the same field-first
// rule the checker uses; a divergence stops the compiler instead of
// emitting a silent mismatch.
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
	if len(byDef) == 0 {
		return
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

// emitAccessors appends the Logos_GetF/Logos_SetF methods of one
// declaring type and records their eject text. Properties merge across
// the asserted interfaces by name; the check already rejected
// conflicting types, so the first spelling of each name wins.
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
	// recvShape spells the bare receiver once; every accessor re-parses
	// it through recvTypeFor, so no two declarations share nodes (A4).
	var recvShape syntax.Expr = nameOf(pos, typeName)
	if len(td.TParamList) > 0 {
		args := make([]syntax.Expr, len(td.TParamList))
		for i, p := range td.TParamList {
			args[i] = nameOf(pos, p.Name.Value)
		}
		recvShape = at(&syntax.IndexExpr{X: recvShape, Index: listOf(pos, args)}, pos)
	}
	var added []*syntax.FuncDecl
	var text bytes.Buffer
	for _, pr := range props {
		getter, setter, ambig, getFn, setFn := propSourceOf(fl.pkg.tpkg, types2.NewPointer(named), pr.Name, pr.Typ)
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
			Recv:       at(&syntax.Field{Name: nameOf(pos, "m"), Type: getterRecv}, pos),
			Name:       nameOf(pos, "Logos_Get"+pr.Name),
			TParamList: cloneTParams(td.TParamList, pos),
			Type:       at(&syntax.FuncType{ResultList: []*syntax.Field{at(&syntax.Field{Type: getResult}, pos)}}, pos),
			Body:       blockOf(pos, []syntax.Stmt{returnOf(pos, []syntax.Expr{getBody})}),
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
		// A method-sourced setter keeps the receiver kind of the
		// forwarded SetF; a field-sourced setter takes a pointer.
		setterRecv := recvTypePointer(recvShape, pos)
		if getFn != nil && setFn != nil && !isPointerRecv(setFn) {
			setterRecv = recvTypeValue(recvShape, pos)
		}
		setParam := nameOf(pos, "v")
		var setBody syntax.Stmt
		if getFn != nil {
			// Method-sourced: forward to SetF. A field of the same
			// name cannot exist here: the checker sources fields
			// first, so a method source means no identical field.
			setBody = exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, "m"), "Set"+pr.Name), []syntax.Expr{setParam}))
		} else {
			lhs := selOf(pos, nameOf(pos, "m"), pr.Name)
			setBody = at(&syntax.AssignStmt{Lhs: lhs, Rhs: setParam}, pos)
		}
		added = append(added, at(&syntax.FuncDecl{
			Recv:       at(&syntax.Field{Name: nameOf(pos, "m"), Type: setterRecv}, pos),
			Name:       nameOf(pos, "Logos_Set"+pr.Name),
			TParamList: cloneTParams(td.TParamList, pos),
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

// cloneTParams copies a type parameter list by cloning each bound, so
// generated methods share no nodes with the type declaration (A4).
func cloneTParams(list []*syntax.Field, pos syntax.Pos) []*syntax.Field {
	var out []*syntax.Field
	for _, p := range list {
		if p == nil || p.Name == nil {
			continue
		}
		np := &syntax.Field{Name: nameOf(pos, p.Name.Value)}
		if p.Type != nil {
			np.Type = cloneType(p.Type, pos)
		}
		out = append(out, at(np, pos))
	}
	return out
}

// recvTypeValue builds the value receiver type S, and recvTypePointer
// the pointer receiver type *S, over one respelled copy of the base:
// respellType re-parses the printed form, so every accessor owns its
// nodes (A4).
func recvTypeValue(base syntax.Expr, pos syntax.Pos) syntax.Expr {
	return respellRecv(base, pos, false)
}

// recvTypePointer builds the pointer receiver type *S.
func recvTypePointer(base syntax.Expr, pos syntax.Pos) syntax.Expr {
	return respellRecv(base, pos, true)
}

// respellRecv re-parses the receiver shape, wrapping it for pointer
// receivers. The boolean selects the spelling the two named builders
// above name explicitly.
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

// isPointerRecv reports whether the method has a pointer receiver.
func isPointerRecv(fn *types2.Func) bool {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return false
	}
	_, ok = types2.Unalias(sig.Recv().Type()).(*types2.Pointer)
	return ok
}

// spellPropType spells t for an accessor signature in this file,
// recording a synthesized import for foreign names. It reports false
// when t mentions an unexported name of another package, which no
// import can spell.
func (fl *fileLower) spellPropType(t types2.Type, pos syntax.Pos) (syntax.Expr, bool) {
	text, ok := fl.spellType(t)
	if !ok {
		return nil, false
	}
	return typeExprAt(text, pos)
}

// propSourceOf finds the accessor sources of one property on V, mirroring
// the checker's logosPropertySource without a checker: it runs after a
// successful check, when every object is complete. A field F of identical
// type sources getter and setter; a method `F() T` sources the getter and
// `SetF(T)` the setter. Embedded fields resolve by Go selector depth; an
// equal-depth match is ambiguous. The getter and setter methods return
// for receiver-kind detection; both are nil for field sources.
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

// propNeedle is one property selector found during collection: its whole
// span for substitution, the original property name, the receiver span,
// and, for a hoisted selector, the temporary holding the value.
type propNeedle struct {
	start, end int
	name       string
	recv       span
	temp       string
}

// propPiece is one hoisted statement; its eject text renders from the
// statement at composition time.
type propPiece struct {
	stmt syntax.Stmt
}

// propHoist collects the write-back temporaries of one statement: the
// prelude statements before it and the write-backs after it.
type propHoist struct {
	pres  []propPiece
	posts []propPiece
}

// propCollect gathers one expression walk: every nested property read
// for Get-inline substitution, plus the write-back hoists when the
// caller supplies hoist state. A nil hoist means a reads-only position
// (conditions, results, deferred calls), where a method call on a
// property value chains off the Get result; a non-nil hoist builds
// read-call-write-back temporaries. The nil check tests for caller
// data, not a mode flag: reads-only positions have no statement to
// attach a prelude or write-back to.
type propCollect struct {
	needles []propNeedle
	hoist   *propHoist
}

// propSelOf returns the property selector when e spells `R.F` with R of
// property-interface type and F in its property set, with the
// receiver, the property, and the selector node.
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
	for {
		p, is := t.(*types2.Pointer)
		if !is {
			break
		}
		t = p.Elem()
	}
	set, is := types2.LogosPropsOf(t)
	if !is {
		return nil, types2.LogosProp{}, nil, false
	}
	for _, pr := range set {
		if pr.Name == sel.Sel.Value {
			return recv, pr, sel, true
		}
	}
	return nil, types2.LogosProp{}, nil, false
}

// propSelSpan returns the whole span of sel and the receiver span. The
// receiver ends at the dot before the selector name, so no token scan
// is needed; trailing space before a spaced dot stays in the receiver
// text, which stays valid.
func (fl *fileLower) propSelSpan(sel *syntax.SelectorExpr) (whole, recv span, ok bool) {
	selOff := fl.off(sel.Sel.Pos())
	start := fl.off(leftmostOf(sel))
	if selOff <= start {
		return span{}, span{}, false
	}
	return span{start: start, end: selOff + len(sel.Sel.Value)}, span{start: start, end: selOff - 1}, true
}

// propTempName mints the temporary for the selector at selOff: one
// spelling per source position, so every temporary is unique without
// threading counter state through the walk (identifiers with the
// _logos_ prefix cannot collide with user code; C6 rejects them).
func propTempName(selOff int) string {
	return "_logos_pt" + itoa(selOff)
}

// propPinName mints the receiver pin for the selector at selOff.
func propPinName(selOff int) string {
	return "_logos_pr" + itoa(selOff)
}

// itoa spells a non-negative offset without importing strconv.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// propTrivialRecv reports whether recv evaluates without observable
// work: a bare variable needs no pin; anything else is pinned once so
// compound updates and write-backs evaluate it exactly once (#132).
func propTrivialRecv(recv syntax.Expr) bool {
	_, ok := syntax.Unparen(recv).(*syntax.Name)
	return ok
}

// propReadIn rewrites property reads in e to Get calls for reads-only
// positions: every property selector becomes `R.Logos_GetF()`, and a
// method call on a property value chains off the Get result.
func (fl *fileLower) propReadIn(e syntax.Expr, c *propCollect) syntax.Expr {
	c.hoist = nil
	return fl.propWalk(e, c)
}

// propWalk rewrites property uses in e, recording reads and hoists in c.
// The tree is mutated in place; text composes later from source offsets,
// which the mutation leaves unchanged. Type positions are never walked:
// callers pass value expressions only, and assertions and composite
// literals keep their type expressions out of the walk.
func (fl *fileLower) propWalk(e syntax.Expr, c *propCollect) syntax.Expr {
	switch e := e.(type) {
	case nil:
		return nil
	case *syntax.Name, *syntax.BasicLit:
		return e
	case *syntax.ParenExpr:
		e.X = fl.propWalk(e.X, c)
		return e
	case *syntax.SelectorExpr:
		if _, pr, sel, ok := fl.propSelOf(e); ok {
			// The outer selector is recorded before descending: a
			// chained interface property (`c.A.HP`) nests needles,
			// which the replacement path substitutes longest-first.
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
		// A generic instantiation argues types, but type spellings
		// never match a property selector (package receivers have no
		// type, and a property set never names a type parameter), so
		// walking the index uniformly is safe; a value index such as
		// x[c.HP] lowers as a read.
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
		// M6-S1 (D8): property reads inside `$"..."` braces lower to
		// Get calls like every other read. Interpolation lowering runs
		// afterwards and drains the needles into its operand texts.
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

// recordPropRead records a Get-inline needle for a property selector.
// The caller renames the selector right after; offsets come from the
// selector name position, which the rename leaves unchanged.
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

// chainPropGet records the read of inner and relinks it as the Get call
// under the outer selector: `R.F.M(args)` becomes `R.Logos_GetF().M(args)`.
func (fl *fileLower) chainPropGet(c *propCollect, pr types2.LogosProp, sel, inner *syntax.SelectorExpr) {
	fl.recordPropRead(c, pr, inner)
	pos := inner.Pos()
	inner.Sel.Value = "Logos_Get" + pr.Name
	sel.X = callOf(pos, inner, nil)
}

// propWalkCall rewrites one call: a method call on a property value
// hoists read-call-write-back into the hoist state, or chains off the
// Get result in reads-only positions; every other call walks its
// function and arguments.
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
		// A field of the property value, a generic method whose type
		// arguments this pass does not respell, or a reads-only
		// position: chain off the Get result. Pointer calls are left
		// for the recheck to reject loudly.
		fl.chainPropGet(c, pr, sel, inner)
		fl.propWalkArgs(e.ArgList, c)
		return e
	}
	fl.hoistPropCall(c, propSite{sel: sel, recv: recv, pr: pr, inner: inner})
	fl.propWalkArgs(e.ArgList, c)
	return e
}

// propSite is one method or field selection on a property value: the
// outer selector, the receiver, the property, and the property selector.
type propSite struct {
	sel   *syntax.SelectorExpr
	recv  syntax.Expr
	pr    types2.LogosProp
	inner *syntax.SelectorExpr
}

// propTemp describes one hoisted property temporary: the value temp,
// the optional receiver pin, and the source selector for text
// composition.
type propTemp struct {
	pos    syntax.Pos
	pin    string
	temp   string
	needle propNeedle
	recv   syntax.Expr
	pr     types2.LogosProp
}

// propCallSel descends a call function through parentheses to the
// outermost selector, reporting whether index expressions (generic
// type arguments) stand between the selector and the call.
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

// propSelIsField reports whether sel selects a struct field rather than
// a method: the check recorded a field selection for it. Unknown
// selectors count as methods, which take the write-back path.
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

// propWalkArgs walks call arguments in evaluation order.
func (fl *fileLower) propWalkArgs(args []syntax.Expr, c *propCollect) {
	for i, a := range args {
		args[i] = fl.propWalk(a, c)
	}
}

// hoistPropCall hoists a method call on a read-write property value:
// `_t := R.Logos_GetF(); _t.M(args); R.Logos_SetF(_t)`, with R pinned
// once when it is not a bare variable (_t is addressable, so no
// explicit & is needed). A read-only property takes the read-call pair
// with no write-back. The selector becomes the temporary, so the call
// itself needs no further rewrite.
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
	// Plain temp receiver: _t is addressable, so _t.M(args) reaches
	// pointer receivers without an explicit &, and prints unambiguously.
	// The temporary takes the receiver's leftmost position: later span
	// lookups descend through it, and the selector position would
	// misplace the statement start.
	site.sel.X = nameOf(leftmostOf(site.recv), t.temp)
	c.needles = append(c.needles, t.needle)
	c.hoist.pres = append(c.hoist.pres, t.prelude()...)
	if !site.pr.ReadOnly && propNeedsWriteBack(fl.pkg.info, site.sel) {
		c.hoist.posts = append(c.hoist.posts, t.writeBack())
	}
	return t.temp
}

// propNeedsWriteBack reports whether a method call on a property value
// must write back: a pointer-receiver call, or any call the check did
// not record. A known value receiver needs no store.
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

// prelude builds the prelude statements for one hoisted property value:
// the receiver pin when the receiver is not trivial, then the temporary
// holding the Get result. Texts compose at statement level.
func (t propTemp) prelude() []propPiece {
	if t.pin == "" {
		return []propPiece{{
			stmt: defineOf(t.pos, []syntax.Expr{nameOf(t.pos, t.temp)}, []syntax.Expr{t.getCall(t.recv)}),
		}}
	}
	return []propPiece{
		{
			stmt: defineOf(t.pos, []syntax.Expr{nameOf(t.pos, t.pin)}, []syntax.Expr{t.recv}),
		},
		{
			stmt: defineOf(t.pos, []syntax.Expr{nameOf(t.pos, t.temp)}, []syntax.Expr{t.getCall(nameOf(t.pos, t.pin))}),
		},
	}
}

// getCall builds `R.Logos_GetF()` for a recorded receiver.
func (t propTemp) getCall(recv syntax.Expr) syntax.Expr {
	return callOf(t.pos, selOf(t.pos, recv, "Logos_Get"+t.pr.Name), nil)
}

// writeBack builds the write-back statement `R.Logos_SetF(_t)`, spelling
// the receiver through the pin when one exists.
func (t propTemp) writeBack() propPiece {
	base := t.recv
	if t.pin != "" {
		base = nameOf(t.pos, t.pin)
	}
	return propPiece{
		stmt: exprStmtOf(t.pos, callOf(t.pos, selOf(t.pos, base, "Logos_Set"+t.pr.Name), []syntax.Expr{nameOf(t.pos, t.temp)})),
	}
}

// lowerPropBlock rewrites one statement list, splicing hoisted property
// temporaries around statements that need them.
func (fl *fileLower) lowerPropBlock(list []syntax.Stmt) []syntax.Stmt {
	var out []syntax.Stmt
	for _, s := range list {
		out = append(out, fl.lowerPropStmt(s)...)
	}
	return out
}

// lowerPropStmt rewrites one statement. Block bodies recurse with their
// own hoist scope; headers (conditions, inits, results) take reads only,
// because a write-back must land before the guarded body runs.
func (fl *fileLower) lowerPropStmt(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		s.List = fl.lowerPropBlock(s.List)
		return []syntax.Stmt{s}
	case *syntax.IfStmt:
		fl.propReadSimple(s.Init)
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
		fl.propReadSimple(s.Init)
		if s.Cond != nil {
			s.Cond = fl.propReadTop(s.Cond)
		}
		fl.propReadSimple(s.Post)
		s.Body.List = fl.lowerPropBlock(s.Body.List)
		return []syntax.Stmt{s}
	case *syntax.SwitchStmt:
		fl.propReadSimple(s.Init)
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

// propReadTop rewrites reads in e and emits their splices immediately,
// for positions without statements.
func (fl *fileLower) propReadTop(e syntax.Expr) syntax.Expr {
	if e == nil {
		return nil
	}
	c := &propCollect{}
	e = fl.propReadIn(e, c)
	fl.emitPropReads(c)
	return e
}

// propReadSimple rewrites reads in a simple statement and emits their
// splices immediately. Assignment targets keep their shape: a property
// target becomes a Get chain, which the recheck rejects loudly.
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

// emitPropReads emits the Get-inline splices of collected reads: the
// selector name becomes `Logos_GetF()`.
func (fl *fileLower) emitPropReads(c *propCollect) {
	for _, n := range c.needles {
		if n.temp != "" {
			continue
		}
		off := n.end - len(n.name)
		fl.add(off, n.end, "Logos_Get"+n.name+"()")
	}
}

// lowerPropAssign rewrites one assignment: a property target becomes a
// Set call, a compound target pins the receiver once, and any other
// shape collects hoists around the preserved statement.
func (fl *fileLower) lowerPropAssign(s *syntax.AssignStmt) []syntax.Stmt {
	lhs := syntax.UnpackListExpr(s.Lhs)
	if s.Rhs == nil {
		// IncDec (`c.HP++`): Op is Add or Sub with no right side.
		if len(lhs) != 1 {
			return []syntax.Stmt{s}
		}
		if recv, pr, ok := fl.propSelRoot(lhs[0]); ok {
			if pr.ReadOnly {
				return fl.lowerPropGeneric(s)
			}
			return fl.lowerPropCompound(s, recv, pr, nil)
		}
		return fl.lowerPropGeneric(s)
	}
	if s.Op != 0 && s.Op != syntax.Def {
		// Compound (`c.HP += v`): one target, one value.
		if len(lhs) != 1 {
			return []syntax.Stmt{s}
		}
		if recv, pr, ok := fl.propSelRoot(lhs[0]); ok {
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
		if recv, pr, ok := fl.propSelRoot(lhs[0]); ok {
			if s.Op != 0 || pr.ReadOnly {
				// `:=` on a property target, or a write to a
				// read-only property, is a check error; keep the
				// shape so the recheck rejects it loudly.
				return fl.lowerPropGeneric(s)
			}
			return fl.lowerPropWrite(s, recv, pr)
		}
	}
	if len(lhs) > 1 {
		roots := make([]int, 0, len(lhs))
		for i, l := range lhs {
			if _, _, ok := fl.propSelRoot(l); ok {
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

// propSelRoot returns the property receiver when e is a property
// selector at its root.
func (fl *fileLower) propSelRoot(e syntax.Expr) (syntax.Expr, types2.LogosProp, bool) {
	recv, pr, _, ok := fl.propSelOf(e)
	if !ok {
		return nil, types2.LogosProp{}, false
	}
	return recv, pr, true
}

// propStmtSpan returns the source span of a simple statement, from the
// leftmost token of first to the statement end.
func (fl *fileLower) propStmtSpan(s syntax.Stmt, first syntax.Expr) (span, bool) {
	sp, ok := fl.platformSpan(first)
	if !ok {
		return span{}, false
	}
	end := stmtEnd(fl.src, sp.start)
	if end <= sp.start {
		return span{}, false
	}
	return span{start: sp.start, end: end}, true
}

// propSpanText returns the lowered text of a value region, draining the
// splices of earlier passes inside it.
func (fl *fileLower) propSpanText(sp span) string {
	return fl.textOf(sp)
}

// propNeedlesNest reports whether any needle strictly contains another.
// Nested selectors cannot take disjoint small splices, so the statement
// takes the replacement path with longest-first substitution.
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

// propSubst represents one needle substitution: the needle's lowered
// text and its replacement.
type propSubst struct {
	needle string
	repl   string
	len    int
}

// propSubstitutions materializes every needle of the statement: reads
// become Get chains over the lowered receiver text, hoisted selectors
// become their temporaries. Callers textOf value pieces first, so the
// receiver texts below match the piece contents.
func (fl *fileLower) propSubstitutions(needles []propNeedle) []propSubst {
	subs := make([]propSubst, 0, len(needles))
	for _, n := range needles {
		if n.temp != "" {
			subs = append(subs, propSubst{needle: fl.textOf(span{start: n.start, end: n.end}), repl: n.temp, len: n.end - n.start})
			continue
		}
		subs = append(subs, propSubst{
			needle: fl.textOf(span{start: n.start, end: n.end}),
			repl:   fl.textOf(n.recv) + ".Logos_Get" + n.name + "()",
			len:    n.end - n.start,
		})
	}
	slices.SortFunc(subs, func(a, b propSubst) int {
		if d := b.len - a.len; d != 0 {
			return d
		}
		return strings.Compare(a.needle, b.needle)
	})
	return subs
}

// propSubstText applies longest-first needle substitution within one
// piece. Identical needles share one replacement, so ReplaceAll is
// exact; needles of other pieces never match.
func propSubstText(text string, subs []propSubst) string {
	for _, s := range subs {
		if s.needle != "" {
			text = strings.ReplaceAll(text, s.needle, s.repl)
		}
	}
	return text
}

// lowerPropGeneric collects hoists through an assignment that keeps its
// shape, replacing the statement only when temporaries hoist around it
// or needles nest.
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
	lhsTexts := make([]string, 0, len(lhs))
	for _, lsp := range lhsSpans {
		lhsTexts = append(lhsTexts, fl.propSpanText(lsp))
	}
	rhsText := ""
	if s.Rhs != nil {
		rhsText = fl.propSpanText(propMustSpan(fl, s.Rhs))
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
	main := strings.Join(lhsTexts, ", ") + " " + op + " " + rhsText
	if s.Rhs == nil {
		// IncDec keeps its postfix spelling.
		main = strings.Join(lhsTexts, ", ") + op + op
	}
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []string{main})
}

// propMustSpan returns the source span of e, stopping the compiler when
// an expression the walker produced has no scannable span. The span
// runs to the expression terminator, so it suits only regions that end
// their statement: right sides, arguments, and whole statements.
func propMustSpan(fl *fileLower, e syntax.Expr) span {
	sp, ok := fl.platformSpan(e)
	if !ok {
		base.Fatalf("logos: cannot span expression at %s", e.Pos().String())
	}
	return sp
}

// propRecvText returns the lowered text of a property receiver: the
// tight receiver span ending at the dot before the selector name, never
// the terminator-bounded platform span, which would swallow the rest of
// the statement.
func (fl *fileLower) propRecvText(sel *syntax.SelectorExpr) string {
	_, recvSp, ok := fl.propSelSpan(sel)
	if !ok {
		base.Fatalf("logos: cannot span property selection at %s", sel.Pos().String())
	}
	return fl.textOf(recvSp)
}

// propLhsSpans returns tight spans for assignment targets: non-final
// targets end at their comma, and the final target ends at the operator,
// because the platform scan would otherwise swallow the operator and
// the right side.
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

// finishPropReplace drains a covering splice, then splices prelude,
// mains, and write-backs over the whole statement. Mains arrive as both
// AST statements and their composed texts; hoist pieces render through
// the syntax printer, whose output the substitution pass already
// lowered inside the pieces.
func (fl *fileLower) finishPropReplace(c *propCollect, sp span, mains []syntax.Stmt, mainTexts []string) []syntax.Stmt {
	_ = fl.textOf(sp)
	subs := fl.propSubstitutions(c.needles)
	var texts []string
	var out []syntax.Stmt
	for _, p := range c.hoist.pres {
		texts = append(texts, propSubstText(fl.propPieceText(p), subs))
		out = append(out, p.stmt)
	}
	for i, m := range mains {
		texts = append(texts, propSubstText(mainTexts[i], subs))
		out = append(out, m)
	}
	for _, p := range c.hoist.posts {
		texts = append(texts, propSubstText(fl.propPieceText(p), subs))
		out = append(out, p.stmt)
	}
	fl.add(sp.start, sp.end, strings.Join(texts, "; "))
	return out
}

// propPieceText renders one hoisted statement for eject text. The nodes
// are synthetic but fully positioned, so the printer output is stable;
// any property needles inside were substituted by the caller.
func (fl *fileLower) propPieceText(p propPiece) string {
	var buf bytes.Buffer
	_, _ = syntax.Fprint(&buf, p.stmt, 0)
	return buf.String()
}

// lowerPropWrite rewrites `R.F = v` to `R.Logos_SetF(v)`, hoisting
// write-backs from the receiver and value around it.
func (fl *fileLower) lowerPropWrite(s *syntax.AssignStmt, recv syntax.Expr, pr types2.LogosProp) []syntax.Stmt {
	// The span precedes the walk: rewriting plants synthetic positions
	// that a later leftmost lookup would mistake for the statement start.
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
	rhsText := fl.propSpanText(propMustSpan(fl, s.Rhs))
	set := exprStmtOf(s.Pos(), callOf(s.Pos(), selOf(s.Pos(), recv, "Logos_Set"+pr.Name), []syntax.Expr{rhs}))
	subs := fl.propSubstitutions(c.needles)
	main := propSubstText(recvText, subs) + ".Logos_Set" + pr.Name + "(" + propSubstText(rhsText, subs) + ")"
	return fl.finishPropReplace(c, sp, []syntax.Stmt{set}, []string{main})
}

// lowerPropCompound rewrites `R.F op= v` to a pinned read-modify-write:
// `_r := R; _r.Logos_SetF(_r.Logos_GetF() op v)`, spelling a bare
// variable receiver directly. IncDec arrives with a nil rhs and an Add
// or Sub operator, and takes a unit step.
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
		c.hoist.pres = append(c.hoist.pres, propPiece{
			stmt: defineOf(pos, []syntax.Expr{nameOf(pos, pin)}, []syntax.Expr{recv}),
		})
	}
	valueText := "1"
	if rhs != nil {
		valueText = fl.propSpanText(propMustSpan(fl, rhs))
	}
	bop := s.Op.String()
	// The receiver node serves once natively: a pinned receiver moves
	// into the prelude, and a trivial (bare-name) receiver is re-spelled
	// fresh for its second use, so no two places share nodes (A4).
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
		ref = pin
	}
	subs := fl.propSubstitutions(c.needles)
	main := propSubstText(ref, subs) + ".Logos_Set" + pr.Name + "(" +
		propSubstText(ref, subs) + ".Logos_Get" + pr.Name + "() " + bop + " " + propSubstText(valueText, subs) + ")"
	return fl.finishPropReplace(c, sp, []syntax.Stmt{set}, []string{main})
}

// mustPropSel returns the root property selector of e, stopping the
// compiler when the shape the assigner matched is gone.
func mustPropSel(fl *fileLower, e syntax.Expr) *syntax.SelectorExpr {
	_, _, sel, ok := fl.propSelOf(e)
	if !ok {
		base.Fatalf("logos: property target lost at %s", e.Pos().String())
	}
	return sel
}

// lowerPropMulti rewrites a multi-assignment with property targets:
// each property target becomes a fresh temporary declared up front,
// and a Set call after the assignment writes it back.
func (fl *fileLower) lowerPropMulti(s *syntax.AssignStmt, lhs []syntax.Expr, roots []int) []syntax.Stmt {
	sps, spOK := fl.propStmtSpan(s, lhs[0])
	c := &propCollect{hoist: &propHoist{}}
	for i, l := range lhs {
		if slices.Contains(roots, i) {
			// A root target lowers structurally in the roots
			// loop below, which also walks its receiver.
			continue
		}
		fl.propWalk(l, c)
	}
	rhsList := syntax.UnpackListExpr(s.Rhs)
	for _, r := range rhsList {
		if call, ok := syntax.Unparen(r).(*syntax.CallExpr); ok && propRawEndsWithQuestion(fl, call) {
			// A direct try as the value: the try lowering owns this
			// statement's text, so this pass leaves it for the
			// recheck to reject loudly.
			return []syntax.Stmt{s}
		}
	}
	s.Rhs = fl.propWalk(s.Rhs, c)
	if !spOK {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	sp := sps
	pos := s.Pos()
	// Target spans precede retargeting: replacing a target plants the
	// temporary's statement position, which a later leftmost lookup
	// would mistake for the target start.
	lhsSpans, ok := fl.propLhsSpans(s, lhs)
	if !ok {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	lhsTexts := make([]string, len(lhs))
	for _, i := range roots {
		l := lhs[i]
		recv, pr, sel, ok := fl.propSelOf(l)
		if !ok || pr.ReadOnly {
			// Lost sources and read-only targets are check
			// errors; keep the shape so the recheck rejects it
			// loudly.
			fl.emitPropReads(c)
			return []syntax.Stmt{s}
		}
		fl.propWalk(recv, c)
		_, recvSp, ok := fl.propSelSpan(sel)
		if !ok {
			fl.emitPropReads(c)
			return []syntax.Stmt{s}
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
			c.hoist.pres = append(c.hoist.pres, propPiece{
				stmt: defineOf(pos, []syntax.Expr{nameOf(pos, pin)}, []syntax.Expr{recv}),
			})
			setBase = nameOf(pos, pin)
		}
		c.hoist.pres = append(c.hoist.pres, propPiece{
			stmt: at(&syntax.DeclStmt{DeclList: []syntax.Decl{
				at(&syntax.VarDecl{NameList: []*syntax.Name{nameOf(pos, temp)}, Type: typ}, pos),
			}}, pos),
		})
		c.hoist.posts = append(c.hoist.posts, propPiece{
			stmt: exprStmtOf(pos, callOf(pos, selOf(pos, setBase, "Logos_Set"+pr.Name), []syntax.Expr{nameOf(pos, temp)})),
		})
		fl.propRetarget(s, i, temp)
		lhsTexts[i] = temp
	}
	for i := range lhs {
		if slices.Contains(roots, i) {
			continue
		}
		lhsTexts[i] = fl.propSpanText(lhsSpans[i])
	}
	rhsText := fl.propSpanText(propMustSpan(fl, s.Rhs))
	subs := fl.propSubstitutions(c.needles)
	for i := range lhsTexts {
		lhsTexts[i] = propSubstText(lhsTexts[i], subs)
	}
	main := strings.Join(lhsTexts, ", ") + " = " + propSubstText(rhsText, subs)
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []string{main})
}

// propRetarget replaces the i-th assignment target with a temporary
// name, detaching the property selector for reuse in a write-back.
func (fl *fileLower) propRetarget(s *syntax.AssignStmt, i int, temp string) {
	le, ok := s.Lhs.(*syntax.ListExpr)
	if !ok {
		base.Fatalf("logos: multi-assign target without a list at %s", s.Pos().String())
	}
	le.ElemList[i] = nameOf(s.Pos(), temp)
}

// propRawEndsWithQuestion reports whether the raw source of a call ends
// with `?`: a direct try the try lowering already claimed. The peek
// reads source bytes without draining any splice.
func propRawEndsWithQuestion(fl *fileLower, call *syntax.CallExpr) bool {
	sp, ok := fl.platformSpan(call)
	if !ok {
		return false
	}
	rest := bytes.TrimSpace(fl.src[sp.start:sp.end])
	return len(rest) > 0 && rest[len(rest)-1] == '?'
}

// lowerPropExprStmt rewrites an expression statement: a method call on a
// property value becomes the read-call-write-back triple (quadrupled
// with a receiver pin); any other call collects hoists from its
// arguments around the preserved call.
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
	sps, spOK := fl.propStmtSpan(s, s.X)
	c := &propCollect{hoist: &propHoist{}}
	temp := fl.hoistPropCall(c, propSite{sel: sel, recv: recv, pr: pr, inner: inner})
	fl.propWalkArgs(call.ArgList, c)
	if !spOK {
		fl.emitPropReads(c)
		return []syntax.Stmt{s}
	}
	sp := sps
	args := make([]string, 0, len(call.ArgList))
	for _, a := range call.ArgList {
		// A trailing `...` scans as part of the last argument, so
		// variadic spreads ride along without a special case.
		args = append(args, fl.propSpanText(propMustSpan(fl, a)))
	}
	subs := fl.propSubstitutions(c.needles)
	for i := range args {
		args[i] = propSubstText(args[i], subs)
	}
	main := temp + "." + sel.Sel.Value + "(" + strings.Join(args, ", ") + ")"
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []string{main})
}

// lowerPropCallArgs collects hoists through a call that keeps its shape,
// replacing the statement only when temporaries hoist around it.
func (fl *fileLower) lowerPropCallArgs(s *syntax.ExprStmt, call *syntax.CallExpr) []syntax.Stmt {
	sps, spOK := fl.propStmtSpan(s, s.X)
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
	sp := sps
	main := fl.propSpanText(sp)
	subs := fl.propSubstitutions(c.needles)
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []string{propSubstText(main, subs)})
}

// lowerPropDeclStmt rewrites a block-level variable declaration,
// hoisting write-back temporaries around the preserved declaration.
func (fl *fileLower) lowerPropDeclStmt(s *syntax.DeclStmt) []syntax.Stmt {
	v0 := propDeclValue(s)
	var sps span
	spOK := v0 != nil
	if spOK {
		sps, spOK = fl.propStmtSpan(s, v0)
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
	sp := sps
	main := fl.propSpanText(sp)
	subs := fl.propSubstitutions(c.needles)
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []string{propSubstText(main, subs)})
}

// propDeclValue returns the first value expression of a declaration
// statement for span purposes, or nil when it declares no values.
func propDeclValue(s *syntax.DeclStmt) syntax.Expr {
	for _, d := range s.DeclList {
		if vd, ok := d.(*syntax.VarDecl); ok && vd.Values != nil {
			return vd.Values
		}
	}
	return nil
}

// lowerPropSend rewrites a send statement, hoisting write-back
// temporaries around the preserved send.
func (fl *fileLower) lowerPropSend(s *syntax.SendStmt) []syntax.Stmt {
	sps, spOK := fl.propStmtSpan(s, s.Chan)
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
	sp := sps
	main := fl.propSpanText(sp)
	subs := fl.propSubstitutions(c.needles)
	return fl.finishPropReplace(c, sp, []syntax.Stmt{s}, []string{propSubstText(main, subs)})
}
