// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"reflect"
)

// M5-S5 zero-copy conversions (SPEC Sec 2.1, T-201). A `string` becomes
// a `const []byte` through unsafe.Slice(unsafe.StringData(s), len(s)),
// and a `const []byte` whose value is string-derived or a fresh
// composite literal becomes a `string` through
// unsafe.String(unsafe.SliceData(b), len(b)). Every other const view
// keeps the ordinary copying conversion.
//
// Soundness rests on three facts. Only plain names and literals are
// rewritten, so no operand evaluates twice. A rewritten string source
// is immutable, and a rewritten literal backing is fresh and reachable
// only through const views, which the checker keeps mutation-free; a
// view of a mutable slice stays a copy. Provenance meets
// downgrade-only across each function body, so a later Other write
// always wins. Package-level variables never qualify: cross-function
// write order is invisible here.
//
// The parent wires one call into lower.go ahead of eraseConstViews,
// which deletes the `const` markers this pass reads.

type zeroCopy struct {
	fl       *fileLower
	info     *types2.Info
	ok       map[*types2.Var]bool
	seen     map[*types2.Var]bool
	pkgVars  map[*types2.Var]bool
	rewrite  bool
	unsafeOK int
}

// lowerZeroCopy rewrites the zero-copy conversions of one file.
func (fl *fileLower) lowerZeroCopy() {
	z := &zeroCopy{
		fl:      fl,
		info:    fl.pkg.info,
		ok:      make(map[*types2.Var]bool),
		seen:    make(map[*types2.Var]bool),
		pkgVars: make(map[*types2.Var]bool),
	}
	if z.scanFile() {
		z.eraseSurvivingConstFuns()
		return
	}
	for _, decl := range fl.file.DeclList {
		if vd, ok := decl.(*syntax.VarDecl); ok {
			for _, name := range vd.NameList {
				if v, ok := z.info.Defs[name].(*types2.Var); ok {
					z.pkgVars[v] = true
				}
			}
		}
	}
	syntax.Inspect(fl.file, func(n syntax.Node) bool { z.collect(n); return true })
	syntax.Inspect(fl.file, func(n syntax.Node) bool { return z.visit(n) })
	if z.rewrite {
		z.ensureUnsafeImport()
	}
	z.eraseSurvivingConstFuns()
}

// eraseSurvivingConstFuns erases every ConstType conversion function
// the rewrite left behind: non-bytes targets, operands the unsafe
// forms cannot take, and shadow-pruned regions all keep the ordinary
// copying conversion. eraseConstViews never visits CallExpr funs, so
// without this the recheck would reject the admitted conversion.
func (z *zeroCopy) eraseSurvivingConstFuns() {
	syntax.Inspect(z.fl.file, func(n syntax.Node) bool {
		if call, ok := n.(*syntax.CallExpr); ok {
			if _, ok := syntax.Unparen(call.Fun).(*syntax.ConstType); ok {
				call.Fun = z.eraseConstFun(call.Fun)
			}
		}
		return true
	})
}

// eraseConstFun replaces ConstType layers with their elements,
// splicing each `const ` prefix out of the eject text.
func (z *zeroCopy) eraseConstFun(e syntax.Expr) syntax.Expr {
	for {
		c, ok := e.(*syntax.ConstType)
		if !ok {
			return e
		}
		start := z.fl.off(c.Pos())
		end := z.fl.off(c.Elem.Pos())
		if end > start {
			z.fl.add(start, end, "")
		}
		e = c.Elem
	}
}

// visit runs the rewrite step with shadowing-safe function pruning.
func (z *zeroCopy) visit(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.FuncDecl:
		if n.Body != nil && z.funcShadowed(n.Body) {
			return false
		}
	case *syntax.FuncLit:
		if z.funcShadowed(n.Body) {
			return false
		}
	}
	z.rewriteNode(n)
	return true
}

// scanFile rejects files where synthesized unsafe or len names collide.
func (z *zeroCopy) scanFile() bool {
	bad := false
	for _, decl := range z.fl.file.DeclList {
		switch n := decl.(type) {
		case *syntax.VarDecl:
			for _, name := range n.NameList {
				bad = bad || name.Value == "unsafe" || name.Value == "len"
			}
		case *syntax.ConstDecl:
			for _, name := range n.NameList {
				bad = bad || name.Value == "unsafe" || name.Value == "len"
			}
		case *syntax.FuncDecl:
			if n.Recv == nil {
				bad = bad || n.Name.Value == "unsafe" || n.Name.Value == "len"
			}
		case *syntax.TypeDecl:
			bad = bad || n.Name.Value == "unsafe" || n.Name.Value == "len"
		case *syntax.ImportDecl:
			if n.LocalPkgName != nil && (n.LocalPkgName.Value == "unsafe" || n.LocalPkgName.Value == "len") {
				return true
			}
			if n.Path != nil && n.Path.Value == `"unsafe"` {
				if n.LocalPkgName == nil || n.LocalPkgName.Value == "unsafe" {
					z.unsafeOK = 2
				} else {
					return true
				}
			}
		}
	}
	return bad
}

// funcShadowed reports bindings of unsafe or len inside one body.
func (z *zeroCopy) funcShadowed(body *syntax.BlockStmt) bool {
	shadowed := false
	syntax.Inspect(body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.VarDecl:
			for _, name := range n.NameList {
				shadowed = shadowed || name.Value == "unsafe" || name.Value == "len"
			}
		case *syntax.AssignStmt:
			if n.Op == syntax.Def {
				for _, lhs := range unlistExpr(n.Lhs) {
					if name, ok := syntax.Unparen(lhs).(*syntax.Name); ok {
						shadowed = shadowed || name.Value == "unsafe" || name.Value == "len"
					}
				}
			}
		case *syntax.FuncType:
			for _, list := range [][]*syntax.Field{n.ParamList, n.ResultList} {
				for _, f := range list {
					if f.Name != nil {
						shadowed = shadowed || f.Name.Value == "unsafe" || f.Name.Value == "len"
					}
				}
			}
		case *syntax.TypeDecl:
			shadowed = shadowed || n.Name.Value == "unsafe" || n.Name.Value == "len"
		}
		return !shadowed
	})
	return shadowed
}

// collect records const-view variable provenance, downgrade-only.
func (z *zeroCopy) collect(n syntax.Node) {
	switch n := n.(type) {
	case *syntax.VarDecl:
		if len(n.NameList) != 1 {
			return
		}
		v, ok := z.info.Defs[n.NameList[0]].(*types2.Var)
		if !ok || !z.isViewBytes(v.Type()) {
			return
		}
		z.setVar(v, z.provOf(n.Values))
	case *syntax.AssignStmt:
		if n.Op != 0 && n.Op != syntax.Def {
			return
		}
		lhs := unlistExpr(n.Lhs)
		vals := unlistExpr(n.Rhs)
		if len(lhs) != 1 {
			return
		}
		name, ok := syntax.Unparen(lhs[0]).(*syntax.Name)
		if !ok {
			return
		}
		var v *types2.Var
		if n.Op == syntax.Def {
			v, _ = z.info.Defs[name].(*types2.Var)
		} else {
			v, _ = z.info.ObjectOf(name).(*types2.Var)
		}
		if v == nil || !z.isViewBytes(v.Type()) {
			return
		}
		var rhs syntax.Expr
		if len(vals) == 1 {
			rhs = vals[0]
		}
		z.setVar(v, z.provOf(rhs))
	}
}

// setVar meets one write into downgrade-only provenance.
func (z *zeroCopy) setVar(v *types2.Var, p bool) {
	if z.pkgVars[v] {
		return
	}
	if !z.seen[v] {
		z.seen[v] = true
		z.ok[v] = p
		return
	}
	z.ok[v] = z.ok[v] && p
}

// provOf classifies one const-view value for zero-copy-back.
func (z *zeroCopy) provOf(e syntax.Expr) bool {
	if e == nil {
		return false
	}
	switch n := syntax.Unparen(e).(type) {
	case *syntax.CallExpr:
		return z.convProv(n)
	case *syntax.CompositeLit:
		return z.isByteSlice(z.info.TypeOf(n))
	case *syntax.Name:
		return z.varProv(n)
	}
	return false
}

// convProv classifies a conversion result.
func (z *zeroCopy) convProv(call *syntax.CallExpr) bool {
	switch syntax.Unparen(call.Fun).(type) {
	case *syntax.ConstType:
		if !z.isViewBytesCall(call) || len(call.ArgList) != 1 {
			return false
		}
		return z.stringOrVarProv(call.ArgList[0])
	case *syntax.SliceType:
		if !z.isByteSlice(z.info.TypeOf(call)) || len(call.ArgList) != 1 {
			return false
		}
		return z.stringOrVarProv(call.ArgList[0])
	}
	return false
}

// stringOrVarProv accepts string-derived values and known-good views.
func (z *zeroCopy) stringOrVarProv(e syntax.Expr) bool {
	a := syntax.Unparen(e)
	if z.isStringType(z.info.TypeOf(a)) {
		return true
	}
	if name, ok := a.(*syntax.Name); ok {
		return z.varProv(name)
	}
	return false
}

// varProv reads function-local provenance; package variables fail closed.
func (z *zeroCopy) varProv(name *syntax.Name) bool {
	v, ok := z.info.ObjectOf(name).(*types2.Var)
	if !ok || z.pkgVars[v] || !z.isViewBytes(v.Type()) {
		return false
	}
	return z.ok[v]
}

// rewriteNode rewrites destination-gated plain conversions, then every
// nested conversion through the field pass.
func (z *zeroCopy) rewriteNode(n syntax.Node) {
	switch n := n.(type) {
	case *syntax.VarDecl:
		if len(n.NameList) == 1 {
			if v, ok := z.info.Defs[n.NameList[0]].(*types2.Var); ok && z.isViewBytes(v.Type()) {
				n.Values = z.rewritePlainConv(n.Values)
			}
		}
	case *syntax.AssignStmt:
		if n.Op != 0 {
			break
		}
		lhs := unlistExpr(n.Lhs)
		vals := unlistExpr(n.Rhs)
		if len(lhs) != 1 || len(vals) != 1 {
			break
		}
		name, ok := syntax.Unparen(lhs[0]).(*syntax.Name)
		if !ok {
			break
		}
		if v, ok := z.info.ObjectOf(name).(*types2.Var); ok && z.isViewBytes(v.Type()) {
			n.Rhs = z.rewritePlainConv(n.Rhs)
		}
	}
	z.rewriteFields(n)
}

// rewritePlainConv rewrites one []byte(string) value to its unsafe form.
func (z *zeroCopy) rewritePlainConv(e syntax.Expr) syntax.Expr {
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok {
		return e
	}
	if _, ok := syntax.Unparen(call.Fun).(*syntax.SliceType); !ok {
		return e
	}
	if !z.isByteSlice(z.info.TypeOf(call)) || len(call.ArgList) != 1 {
		return e
	}
	arg := syntax.Unparen(call.ArgList[0])
	if !z.singleEval(arg) || !z.isStringType(z.info.TypeOf(arg)) {
		return e
	}
	return z.sliceOfString(call, arg)
}

// rewriteFields rewrites the conversions held directly by n.
func (z *zeroCopy) rewriteFields(n syntax.Node) {
	rv := reflect.ValueOf(n)
	if rv.Kind() != reflect.Pointer || rv.IsNil() {
		return
	}
	rv = rv.Elem()
	if rv.Kind() != reflect.Struct {
		return
	}
	rt := rv.Type()
	for i := range rt.NumField() {
		fv := rv.Field(i)
		if !fv.CanSet() {
			continue
		}
		ft := rt.Field(i).Type
		switch {
		case ft == reflect.TypeFor[syntax.Expr]():
			if e, _ := fv.Interface().(syntax.Expr); e != nil {
				if ne, ok := z.rewriteCall(e); ok {
					fv.Set(reflect.ValueOf(ne))
				}
			}
		case (ft.Kind() == reflect.Slice || ft.Kind() == reflect.Array) && ft.Elem() == reflect.TypeFor[syntax.Expr]():
			for j := range fv.Len() {
				ev := fv.Index(j)
				if e, _ := ev.Interface().(syntax.Expr); e != nil {
					if ne, ok := z.rewriteCall(e); ok {
						ev.Set(reflect.ValueOf(ne))
					}
				}
			}
		case ft == reflect.TypeFor[*syntax.CallExpr]():
			if e, _ := fv.Interface().(*syntax.CallExpr); e != nil {
				if ne, ok := z.rewriteCall(e); ok {
					fv.Set(reflect.ValueOf(ne))
				}
			}
		}
	}
}

// rewriteCall rewrites one conversion to its unsafe form when the
// checker proved its shape and provenance.
func (z *zeroCopy) rewriteCall(e syntax.Expr) (syntax.Expr, bool) {
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok {
		return nil, false
	}
	switch fun := syntax.Unparen(call.Fun).(type) {
	case *syntax.ConstType:
		if !z.isViewBytesCall(call) || len(call.ArgList) != 1 {
			return nil, false
		}
		arg := syntax.Unparen(call.ArgList[0])
		if !z.singleEval(arg) || !z.isStringType(z.info.TypeOf(arg)) {
			return nil, false
		}
		return z.sliceOfString(call, arg), true
	case *syntax.Name:
		if fun.Value != "string" || !z.bareStringName(fun) {
			return nil, false
		}
		if !z.isStringType(z.info.TypeOf(call)) || len(call.ArgList) != 1 {
			return nil, false
		}
		arg, ok := syntax.Unparen(call.ArgList[0]).(*syntax.Name)
		if !ok || !z.varProv(arg) {
			return nil, false
		}
		return z.stringOfSlice(call, arg), true
	}
	return nil, false
}

// bareStringName accepts the builtin conversion and the universe type
// use, and rejects a shadowed string value.
func (z *zeroCopy) bareStringName(name *syntax.Name) bool {
	if obj := z.info.ObjectOf(name); obj != nil {
		_, ok := obj.(*types2.TypeName)
		return ok
	}
	return true
}

// singleEval allows operands that read twice without effects.
func (z *zeroCopy) singleEval(e syntax.Expr) bool {
	switch e.(type) {
	case *syntax.Name, *syntax.BasicLit:
		return true
	}
	return false
}

// sliceOfString builds unsafe.Slice(unsafe.StringData(arg), len(arg)).
func (z *zeroCopy) sliceOfString(call *syntax.CallExpr, arg syntax.Expr) syntax.Expr {
	return z.unsafeConvert(call, arg, "Slice", "StringData", "unsafe.Slice(unsafe.StringData(%s), len(%s))")
}

// stringOfSlice builds unsafe.String(unsafe.SliceData(arg), len(arg)).
func (z *zeroCopy) stringOfSlice(call *syntax.CallExpr, arg syntax.Expr) syntax.Expr {
	return z.unsafeConvert(call, arg, "String", "SliceData", "unsafe.String(unsafe.SliceData(%s), len(%s))")
}

// unsafeConvert builds one unsafe conversion call and records its eject
// splice. The original operand fills the data slot; a clone fills len.
func (z *zeroCopy) unsafeConvert(call *syntax.CallExpr, arg syntax.Expr, outer, inner, form string) syntax.Expr {
	pos := call.Pos()
	start := z.fl.off(syntax.Unparen(call.Fun).Pos())
	end := z.fl.exprEnd(start)
	argStart := z.fl.off(arg.Pos())
	argEnd := z.fl.exprEnd(argStart)
	text := z.fl.textOf(span{start: argStart, end: argEnd})
	out := callOf(pos,
		selOf(pos, nameOf(pos, "unsafe"), outer),
		[]syntax.Expr{
			callOf(pos, selOf(pos, nameOf(pos, "unsafe"), inner), []syntax.Expr{arg}),
			callOf(pos, nameOf(pos, "len"), []syntax.Expr{z.dupOf(pos, arg)}),
		})
	z.fl.add(start, end, fmt.Sprintf(form, text, text))
	z.rewrite = true
	return out
}

// dupOf clones a single-evaluation operand for its second read.
func (z *zeroCopy) dupOf(pos syntax.Pos, arg syntax.Expr) syntax.Expr {
	switch arg := arg.(type) {
	case *syntax.Name:
		return nameOf(pos, arg.Value)
	case *syntax.BasicLit:
		return litOf(pos, arg.Value, arg.Kind)
	}
	return nameOf(pos, "_")
}

// ensureUnsafeImport prepends the unsafe import the rewrites need.
func (z *zeroCopy) ensureUnsafeImport() {
	if z.unsafeOK == 2 {
		return
	}
	pos := z.fl.file.Pos()
	z.fl.file.DeclList = append([]syntax.Decl{
		at(&syntax.ImportDecl{Path: litOf(pos, `"unsafe"`, syntax.StringLit)}, pos),
	}, z.fl.file.DeclList...)
	off := z.fl.off(z.fl.file.PkgName.Pos())
	ins := -1
	scanSegment(z.fl.src, off, len(z.fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		ins = t.end
		return false
	})
	if ins < 0 {
		ins = off
	}
	z.fl.add(ins, ins, "\nimport \"unsafe\"")
}

// isViewBytesCall reports a recorded const-view-of-bytes conversion.
func (z *zeroCopy) isViewBytesCall(call *syntax.CallExpr) bool {
	return z.isViewBytes(z.info.TypeOf(call))
}

// isViewBytes reports const views of byte slices.
func (z *zeroCopy) isViewBytes(t types2.Type) bool {
	c := constAsView(t)
	return c != nil && z.isByteSlice(c.Elem)
}

// isByteSlice reports []byte.
func (z *zeroCopy) isByteSlice(t types2.Type) bool {
	s, ok := types2.Unalias(t).(*types2.Slice)
	if !ok {
		return false
	}
	b, ok := types2.Unalias(s.Elem()).(*types2.Basic)
	return ok && b.Kind() == types2.Uint8
}

// isStringType reports string types, including untyped constants.
func (z *zeroCopy) isStringType(t types2.Type) bool {
	b, ok := types2.Unalias(t).(*types2.Basic)
	return ok && (b.Kind() == types2.String || b.Kind() == types2.UntypedString)
}

// unlistExpr unpacks a possibly-listed expression side.
func unlistExpr(e syntax.Expr) []syntax.Expr {
	if e == nil {
		return nil
	}
	if l, ok := syntax.Unparen(e).(*syntax.ListExpr); ok {
		return l.ElemList
	}
	return []syntax.Expr{e}
}
