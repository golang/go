// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"reflect"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func tupleTempKey(pos syntax.Pos, i int) string {
	return fmt.Sprintf("_logos_tup%d_%d_%d", pos.Line(), pos.Col(), i)
}

func joinO(parts []otext, sep string) otext {
	var out otext
	for i, p := range parts {
		if i > 0 {
			out = cat(out, gen(sep))
		}
		out = cat(out, p)
	}
	return out
}

func genList(list []string) []otext {
	out := make([]otext, len(list))
	for i, s := range list {
		out[i] = gen(s)
	}
	return out
}

func (fl *fileLower) tupleSplit(pos syntax.Pos, n int) callSplit {
	split, ok := splitCall(fl.src, fl.off(pos))
	if !ok || len(split.args) != n {
		base.Fatalf("logos: tuple at %s has no matching parentheses", pos.String())
	}
	return split
}

func (fl *fileLower) exprListSpan(e syntax.Expr) span {
	elems := syntax.UnpackListExpr(e)
	return span{fl.off(syntax.StartPos(elems[0])), fl.exprEndOf(elems[len(elems)-1])}
}

func (fl *fileLower) lowerTuples() {
	fl.tupleArity = -1
	fl.lowerTupleTypes()
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if e, ok := n.(*syntax.SelectorExpr); ok {
			fl.lowerTupleSelect(e)
		}
		return true
	})
	fl.file.DeclList = fl.lowerTupleDeclList(fl.file.DeclList)
	fl.flushTypeImports()
}

func (fl *fileLower) lowerTupleDeclList(decls []syntax.Decl) []syntax.Decl {
	var out []syntax.Decl
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *syntax.FuncDecl:
			if decl.Body != nil {
				decl.Body.List = fl.lowerTupleBody(len(decl.Type.ResultList), decl.Body.List)
			}
			out = append(out, decl)
		case *syntax.VarDecl:
			out = append(out, fl.lowerTuplePackDecl(decl)...)
		default:
			out = append(out, decl)
		}
	}
	return out
}

func (fl *fileLower) lowerTuplePackDecl(decl *syntax.VarDecl) []syntax.Decl {
	if decl.Values != nil {
		vals := syntax.UnpackListExpr(decl.Values)
		if len(vals) == 1 {
			if call, ok := syntax.Unparen(vals[0]).(*syntax.CallExpr); ok {
				if repl := fl.lowerTuplePackValues(decl, call); repl != nil {
					return repl
				}
			}
			if repl := fl.lowerTupleUnpackDecl(decl, vals[0]); repl != nil {
				return repl
			}
		}
	}
	fl.lowerTupleVarValues(decl)
	return []syntax.Decl{decl}
}

func (fl *fileLower) lowerTupleUnpackDecl(decl *syntax.VarDecl, val syntax.Expr) []syntax.Decl {
	fields, _ := fl.tupleTargetOfType(fl.pkg.info.TypeOf(val))
	if len(decl.NameList) < 2 || len(fields) != len(decl.NameList) {
		return nil
	}
	pos := decl.Pos()
	r := syntax.Unparen(val)
	if lit, ok := r.(*syntax.TupleLit); ok {
		split := fl.tupleSplit(lit.Pos(), len(lit.Elems))
		start, end := extendParens(fl.src, fl.off(syntax.StartPos(lit)), split.end)
		fl.cutTupleLit(start, end, split, otext{}, otext{})
		decl.Values = listOf(pos, slices.Clone(lit.Elems))
		fl.lowerTupleVarValues(decl)
		return []syntax.Decl{decl}
	}
	tn := tupleTempKey(pos, 0)
	last := decl.NameList[len(decl.NameList)-1]
	nameStart := fl.off(decl.NameList[0].Pos())
	namesEnd := fl.off(last.Pos()) + len(last.Value)
	if decl.Type != nil {
		namesEnd = fl.exprEndNested(fl.off(syntax.StartPos(decl.Type)))
	}
	start, end := fl.tupleOperandSpan(r)
	reads, texts := tupleReads(pos, tn, fields)
	next := "\nvar "
	if decl.Group != nil {
		next = "\n\t"
	}
	fl.add(nameStart, start, tn+" = ")
	fl.addT(end, end, cat(gen(next), fl.lazyText(span{nameStart, namesEnd}), gen(" = "+texts)))
	hoist := &syntax.VarDecl{NameList: []*syntax.Name{nameOf(pos, tn)}, Group: decl.Group, Values: val}
	hoist.SetPos(pos)
	decl.Values = listOf(pos, reads)
	return []syntax.Decl{hoist, decl}
}

func (fl *fileLower) tupleOperandSpan(r syntax.Expr) (int, int) {
	rs := fl.exprSpan(r)
	return extendParens(fl.src, rs.start, rs.end)
}

func tupleReads(pos syntax.Pos, base string, fields []string) ([]syntax.Expr, string) {
	reads := make([]syntax.Expr, len(fields))
	texts := make([]string, len(fields))
	for i, f := range fields {
		reads[i] = selOf(pos, nameOf(pos, base), f)
		texts[i] = base + "." + f
	}
	return reads, strings.Join(texts, ", ")
}

func (fl *fileLower) isMultiValueCall(call *syntax.CallExpr) bool {
	_, ok := fl.pkg.info.TypeOf(call).(*types2.Tuple)
	return ok
}

func (fl *fileLower) lowerTuplePackValues(decl *syntax.VarDecl, call *syntax.CallExpr) []syntax.Decl {
	tup, ok := fl.pkg.info.TypeOf(call).(*types2.Tuple)
	if !ok || tup.Len() < 2 || len(decl.NameList) != 1 {
		return nil
	}
	names, types := fl.packDeclTypes(decl, tup)
	if names == nil {
		return nil
	}
	pos := decl.Pos()
	tempNames := make([]string, len(names))
	nameList := make([]*syntax.Name, len(names))
	for i := range names {
		tempNames[i] = tupleTempKey(pos, i)
		nameList[i] = nameOf(pos, tempNames[i])
	}
	callSpan := fl.callBodySpan(call)
	callText := fl.textOf(callSpan)
	comp, compText := fl.tupleComposite(pos, names, types, tempNodes(pos, tempNames), genList(tempNames))
	next := "\nvar "
	if decl.Group != nil {
		next = "\n\t"
	}
	parts := []otext{gen(strings.Join(tempNames, ", ") + " = "), callText, gen(next + decl.NameList[0].Value)}
	if decl.Type != nil {
		typeStart := fl.off(syntax.StartPos(decl.Type))
		parts = append(parts, gen(" "), fl.textOf(span{typeStart, fl.exprEndNested(typeStart)}))
	}
	parts = append(parts, gen(" = "), compText)
	fl.addT(fl.off(syntax.StartPos(decl)), callSpan.end, cat(parts...))
	hoist := &syntax.VarDecl{NameList: nameList}
	hoist.SetPos(pos)
	hoist.Values = call
	decl.Values = comp
	return []syntax.Decl{hoist, decl}
}

func (fl *fileLower) packDeclTypes(decl *syntax.VarDecl, tup *types2.Tuple) ([]string, []types2.Type) {
	if decl.Type != nil {
		if obj, ok := fl.pkg.info.Defs[decl.NameList[0]]; ok {
			if v, ok := obj.(*types2.Var); ok {
				return fl.tupleTargetOfType(v.Type())
			}
		}
		return nil, nil
	}
	if obj, ok := fl.pkg.info.Defs[decl.NameList[0]]; ok {
		if v, ok := obj.(*types2.Var); ok {
			if names, types := fl.tupleTargetOfType(v.Type()); names != nil {
				return names, types
			}
		}
	}
	names := make([]string, tup.Len())
	types := make([]types2.Type, tup.Len())
	for i := range names {
		names[i] = tupleLoweredName(tup.At(i).Name(), i)
		types[i] = tup.At(i).Type()
	}
	return names, types
}

func (fl *fileLower) lowerTupleTypes() {
	var types []*syntax.TupleType
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if t, ok := n.(*syntax.TupleType); ok {
			types = append(types, t)
		}
		return true
	})
	slices.Reverse(types)
	for _, t := range types {
		fl.lowerTupleType(t)
	}
}

func (fl *fileLower) lowerTupleList(list []syntax.Stmt) []syntax.Stmt {
	var out []syntax.Stmt
	for _, s := range list {
		if repl := fl.lowerTupleStmt(s); repl != nil {
			for _, r := range repl {
				fl.lowerTupleNested(r)
				out = append(out, r)
			}
			continue
		}
		if ds, ok := s.(*syntax.DeclStmt); ok {
			ds.DeclList = fl.lowerTupleDeclList(ds.DeclList)
		}
		fl.lowerTupleNested(s)
		out = append(out, s)
		for _, child := range tupleChildLists(s) {
			*child = fl.lowerTupleList(*child)
		}
	}
	return out
}

func (fl *fileLower) lowerTupleBody(results int, list []syntax.Stmt) []syntax.Stmt {
	saved := fl.tupleArity
	fl.tupleArity = results
	defer func() { fl.tupleArity = saved }()
	return fl.lowerTupleList(list)
}

func (fl *fileLower) returnsArity(n int) bool {
	return fl.tupleArity < 0 || fl.tupleArity == n
}

func tupleChildLists(s syntax.Stmt) []*[]syntax.Stmt {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		return []*[]syntax.Stmt{&s.List}
	case *syntax.IfStmt:
		out := []*[]syntax.Stmt{&s.Then.List}
		if els, ok := s.Else.(*syntax.BlockStmt); ok {
			out = append(out, &els.List)
		}
		return out
	case *syntax.ForStmt:
		return []*[]syntax.Stmt{&s.Body.List}
	case *syntax.SwitchStmt:
		var out []*[]syntax.Stmt
		for _, c := range s.Body {
			out = append(out, &c.Body)
		}
		return out
	case *syntax.SelectStmt:
		var out []*[]syntax.Stmt
		for _, c := range s.Body {
			out = append(out, &c.Body)
		}
		return out
	case *syntax.LabeledStmt:
		return tupleChildLists(s.Stmt)
	}
	return nil
}

func (fl *fileLower) lowerTupleNested(s syntax.Node) {
	saved := fl.tupleScope
	fl.tupleScope = s
	defer func() { fl.tupleScope = saved }()
	syntax.Inspect(s, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			n.Body.List = fl.lowerTupleBody(len(n.Type.ResultList), n.Body.List)
			return false
		case *syntax.LambdaExpr:
			if block, ok := n.Body.(*syntax.BlockStmt); ok {
				if sig, ok := fl.pkg.info.TypeOf(n).(*types2.Signature); ok {
					block.List = fl.lowerTupleBody(sig.Results().Len(), block.List)
					return false
				}
			}
		case *syntax.CallExpr:
			fl.lowerTupleCallArgs(n)
		case *syntax.CompositeLit:
			fl.lowerTupleCompositeElems(n)
		case *syntax.TupleLit:
			fl.lowerTupleLitOwn(n)
		case *syntax.VarDecl:
			fl.lowerTupleVarValues(n)
		}
		return true
	})
}

func (fl *fileLower) lowerTupleStmt(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.AssignStmt:
		return fl.lowerTupleAssignStmt(s)
	case *syntax.ReturnStmt:
		return fl.lowerTupleReturnStmt(s)
	}
	return nil
}

func (fl *fileLower) lowerTupleVarValues(decl *syntax.VarDecl) {
	if decl.Values == nil {
		return
	}
	vals := syntax.UnpackListExpr(decl.Values)
	names, types := fl.tupleDeclTarget(decl)
	for _, v := range vals {
		switch v := syntax.Unparen(v).(type) {
		case *syntax.TupleLit:
			fl.lowerTupleLitTargeted(v, names, types)
		case *syntax.CallExpr:
			fl.lowerTupleCallArgs(v)
		case *syntax.CompositeLit:
			fl.lowerTupleCompositeElems(v)
		}
	}
}

func (fl *fileLower) tupleDeclTarget(decl *syntax.VarDecl) ([]string, []types2.Type) {
	if decl.Type == nil || len(decl.NameList) == 0 {
		return nil, nil
	}
	if obj, ok := fl.pkg.info.Defs[decl.NameList[0]]; ok {
		if v, ok := obj.(*types2.Var); ok {
			return fl.tupleTargetOfType(v.Type())
		}
	}
	return nil, nil
}

func (fl *fileLower) tupleTargetOfType(t types2.Type) ([]string, []types2.Type) {
	if t == nil {
		return nil, nil
	}
	t = tupleUnderlying(t)
	if tup, ok := types2.Unalias(t).(*types2.LogosTuple); ok {
		names := make([]string, tup.Len())
		types := make([]types2.Type, tup.Len())
		for i := range names {
			names[i] = tupleLoweredName(tup.Name(i), i)
			types[i] = tup.Elem(i)
		}
		return names, types
	}
	if st, ok := types2.Unalias(t).(*types2.Struct); ok && tupleStructShaped(st) {
		names := make([]string, st.NumFields())
		types := make([]types2.Type, st.NumFields())
		for i := range names {
			names[i] = st.Field(i).Name()
			types[i] = st.Field(i).Type()
		}
		return names, types
	}
	return nil, nil
}

func tupleLoweredName(name string, i int) string {
	if name == "" {
		return fmt.Sprintf("T_%d", i)
	}
	return "T_" + name
}

func tupleStructShaped(st *types2.Struct) bool {
	if st.NumFields() < 2 {
		return false
	}
	for i := range st.NumFields() {
		nm := st.Field(i).Name()
		if len(nm) < 3 || nm[:2] != "T_" {
			return false
		}
	}
	return true
}

func (fl *fileLower) lowerTupleType(t *syntax.TupleType) {
	pos := t.Pos()
	st := &syntax.StructType{}
	st.SetPos(pos)
	split := fl.tupleSplit(pos, len(t.Fields))
	from := fl.off(syntax.StartPos(t))
	sep := "struct{ "
	for i, f := range t.Fields {
		name := ""
		to := split.args[i].start
		if f.Name != nil {
			name = f.Name.Value
			to += len(name)
		}
		lname := tupleLoweredName(name, i)
		st.FieldList = append(st.FieldList, &syntax.Field{Name: nameOf(pos, lname), Type: f.Type})
		fl.add(from, to, sep+lname+" ")
		sep = "; "
		from = split.args[i].end
	}
	fl.add(from, split.end, " }")
	fl.replaceTuple(t, st)
}

func extendParens(src []byte, start, end int) (int, int) {
	for {
		s, e, ok := wrapParenPair(src, start, end)
		if !ok {
			return start, end
		}
		start, end = s, e
	}
}

func wrapParenPair(src []byte, start, end int) (int, int, bool) {
	i := skipBlankBack(src, start)
	if i <= 0 || src[i-1] != '(' {
		return start, end, false
	}
	closer := -1
	scanSegment(src, end, len(src), func(t tok) bool {
		if t.comment {
			return true
		}
		if t.kind == token.RPAREN {
			closer = t.end
		}
		return false
	})
	if closer < 0 {
		return start, end, false
	}
	return i - 1, closer, true
}

func (fl *fileLower) replaceTuple(old, repl syntax.Expr) {
	var root syntax.Node = fl.file
	if fl.tupleScope != nil {
		root = fl.tupleScope
	}
	replaceTupleExpr(root, old, repl)
}

func replaceTupleExpr(root syntax.Node, old, repl syntax.Expr) {
	replaceTupleField(reflect.ValueOf(root), reflect.ValueOf(old), reflect.ValueOf(repl))
}

func replaceTupleField(f, old, repl reflect.Value) bool {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() {
			return false
		}
		if f.Elem().Interface() == old.Interface() {
			if !f.CanSet() {
				return false
			}
			f.Set(repl)
			return true
		}
		if ptr := f.Elem(); ptr.Kind() == reflect.Pointer && !ptr.IsNil() {
			if _, ok := ptr.Interface().(syntax.Node); ok {
				return replaceTupleField(ptr, old, repl)
			}
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			if replaceTupleField(f.Index(i), old, repl) {
				return true
			}
		}
	case reflect.Pointer:
		if f.IsNil() {
			return false
		}
		n, ok := f.Interface().(syntax.Node)
		if !ok || n == nil {
			return false
		}
		if _, leaf := n.(*syntax.BranchStmt); leaf {
			return false
		}
		if f.Interface() == old.Interface() {
			return false
		}
		v := f.Elem()
		if v.Kind() != reflect.Struct {
			return false
		}
		typ := v.Type()
		for i := range v.NumField() {
			if !typ.Field(i).IsExported() && !v.Field(i).CanSet() {
				continue
			}
			if replaceTupleField(v.Field(i), old, repl) {
				return true
			}
		}
	}
	return false
}
func (fl *fileLower) stageReplacement(old, repl syntax.Expr) {
	if fl.stagedReplacements == nil {
		fl.stagedReplacements = make(map[syntax.Expr]syntax.Expr)
	}
	fl.stagedReplacements[old] = repl
}

func (fl *fileLower) flushOuterFirst() {
	fl.flushStaged(false)
}

func (fl *fileLower) flushInnerFirst() {
	fl.flushStaged(true)
}

func (fl *fileLower) flushStaged(innerFirst bool) {
	if len(fl.stagedReplacements) == 0 {
		return
	}
	if innerFirst {
		replaceInnerFirst(fl.file, fl.stagedReplacements)
	} else {
		replaceOuterFirst(fl.file, fl.stagedReplacements)
	}
	fl.stagedReplacements = nil
}

// replaceOuterFirst swaps each staged original in one walk, parents before
// children. A nested original inside a swapped parent leaves with the
// detached subtree, whose replacement text already covers it.
func replaceOuterFirst(root syntax.Node, reps map[syntax.Expr]syntax.Expr) {
	sweepExpr(reflect.ValueOf(root), reps, false)
}

// replaceInnerFirst swaps each staged original children before parents,
// matching sequential inner-first replacement for nested lambdas, whose
// replacements share nodes with the originals.
func replaceInnerFirst(root syntax.Node, reps map[syntax.Expr]syntax.Expr) {
	sweepExpr(reflect.ValueOf(root), reps, true)
}

func sweepExpr(f reflect.Value, reps map[syntax.Expr]syntax.Expr, innerFirst bool) {
	switch f.Kind() {
	case reflect.Interface:
		sweepInterface(f, reps, innerFirst)
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			sweepExpr(f.Index(i), reps, innerFirst)
		}
	case reflect.Pointer:
		sweepPointer(f, reps, innerFirst)
	}
}

func sweepInterface(f reflect.Value, reps map[syntax.Expr]syntax.Expr, innerFirst bool) {
	if f.IsNil() {
		return
	}
	if !innerFirst && swapStaged(f, reps) {
		return
	}
	descendNode(f.Elem(), reps, innerFirst)
	if innerFirst {
		swapStaged(f, reps)
	}
}

func swapStaged(f reflect.Value, reps map[syntax.Expr]syntax.Expr) bool {
	e, ok := f.Elem().Interface().(syntax.Expr)
	if !ok {
		return false
	}
	r, ok := reps[e]
	if !ok {
		return false
	}
	if f.CanSet() {
		f.Set(reflect.ValueOf(r))
	}
	return true
}

func descendNode(v reflect.Value, reps map[syntax.Expr]syntax.Expr, innerFirst bool) {
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	if _, ok := v.Interface().(syntax.Node); !ok {
		return
	}
	sweepExpr(v, reps, innerFirst)
}

func sweepPointer(f reflect.Value, reps map[syntax.Expr]syntax.Expr, innerFirst bool) {
	if f.IsNil() {
		return
	}
	n, ok := f.Interface().(syntax.Node)
	if !ok || n == nil {
		return
	}
	if _, leaf := n.(*syntax.BranchStmt); leaf {
		return
	}
	if !innerFirst {
		if e, ok := n.(syntax.Expr); ok {
			if _, want := reps[e]; want {
				return
			}
		}
	}
	sweepStructFields(f.Elem(), reps, innerFirst)
}

func sweepStructFields(v reflect.Value, reps map[syntax.Expr]syntax.Expr, innerFirst bool) {
	if v.Kind() != reflect.Struct {
		return
	}
	typ := v.Type()
	for i := range v.NumField() {
		if !typ.Field(i).IsExported() && !v.Field(i).CanSet() {
			continue
		}
		sweepExpr(v.Field(i), reps, innerFirst)
	}
}

func tupleUnderlying(t types2.Type) types2.Type {
	t = types2.Unalias(t)
	if n, ok := t.(*types2.Named); ok {
		return n.Underlying()
	}
	return t
}

func tupleSelectionName(t types2.Type, sel string) string {
	if t == nil {
		return sel
	}
	t = compositeUnderlying(t)
	if tup, ok := t.(*types2.LogosTuple); ok {
		for i := range tup.Len() {
			if tup.Name(i) == sel || tupleLoweredName(tup.Name(i), i) == sel {
				return tupleLoweredName(tup.Name(i), i)
			}
		}
	}
	if st, ok := t.(*types2.Struct); ok && tupleStructShaped(st) {
		for i := range st.NumFields() {
			if st.Field(i).Name() == sel {
				return sel
			}
		}
		want := "T_" + sel
		for i := range st.NumFields() {
			if st.Field(i).Name() == want {
				return want
			}
		}
	}
	return sel
}

func (fl *fileLower) lowerTupleSelect(e *syntax.SelectorExpr) {
	fl.renameTupleSel(e, tupleSelectionName(fl.pkg.info.TypeOf(e.X), e.Sel.Value))
}

func (fl *fileLower) renameTupleSel(e *syntax.SelectorExpr, want string) {
	if e.Sel.Value == want {
		return
	}
	off := fl.off(e.Sel.Pos())
	fl.add(off, off+len(e.Sel.Value), want)
	e.Sel.Value = want
}

func (fl *fileLower) lowerTupleAssignStmt(s *syntax.AssignStmt) []syntax.Stmt {
	if s.Op != syntax.Def && s.Op != 0 {
		return nil
	}
	lhs := syntax.UnpackListExpr(s.Lhs)
	rhs := syntax.UnpackListExpr(s.Rhs)
	if len(rhs) != 1 {
		return nil
	}
	r := syntax.Unparen(rhs[0])
	if call, ok := r.(*syntax.CallExpr); ok && fl.isMultiValueCall(call) {
		return fl.lowerTuplePackStmt(s, lhs, call)
	}
	if len(lhs) == 1 {
		if lit, ok := r.(*syntax.TupleLit); ok {
			names, types := fl.tupleAssignTarget(lhs[0], lit)
			fl.lowerTupleLitTargeted(lit, names, types)
		}
		return nil
	}
	return fl.lowerTupleUnpackStmt(s, lhs, r)
}

func (fl *fileLower) tupleAssignTarget(lhs syntax.Expr, lit *syntax.TupleLit) ([]string, []types2.Type) {
	if nm, ok := syntax.Unparen(lhs).(*syntax.Name); ok {
		if obj, ok := fl.pkg.info.Uses[nm]; ok {
			if v, ok := obj.(*types2.Var); ok {
				if names, types := fl.tupleTargetOfType(v.Type()); names != nil {
					return names, types
				}
			}
		}
		if obj, ok := fl.pkg.info.Defs[nm]; ok {
			if v, ok := obj.(*types2.Var); ok {
				if names, types := fl.tupleTargetOfType(v.Type()); names != nil {
					return names, types
				}
			}
		}
		return nil, nil
	}
	if names, types := fl.tupleTargetOfType(fl.pkg.info.TypeOf(lhs)); names != nil {
		return names, types
	}
	return fl.tupleTargetOfType(fl.pkg.info.TypeOf(lit))
}

func (fl *fileLower) lowerTuplePackStmt(s *syntax.AssignStmt, lhs []syntax.Expr, call *syntax.CallExpr) []syntax.Stmt {
	tup, ok := fl.pkg.info.TypeOf(call).(*types2.Tuple)
	if !ok || tup.Len() < 2 || len(lhs) != 1 {
		return nil
	}
	names, types := fl.packTargetTypes(lhs[0], tup)
	if names == nil {
		return nil
	}
	pos := s.Pos()
	temps := make([]syntax.Expr, len(names))
	tempNames := make([]string, len(names))
	for i := range names {
		tempNames[i] = tupleTempKey(pos, i)
		temps[i] = nameOf(pos, tempNames[i])
	}
	callSpan := fl.callBodySpan(call)
	callText := fl.textOf(callSpan)
	comp, compText := fl.tupleComposite(pos, names, types, tempNodes(pos, tempNames), genList(tempNames))
	parts := []otext{
		gen(strings.Join(tempNames, ", ") + " := "),
		callText,
		gen("; "),
		fl.textOf(fl.exprListSpan(s.Lhs)),
	}
	if s.Op == syntax.Def {
		parts = append(parts, gen(" := "))
	} else {
		parts = append(parts, gen(" = "))
	}
	parts = append(parts, compText)
	_, callEnd := extendParens(fl.src, callSpan.start, callSpan.end)
	fl.addT(fl.off(syntax.StartPos(s)), callEnd, cat(parts...))
	hoist := defineOf(pos, temps, []syntax.Expr{call})
	s.Rhs = comp
	return []syntax.Stmt{hoist, s}
}

func (fl *fileLower) packTargetTypes(lhs syntax.Expr, tup *types2.Tuple) ([]string, []types2.Type) {
	if nm, ok := syntax.Unparen(lhs).(*syntax.Name); ok {
		if nm.Value == "_" {
			return nil, nil
		}
		var v *types2.Var
		if obj, ok := fl.pkg.info.Uses[nm]; ok {
			v, _ = obj.(*types2.Var)
		}
		if v == nil {
			if obj, ok := fl.pkg.info.Defs[nm]; ok {
				v, _ = obj.(*types2.Var)
			}
		}
		if v != nil {
			names, types := fl.tupleTargetOfType(v.Type())
			if names == nil {
				return nil, nil
			}
			return names, types
		}
	}
	names := make([]string, tup.Len())
	types := make([]types2.Type, tup.Len())
	for i := range names {
		names[i] = tupleLoweredName(tup.At(i).Name(), i)
		types[i] = tup.At(i).Type()
	}
	return names, types
}

func (fl *fileLower) lowerTupleUnpackStmt(s *syntax.AssignStmt, lhs []syntax.Expr, r syntax.Expr) []syntax.Stmt {
	if lit, ok := r.(*syntax.TupleLit); ok {
		if len(lit.Elems) != len(lhs) {
			return nil
		}
		split := fl.tupleSplit(lit.Pos(), len(lit.Elems))
		start, end := extendParens(fl.src, fl.off(syntax.StartPos(r)), split.end)
		fl.cutTupleLit(start, end, split, otext{}, otext{})
		s.Rhs = listOf(s.Pos(), append([]syntax.Expr{}, lit.Elems...))
		return nil
	}
	elems, _ := fl.tupleTargetOfType(fl.pkg.info.TypeOf(r))
	if len(elems) != len(lhs) {
		return nil
	}
	pos := s.Pos()
	if _, ok := syntax.Unparen(r).(*syntax.Name); ok {
		baseText := fl.textOf(span{fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r))})
		reads := make([]syntax.Expr, len(lhs))
		var texts []otext
		for i := range lhs {
			reads[i] = selOf(pos, r, elems[i])
			texts = append(texts, cat(baseText, gen("."+elems[i])))
		}
		s.Rhs = listOf(pos, reads)
		start, end := extendParens(fl.src, fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r)))
		fl.addT(start, end, joinO(texts, ", "))
		return nil
	}
	tn := tupleTempKey(pos, 0)
	start, end := fl.tupleOperandSpan(r)
	reads, texts := tupleReads(pos, tn, elems)
	op := " = "
	if s.Op == syntax.Def {
		op = " := "
	}
	fl.add(fl.off(syntax.StartPos(s)), start, tn+" := ")
	fl.addT(end, end, cat(gen("; "), fl.lazyText(fl.exprListSpan(s.Lhs)), gen(op+texts)))
	hoist := defineOf(pos, []syntax.Expr{nameOf(pos, tn)}, []syntax.Expr{r})
	s.Rhs = listOf(pos, reads)
	return []syntax.Stmt{hoist, s}
}

func (fl *fileLower) recordArmResult(s *syntax.ReturnStmt, typ types2.Type) {
	if fl.armResult == nil {
		fl.armResult = make(map[*syntax.ReturnStmt]types2.Type)
	}
	fl.armResult[s] = typ
}

func (fl *fileLower) lowerTupleReturnStmt(s *syntax.ReturnStmt) []syntax.Stmt {
	results := syntax.UnpackListExpr(s.Results)
	if len(results) != 1 {
		return nil
	}
	r := syntax.Unparen(results[0])
	if call, ok := r.(*syntax.CallExpr); ok && fl.isMultiValueCall(call) {
		return nil
	}
	pos := s.Pos()
	if lit, ok := r.(*syntax.TupleLit); ok && fl.returnsArity(len(lit.Elems)) {
		split := fl.tupleSplit(lit.Pos(), len(lit.Elems))
		start, end := extendParens(fl.src, fl.off(syntax.StartPos(r)), split.end)
		fl.cutTupleLit(start, end, split, otext{}, otext{})
		s.Results = listOf(pos, append([]syntax.Expr{}, lit.Elems...))
		return nil
	}
	if _, arm := fl.armResult[s]; arm {
		return nil
	}
	elems, _ := fl.tupleTargetOfType(fl.pkg.info.TypeOf(results[0]))
	if len(elems) == 0 || !fl.returnsArity(len(elems)) {
		return nil
	}
	if _, ok := syntax.Unparen(r).(*syntax.Name); ok {
		baseText := fl.textOf(span{fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r))})
		reads := make([]syntax.Expr, len(elems))
		var texts []otext
		for i := range elems {
			reads[i] = selOf(pos, r, elems[i])
			texts = append(texts, cat(baseText, gen("."+elems[i])))
		}
		s.Results = listOf(pos, reads)
		start, end := extendParens(fl.src, fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r)))
		fl.addT(start, end, joinO(texts, ", "))
		return nil
	}
	tn := tupleTempKey(pos, 0)
	start, end := fl.tupleOperandSpan(r)
	reads, texts := tupleReads(pos, tn, elems)
	fl.add(fl.off(syntax.StartPos(s)), start, tn+" := ")
	fl.add(end, end, "; return "+texts)
	hoist := defineOf(pos, []syntax.Expr{nameOf(pos, tn)}, []syntax.Expr{r})
	s.Results = listOf(pos, reads)
	return []syntax.Stmt{hoist, s}
}

func (fl *fileLower) lowerTupleCallArgs(call *syntax.CallExpr) {
	var targets []types2.Type
	if sig := fl.callSig(call); sig != nil {
		targets = fl.callTargets(call, sig)
	}
	for i, a := range call.ArgList {
		lit, ok := syntax.Unparen(a).(*syntax.TupleLit)
		if !ok {
			continue
		}
		var names []string
		var types []types2.Type
		if i < len(targets) && targets[i] != nil {
			names, types = fl.tupleTargetOfType(targets[i])
		}
		fl.lowerTupleLitTargeted(lit, names, types)
	}
}

func (fl *fileLower) lowerTupleCompositeElems(comp *syntax.CompositeLit) {
	ct := fl.pkg.info.TypeOf(comp)
	lower := func(e syntax.Expr, names []string, types []types2.Type) {
		if lit, ok := syntax.Unparen(e).(*syntax.TupleLit); ok {
			fl.lowerTupleLitTargeted(lit, names, types)
		}
	}
	if ct == nil {
		for _, e := range comp.ElemList {
			lower(bareTupleValue(e), nil, nil)
		}
		return
	}
	switch ct := types2.Unalias(ct).(type) {
	case *types2.Slice:
		names, types := fl.tupleTargetOfType(ct.Elem())
		for _, e := range comp.ElemList {
			lower(bareTupleValue(e), names, types)
		}
	case *types2.Array:
		names, types := fl.tupleTargetOfType(ct.Elem())
		for _, e := range comp.ElemList {
			lower(bareTupleValue(e), names, types)
		}
	case *types2.Map:
		knames, ktypes := fl.tupleTargetOfType(ct.Key())
		vnames, vtypes := fl.tupleTargetOfType(ct.Elem())
		for _, e := range comp.ElemList {
			if kv, ok := e.(*syntax.KeyValueExpr); ok {
				lower(kv.Key, knames, ktypes)
				lower(kv.Value, vnames, vtypes)
				continue
			}
			lower(e, vnames, vtypes)
		}
	case *types2.Struct:
		for i, e := range comp.ElemList {
			if kv, ok := e.(*syntax.KeyValueExpr); ok {
				var ft types2.Type
				if key, ok := kv.Key.(*syntax.Name); ok {
					ft = tupleStructField(ct, key.Value)
				}
				names, types := fl.tupleTargetOfType(ft)
				lower(kv.Value, names, types)
				continue
			}
			var ft types2.Type
			if i < ct.NumFields() {
				ft = ct.Field(i).Type()
			}
			names, types := fl.tupleTargetOfType(ft)
			lower(e, names, types)
		}
	default:
		for _, e := range comp.ElemList {
			lower(bareTupleValue(e), nil, nil)
		}
	}
}

func bareTupleValue(e syntax.Expr) syntax.Expr {
	if kv, ok := e.(*syntax.KeyValueExpr); ok {
		return kv.Value
	}
	return e
}

func tupleStructField(st *types2.Struct, name string) types2.Type {
	for i := range st.NumFields() {
		if st.Field(i).Name() == name {
			return st.Field(i).Type()
		}
	}
	return nil
}

func (fl *fileLower) lowerTupleLitOwn(lit *syntax.TupleLit) {
	names, types := fl.tupleTargetOfType(fl.pkg.info.TypeOf(lit))
	fl.lowerTupleLitTargeted(lit, names, types)
}

func (fl *fileLower) lowerTupleLitTargeted(lit *syntax.TupleLit, names []string, types []types2.Type) {
	if names == nil {
		names, types = fl.tupleTargetOfType(fl.pkg.info.TypeOf(lit))
	}
	if names == nil || len(names) != len(lit.Elems) {
		return
	}
	for i, e := range lit.Elems {
		var subNames []string
		var subTypes []types2.Type
		if i < len(types) && types[i] != nil {
			subNames, subTypes = fl.tupleTargetOfType(types[i])
		}
		fl.lowerTupleNestedExprTargeted(e, subNames, subTypes)
	}
	pos := lit.Pos()
	split := fl.tupleSplit(pos, len(lit.Elems))
	comp, head := fl.tupleLiteral(pos, names, types, slices.Clone(lit.Elems))
	fl.cutTupleLit(fl.off(syntax.StartPos(lit)), split.end, split, gen(head), gen("}"))
	fl.stashBodyType(fl.pkg.info.TypeOf(lit), lit, comp)
	fl.replaceTuple(lit, comp)
}

func (fl *fileLower) lowerTupleNestedExprTargeted(e syntax.Expr, names []string, types []types2.Type) {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.TupleLit:
		fl.lowerTupleLitTargeted(e, names, types)
	case *syntax.CallExpr:
		fl.lowerTupleCallArgs(e)
	case *syntax.CompositeLit:
		fl.lowerTupleCompositeElems(e)
	}
}

func (fl *fileLower) tupleComposite(pos syntax.Pos, names []string, types []types2.Type, vals []syntax.Expr, texts []otext) (*syntax.CompositeLit, otext) {
	comp, head := fl.tupleLiteral(pos, names, types, vals)
	return comp, cat(gen(head), joinO(texts, ", "), gen("}"))
}

func (fl *fileLower) tupleLiteral(pos syntax.Pos, names []string, types []types2.Type, vals []syntax.Expr) (*syntax.CompositeLit, string) {
	st, structText := fl.spellTupleStruct(pos, names, types)
	comp := &syntax.CompositeLit{Type: st, Rbrace: pos}
	comp.SetPos(pos)
	comp.ElemList = vals
	return comp, structText + "{"
}

func (fl *fileLower) cutTupleLit(start, end int, split callSplit, head, tail otext) {
	fl.addT(start, split.args[0].start, head)
	fl.addT(split.args[len(split.args)-1].end, end, tail)
}

func (fl *fileLower) spellTupleStruct(pos syntax.Pos, names []string, types []types2.Type) (syntax.Expr, string) {
	var text strings.Builder
	text.WriteString("struct{ ")
	for i := range names {
		if i > 0 {
			text.WriteString("; ")
		}
		spelled, ok := fl.spellLowered(types[i])
		if !ok {
			spelled, ok = fl.spellType(types[i])
			if !ok {
				base.Fatalf("logos: tuple element type %s has no Go spelling", types[i].String())
			}
		}
		text.WriteString(names[i])
		text.WriteString(" ")
		text.WriteString(spelled)
	}
	text.WriteString(" }")
	st, ok := typeExprAt(text.String(), pos)
	if !ok {
		base.Fatalf("logos: cannot respell tuple type %q", text.String())
	}
	return st, text.String()
}
