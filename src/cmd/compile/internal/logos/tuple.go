// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"reflect"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers the M6-S5 tuple nodes (SPEC Sec 2.4, T-204.1):
// TupleType becomes the struct with exported `T_` fields, TupleLit
// becomes a composite literal of that struct, `t.x` selections become
// `t.T_x`, and pack/unpack assignments bind calls and tuple values
// through field accesses. Every rewrite builds the native tree and the
// eject text together, through small splices over exact token spans.
//
// The pass runs after the statement lowering, so `?` and `errdefer`
// templates never see tuple forms. Replacement spans never contain a
// later splice: selectors and right-hand sides splice exact tokens,
// struct spellings compose earlier splices through textOf, and generated
// statements hold no logos nodes.

// tupleTempKey builds a per-site temporary name from a position.
func tupleTempKey(pos syntax.Pos, i int) string {
	return fmt.Sprintf("_logos_tup%d_%d_%d", pos.Line(), pos.Col(), i)
}

// tupleCallSpan reports the source span of a call, including its
// argument parentheses (the node end sits at the function).
func (fl *fileLower) tupleCallSpan(call *syntax.CallExpr) span {
	start := fl.off(leftmostOf(call))
	_, rparen := fl.findCallParens(fl.off(syntax.EndPos(syntax.Unparen(call.Fun))))
	return span{start, rparen + 1}
}

// lowerTuples rewrites every tuple form of one file.
func (fl *fileLower) lowerTuples() {
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

// lowerTupleDeclList lowers tuple forms in declarations, splicing
// pack hoists in place.
func (fl *fileLower) lowerTupleDeclList(decls []syntax.Decl) []syntax.Decl {
	var out []syntax.Decl
	for _, decl := range decls {
		switch decl := decl.(type) {
		case *syntax.FuncDecl:
			if decl.Body != nil {
				decl.Body.List = fl.lowerTupleList(decl.Body.List)
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

// lowerTuplePackDecl lowers pack shapes in one var declaration,
// returning the replacement declarations.
func (fl *fileLower) lowerTuplePackDecl(decl *syntax.VarDecl) []syntax.Decl {
	if decl.Values != nil {
		vals := syntax.UnpackListExpr(decl.Values)
		if len(vals) == 1 {
			if call, ok := syntax.Unparen(vals[0]).(*syntax.CallExpr); ok {
				if repl := fl.lowerTuplePackValues(decl, call); repl != nil {
					return repl
				}
			}
		}
	}
	fl.lowerTupleVarValues(decl)
	return []syntax.Decl{decl}
}

// lowerTuplePackValues lowers `var t T = f()` and `var t = f()` over a
// multi-value call into a temporary declaration plus the composite.
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
	callSpan := fl.tupleCallSpan(call)
	callText := fl.textOf(callSpan)
	comp, compText := fl.tupleComposite(pos, names, types, tempNamesAsExpr(pos, tempNames), tempNames)
	var text strings.Builder
	text.WriteString("var " + strings.Join(tempNames, ", ") + " = " + callText + "\n")
	text.WriteString("var " + decl.NameList[0].Value)
	if decl.Type != nil {
		text.WriteString(" " + fl.textOf(span{fl.off(syntax.StartPos(decl.Type)), fl.off(syntax.EndPos(decl.Type))}))
	}
	text.WriteString(" = " + compText)
	fl.add(fl.off(syntax.StartPos(decl)), callSpan.end, text.String())
	hoist := &syntax.VarDecl{NameList: nameList}
	hoist.SetPos(pos)
	hoist.Values = call
	decl.Values = comp
	return []syntax.Decl{hoist, decl}
}

// packDeclTypes reports the lowered names and element types for a var
// pack target: the declared type, else the call result names and types.
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

// lowerTupleTypes rewrites every TupleType to its struct, innermost
// first so each eject splice composes the ones inside it through textOf.
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

// lowerTupleList lowers tuple statements in list, splicing replacement
// statement lists in place, then recurses into child bodies including
// function literals (temporary names are file-unique).
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

// tupleChildLists reports the statement lists nested directly under s.
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

// lowerTupleNested lowers tuple expressions under s: call arguments,
// composite elements, stray literals, and local declarations.
func (fl *fileLower) lowerTupleNested(s syntax.Stmt) {
	syntax.Inspect(s, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			n.Body.List = fl.lowerTupleList(n.Body.List)
			return false
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

// lowerTupleStmt lowers tuple shapes in one statement, returning the
// replacement list or nil to keep the statement.
func (fl *fileLower) lowerTupleStmt(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.AssignStmt:
		return fl.lowerTupleAssignStmt(s)
	case *syntax.ReturnStmt:
		return fl.lowerTupleReturnStmt(s)
	}
	return nil
}

// lowerTupleVarValues lowers TupleLit values of one declaration,
// through the declared type or the literal's own type.
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

// tupleDeclTarget reports the declared tuple target of a var
// declaration as lowered field names and element types, through the
// declared variable object.
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

// tupleTargetOfType maps a checked type to lowered tuple field names
// and element types, or nil when it is not a tuple.
func (fl *fileLower) tupleTargetOfType(t types2.Type) ([]string, []types2.Type) {
	if t == nil {
		return nil, nil
	}
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

// tupleLoweredName spells the lowered field of a source element name.
func tupleLoweredName(name string, i int) string {
	if name == "" {
		return fmt.Sprintf("T_%d", i)
	}
	return "T_" + name
}

// tupleStructShaped reports whether st is the lowered tuple shape.
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

// lowerTupleType rewrites one TupleType to its struct. Field types move
// into the struct, keeping their checked identity; the eject text
// composes earlier splices inside them.
func (fl *fileLower) lowerTupleType(t *syntax.TupleType) {
	pos := t.Pos()
	st := &syntax.StructType{}
	st.SetPos(pos)
	var text strings.Builder
	text.WriteString("struct{ ")
	for i, f := range t.Fields {
		name := ""
		typ := f.Type
		if f.Name != nil {
			name = f.Name.Value
		}
		lname := tupleLoweredName(name, i)
		st.FieldList = append(st.FieldList, &syntax.Field{Name: nameOf(pos, lname), Type: typ})
		if i > 0 {
			text.WriteString("; ")
		}
		tspan := span{fl.off(syntax.StartPos(typ)), fl.off(syntax.EndPos(typ))}
		text.WriteString(lname + " " + fl.textOf(tspan))
	}
	text.WriteString(" }")
	fl.add(fl.off(syntax.StartPos(t)), extendPastComma(fl.src, fl.off(syntax.EndPos(t))), text.String())
	replaceTupleExpr(fl.file, t, st)
}

// extendPastComma extends end past a trailing comma, its trivia, and
// the closing paren of the tuple form.
func extendPastComma(src []byte, end int) int {
	i := end
	for i < len(src) {
		switch src[i] {
		case ' ', '\t', '\n', '\r':
			i++
		case ',':
			i++
			for i < len(src) {
				switch src[i] {
				case ' ', '\t', '\n', '\r':
					i++
				case '/':
					i = skipComment(src, i)
					if i < 0 {
						return end
					}
				case ')':
					return i + 1
				default:
					return end
				}
			}
			return end
		case '/':
			i = skipComment(src, i)
			if i < 0 {
				return end
			}
		case ')':
			return i + 1
		default:
			return end
		}
	}
	return end
}

// extendParens extends [start, end) past every level of directly
// adjacent redundant parentheses left behind by dropped groupings.
func extendParens(src []byte, start, end int) (int, int) {
	for {
		s, e, ok := wrapParenPair(src, start, end)
		if !ok {
			return start, end
		}
		start, end = s, e
	}
}

// wrapParenPair widens [start, end) by one directly adjacent `(` ... `)`
// pair, blanks allowed between each paren and the span, and reports
// whether such a pair exists.
func wrapParenPair(src []byte, start, end int) (int, int, bool) {
	i := start
	for i > 0 {
		c := src[i-1]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			i--
			continue
		}
		break
	}
	if i <= 0 || src[i-1] != '(' {
		return start, end, false
	}
	j := end
	for j < len(src) {
		c := src[j]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			j++
			continue
		}
		break
	}
	if j >= len(src) || src[j] != ')' {
		return start, end, false
	}
	return i - 1, j + 1, true
}

// skipComment skips a comment at i, or reports -1.
func skipComment(src []byte, i int) int {
	if i+1 < len(src) && src[i+1] == '/' {
		for i < len(src) && src[i] != '\n' {
			i++
		}
		return i
	}
	if i+1 < len(src) && src[i+1] == '*' {
		j := i + 2
		for j+1 < len(src) && (src[j] != '*' || src[j+1] != '/') {
			j++
		}
		return j + 2
	}
	return -1
}

// replaceTupleExpr swaps the tuple node old for repl wherever the file
// holds it directly.
func replaceTupleExpr(root syntax.Node, old, repl syntax.Expr) {
	replaceTupleField(reflect.ValueOf(root), reflect.ValueOf(old), reflect.ValueOf(repl))
}

// replaceTupleField swaps old for repl in interfaces, slices, and child
// nodes reachable through f. A branch statement is a leaf: its Target
// points to the enclosing loop, switch, or select, or to a labeled
// statement elsewhere in the tree, so following it walks a cycle from
// every `break` and `continue`, as syntax.Inspect notes. It holds no
// expression, so no replacement is lost.
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

// lowerTupleSelect rewrites `t.x` over a tuple to `t.T_x`.
func (fl *fileLower) lowerTupleSelect(e *syntax.SelectorExpr) {
	base := fl.pkg.info.TypeOf(e.X)
	if base == nil {
		return
	}
	sel := e.Sel.Value
	if tup, ok := types2.Unalias(base).(*types2.LogosTuple); ok {
		for i := range tup.Len() {
			if tup.Name(i) == sel || tupleLoweredName(tup.Name(i), i) == sel {
				fl.renameTupleSel(e, tupleLoweredName(tup.Name(i), i))
				return
			}
		}
		return
	}
	if st, ok := types2.Unalias(base).(*types2.Struct); ok && tupleStructShaped(st) {
		for i := range st.NumFields() {
			if st.Field(i).Name() == sel {
				return
			}
		}
		want := "T_" + sel
		for i := range st.NumFields() {
			if st.Field(i).Name() == want {
				fl.renameTupleSel(e, want)
				return
			}
		}
	}
}

// renameTupleSel renames the selector to the lowered field.
func (fl *fileLower) renameTupleSel(e *syntax.SelectorExpr, want string) {
	if e.Sel.Value == want {
		return
	}
	off := fl.off(e.Sel.Pos())
	fl.add(off, off+len(e.Sel.Value), want)
	e.Sel.Value = want
}

// lowerTupleAssignStmt lowers tuple shapes in one assignment,
// returning the replacement list or nil to keep the statement.
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
	if call, ok := r.(*syntax.CallExpr); ok {
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

// tupleAssignTarget reports the lowered names and types for a
// single-target tuple literal: the declared variable type, else the
// literal's own type.
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

// lowerTuplePackStmt lowers `t := f()` and `t = f()` over a
// multi-value call into temporaries plus a composite.
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
	callSpan := fl.tupleCallSpan(call)
	callText := fl.textOf(callSpan)
	comp, compText := fl.tupleComposite(pos, names, types, tempNamesAsExpr(pos, tempNames), tempNames)
	var text strings.Builder
	text.WriteString(strings.Join(tempNames, ", ") + " := " + callText + "; ")
	text.WriteString(fl.textOf(span{fl.off(syntax.StartPos(s.Lhs)), fl.off(syntax.EndPos(s.Lhs))}))
	if s.Op == syntax.Def {
		text.WriteString(" := ")
	} else {
		text.WriteString(" = ")
	}
	text.WriteString(compText)
	_, callEnd := extendParens(fl.src, callSpan.start, callSpan.end)
	fl.add(fl.off(syntax.StartPos(s)), callEnd, text.String())
	hoist := defineOf(pos, temps, []syntax.Expr{call})
	s.Rhs = comp
	return []syntax.Stmt{hoist, s}
}

// tempNamesAsExpr builds name expressions for temps.
func tempNamesAsExpr(pos syntax.Pos, temps []string) []syntax.Expr {
	out := make([]syntax.Expr, len(temps))
	for i, tn := range temps {
		out[i] = nameOf(pos, tn)
	}
	return out
}

// packTargetTypes reports the lowered names and element types for a
// pack target: the declared variable type, else the call result names
// and types. The blank identifier stays an upstream mismatch.
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

// lowerTupleUnpackStmt lowers `a, b := t` and `a, b = t` over a single
// tuple value into field accesses, binding non-trivial sources once. A
// literal unpacks to its elements directly.
func (fl *fileLower) lowerTupleUnpackStmt(s *syntax.AssignStmt, lhs []syntax.Expr, r syntax.Expr) []syntax.Stmt {
	if lit, ok := r.(*syntax.TupleLit); ok {
		if len(lit.Elems) != len(lhs) {
			return nil
		}
		var texts []string
		for _, e := range lit.Elems {
			texts = append(texts, fl.textOf(span{fl.off(syntax.StartPos(e)), fl.off(syntax.EndPos(e))}))
		}
		start := fl.off(syntax.StartPos(r))
		end := extendPastComma(fl.src, fl.off(syntax.EndPos(r)))
		fl.add(start, end, strings.Join(texts, ", "))
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
		var texts []string
		for i := range lhs {
			reads[i] = selOf(pos, r, elems[i])
			texts = append(texts, baseText+"."+elems[i])
		}
		s.Rhs = listOf(pos, reads)
		start, end := extendParens(fl.src, fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r)))
		fl.add(start, end, strings.Join(texts, ", "))
		return nil
	}
	tn := tupleTempKey(pos, 0)
	rSpan := span{fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r))}
	if _, ok := r.(*syntax.CallExpr); ok {
		rSpan = fl.tupleCallSpan(r.(*syntax.CallExpr))
	}
	rText := fl.textOf(rSpan)
	reads := make([]syntax.Expr, len(lhs))
	var texts []string
	for i := range lhs {
		reads[i] = selOf(pos, nameOf(pos, tn), elems[i])
		texts = append(texts, tn+"."+elems[i])
	}
	var text strings.Builder
	text.WriteString(tn + " := " + rText + "; ")
	text.WriteString(fl.textOf(span{fl.off(syntax.StartPos(s.Lhs)), fl.off(syntax.EndPos(s.Lhs))}))
	if s.Op == syntax.Def {
		text.WriteString(" := ")
	} else {
		text.WriteString(" = ")
	}
	text.WriteString(strings.Join(texts, ", "))
	_, rEnd := extendParens(fl.src, rSpan.start, rSpan.end)
	fl.add(fl.off(syntax.StartPos(s)), rEnd, text.String())
	hoist := defineOf(pos, []syntax.Expr{nameOf(pos, tn)}, []syntax.Expr{r})
	s.Rhs = listOf(pos, reads)
	return []syntax.Stmt{hoist, s}
}

// lowerTupleReturnStmt lowers `return t` and `return (1, 2)` over a
// single tuple into field reads, hoisting non-trivial sources. A
// literal returns its elements directly.
func (fl *fileLower) lowerTupleReturnStmt(s *syntax.ReturnStmt) []syntax.Stmt {
	results := syntax.UnpackListExpr(s.Results)
	if len(results) != 1 {
		return nil
	}
	r := syntax.Unparen(results[0])
	if _, ok := r.(*syntax.CallExpr); ok {
		return nil
	}
	pos := s.Pos()
	if lit, ok := r.(*syntax.TupleLit); ok {
		var texts []string
		for _, e := range lit.Elems {
			texts = append(texts, fl.textOf(span{fl.off(syntax.StartPos(e)), fl.off(syntax.EndPos(e))}))
		}
		start := fl.off(syntax.StartPos(r))
		end := extendPastComma(fl.src, fl.off(syntax.EndPos(r)))
		fl.add(start, end, strings.Join(texts, ", "))
		s.Results = listOf(pos, append([]syntax.Expr{}, lit.Elems...))
		return nil
	}
	elems, _ := fl.tupleTargetOfType(fl.pkg.info.TypeOf(results[0]))
	if len(elems) == 0 {
		return nil
	}
	if _, ok := syntax.Unparen(r).(*syntax.Name); ok {
		baseText := fl.textOf(span{fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r))})
		reads := make([]syntax.Expr, len(elems))
		var texts []string
		for i := range elems {
			reads[i] = selOf(pos, r, elems[i])
			texts = append(texts, baseText+"."+elems[i])
		}
		s.Results = listOf(pos, reads)
		start, end := extendParens(fl.src, fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r)))
		fl.add(start, end, strings.Join(texts, ", "))
		return nil
	}
	tn := tupleTempKey(pos, 0)
	rText := fl.textOf(span{fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r))})
	fl.add(fl.off(syntax.StartPos(s)), fl.off(syntax.StartPos(s)), tn+" := "+rText+"; ")
	hoist := defineOf(pos, []syntax.Expr{nameOf(pos, tn)}, []syntax.Expr{r})
	reads := make([]syntax.Expr, len(elems))
	var texts []string
	for i := range elems {
		reads[i] = selOf(pos, nameOf(pos, tn), elems[i])
		texts = append(texts, tn+"."+elems[i])
	}
	s.Results = listOf(pos, reads)
	start, end := extendParens(fl.src, fl.off(syntax.StartPos(r)), fl.off(syntax.EndPos(r)))
	fl.add(start, end, strings.Join(texts, ", "))
	return []syntax.Stmt{hoist, s}
}

// lowerTupleCallArgs lowers TupleLit arguments through parameter types.
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

// lowerTupleCompositeElems lowers TupleLit elements through the
// composite element, key, or field types.
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

// bareTupleValue unwraps a key-value element to its value.
func bareTupleValue(e syntax.Expr) syntax.Expr {
	if kv, ok := e.(*syntax.KeyValueExpr); ok {
		return kv.Value
	}
	return e
}

// tupleStructField reports the type of the struct field name.
func tupleStructField(st *types2.Struct, name string) types2.Type {
	for i := range st.NumFields() {
		if st.Field(i).Name() == name {
			return st.Field(i).Type()
		}
	}
	return nil
}

// lowerTupleLitOwn lowers a TupleLit through its own type.
func (fl *fileLower) lowerTupleLitOwn(lit *syntax.TupleLit) {
	names, types := fl.tupleTargetOfType(fl.pkg.info.TypeOf(lit))
	fl.lowerTupleLitTargeted(lit, names, types)
}

// lowerTupleLitTargeted lowers lit to a composite of the struct spelled
// by names and types, defaulting to the literal's own type.
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
	vals := make([]syntax.Expr, len(lit.Elems))
	texts := make([]string, len(lit.Elems))
	for i, e := range lit.Elems {
		vals[i] = e
		texts[i] = fl.textOf(span{fl.off(syntax.StartPos(e)), fl.off(syntax.EndPos(e))})
	}
	comp, compText := fl.tupleComposite(pos, names, types, vals, texts)
	fl.add(fl.off(syntax.StartPos(lit)), extendPastComma(fl.src, fl.off(syntax.EndPos(lit))), compText)
	replaceTupleExpr(fl.file, lit, comp)
}

// lowerTupleNestedExprTargeted lowers tuple forms under e, spelling
// nested literals through their element targets.
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

// tupleComposite builds the composite literal and its text for names
// and types over vals with precomputed texts.
func (fl *fileLower) tupleComposite(pos syntax.Pos, names []string, types []types2.Type, vals []syntax.Expr, texts []string) (*syntax.CompositeLit, string) {
	st, structText := fl.spellTupleStruct(pos, names, types)
	comp := &syntax.CompositeLit{Type: st, Rbrace: pos}
	comp.SetPos(pos)
	comp.ElemList = vals
	return comp, structText + "{" + strings.Join(texts, ", ") + "}"
}

// spellTupleStruct spells `struct{ T0 ...; ... }` for names and types
// and parses it back. It stops the compiler when a type has no Go
// spelling, like the neighboring spell helpers.
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
		text.WriteString(names[i] + " " + spelled)
	}
	text.WriteString(" }")
	st, ok := typeExprAt(text.String(), pos)
	if !ok {
		base.Fatalf("logos: cannot respell tuple type %q", text.String())
	}
	return st, text.String()
}
