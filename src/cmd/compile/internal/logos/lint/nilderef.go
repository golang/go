// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"bytes"
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// nilDerefID is the ID of correctness::nil_deref_unproven.
const nilDerefID = "correctness::nil_deref_unproven"

// NilDerefTag is the rule tag types2 puts in front of the message of a
// nil_deref_unproven error.
const NilDerefTag = "[" + nilDerefID + "] "

// NilDerefResult builds the finding for a nil_deref_unproven error that
// types2 reported at pos, the member name of a selection on an option, with
// the autofix candidates of SPEC Sec 1.2. msg is the text of the types2
// error and must start with the rule tag. The span of the finding is the
// expression the first candidate replaces: the selection, or the call when
// the selection is called. info is the checker's Info while it checks the
// package, so only the types of sub-expressions are read. It reports false
// when msg names another rule or pos is not the member of a selection in f,
// and the caller keeps the plain error.
func NilDerefResult(info *types2.Info, f *syntax.File, src []byte, pos syntax.Pos, msg string) (Result, bool) {
	message, ok := strings.CutPrefix(msg, NilDerefTag)
	if !ok {
		return Result{}, false
	}
	stack := selectionPath(f, pos)
	if stack == nil {
		return Result{}, false
	}
	site := derefSite{info: info, src: src, stack: stack}
	recv := site.sel().X
	r := Result{
		Rule:     nilDerefID,
		Severity: Forbid,
		Pos:      syntax.StartPos(site.target()),
		End:      site.end(),
		Message:  message,
		Provenance: Provenance{
			OriginExpr:   syntax.String(recv),
			OriginLine:   site.originLine(),
			InferredType: typeText(TypeOf(recv)),
		},
	}
	for _, fix := range []func() (Fix, bool){site.chainFix, site.guardFix} {
		if fx, ok := fix(); ok {
			r.Fixes = append(r.Fixes, fx)
		}
	}
	return r, true
}

// typeText prints t with package names, or "" for a nil type.
func typeText(t types2.Type) string {
	if t == nil {
		return ""
	}
	return types2.TypeString(t, func(p *types2.Package) string { return p.Name() })
}

// selectionPath returns the nodes from f down to the selector expression
// whose member name is at pos, or nil.
func selectionPath(f *syntax.File, pos syntax.Pos) []syntax.Node {
	var stack, found []syntax.Node
	syntax.Inspect(f, func(n syntax.Node) bool {
		switch {
		case n == nil:
			stack = stack[:len(stack)-1]
			return false
		case found != nil:
			return false
		}
		stack = append(stack, n)
		if sel, ok := n.(*syntax.SelectorExpr); ok && sel.Sel.Pos() == pos {
			found = slices.Clone(stack)
		}
		return true
	})
	return found
}

// derefSite is one unproven selection: the path from the file to its
// selector expression and the source it was parsed from.
type derefSite struct {
	info  *types2.Info
	src   []byte
	stack []syntax.Node
}

// sel returns the selector expression.
func (d derefSite) sel() *syntax.SelectorExpr {
	return d.stack[len(d.stack)-1].(*syntax.SelectorExpr)
}

// call returns the call whose callee is the selection, or nil.
func (d derefSite) call() *syntax.CallExpr {
	if len(d.stack) < 2 {
		return nil
	}
	call, ok := d.stack[len(d.stack)-2].(*syntax.CallExpr)
	if !ok || call.Fun != d.sel() {
		return nil
	}
	return call
}

// target returns the expression a replacement covers: the call when the
// selection is called, else the selection.
func (d derefSite) target() syntax.Expr {
	if call := d.call(); call != nil {
		return call
	}
	return d.sel()
}

// end returns the position after the target.
func (d derefSite) end() syntax.Pos {
	if call := d.call(); call != nil {
		if end, ok := syntax.ParenEnd(d.src, call.Pos(), syntax.LogosMode); ok {
			return end
		}
	}
	return syntax.EndPos(d.target())
}

// originLine returns the line of the declaration of the receiver when it is
// a variable, else the line of the receiver.
func (d derefSite) originLine() int {
	recv := d.sel().X
	if name, ok := syntax.Unparen(recv).(*syntax.Name); ok {
		if obj := d.info.Uses[name]; obj != nil && obj.Pos().IsKnown() {
			return int(obj.Pos().Line())
		}
	}
	return int(syntax.StartPos(recv).Line())
}

// offset returns the byte offset of pos in the source.
func (d derefSite) offset(pos syntax.Pos) (int, bool) {
	off := 0
	for line := uint(1); line < pos.Line(); line++ {
		i := bytes.IndexByte(d.src[off:], '\n')
		if i < 0 {
			return 0, false
		}
		off += i + 1
	}
	off += int(pos.Col()) - 1
	return off, off >= 0 && off <= len(d.src)
}

// chainFix offers optional chaining: the target with `?.` in place of the
// `.` before the member, and a `?? <zero>` default unless the result is
// already an option or the call returns nothing. It is offered when the
// member is a field, a method value, or a method with at most one result,
// and a zero value of the result type is known.
func (d derefSite) chainFix() (Fix, bool) {
	opt := optionOf(TypeOf(d.sel().X))
	if opt == nil {
		return Fix{}, false
	}
	result, hasResult, ok := d.memberResult(opt)
	if !ok {
		return Fix{}, false
	}
	chained, ok := d.chained()
	if !ok {
		return Fix{}, false
	}
	if !hasResult || optionOf(result) != nil {
		return Fix{Description: "use optional chaining", Replacement: chained}, true
	}
	zero, ok := zeroLiteral(result)
	if !ok {
		return Fix{}, false
	}
	return Fix{Description: "use optional chaining with a ?? default", Replacement: "(" + chained + " ?? " + zero + ")"}, true
}

// memberResult returns the type the selection yields: the field type, or the
// single result of a called method. hasResult is false for a method value
// and for a call with no result; ok is false for a member that cannot chain.
func (d derefSite) memberResult(opt *types2.Option) (result types2.Type, hasResult, ok bool) {
	obj, _, _ := types2.LookupFieldOrMethod(opt.Elem, true, memberPackage(opt.Elem), d.sel().Sel.Value)
	switch obj := obj.(type) {
	case *types2.Var:
		return obj.Type(), obj.Type() != nil, obj.Type() != nil
	case *types2.Func:
		return d.methodResult(obj)
	}
	return nil, false, false
}

// methodResult is memberResult for a method: a method value yields no result
// to default, and a called method must have at most one.
func (d derefSite) methodResult(fn *types2.Func) (result types2.Type, hasResult, ok bool) {
	results := fn.Signature().Results()
	switch {
	case d.call() == nil || results.Len() == 0:
		return nil, false, true
	case results.Len() == 1:
		return results.At(0).Type(), true, true
	}
	return nil, false, false
}

// memberPackage returns the package that declares the named type behind t,
// which a lookup of an unexported member needs, or nil.
func memberPackage(t types2.Type) *types2.Package {
	if p, ok := types2.Unalias(t).(*types2.Pointer); ok {
		t = p.Elem()
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		return n.Obj().Pkg()
	}
	return nil
}

// chained returns the source of the target with `?` inserted before the dot
// of the selection.
func (d derefSite) chained() (string, bool) {
	start, ok1 := d.offset(syntax.StartPos(d.target()))
	end, ok2 := d.offset(d.end())
	member, ok3 := d.offset(d.sel().Sel.Pos())
	if !ok1 || !ok2 || !ok3 || start > member || member > end {
		return "", false
	}
	dot := len(bytes.TrimRight(d.src[start:member], " \t\r\n")) + start - 1
	if dot < start || d.src[dot] != '.' {
		return "", false
	}
	return string(d.src[start:dot]) + "?" + string(d.src[dot:end]), true
}

// guardFix offers an early-return guard before the statement that holds the
// selection: `if x == nil { return <zero results> }`, for a receiver that is
// a plain name, since a nil test narrows only a name. It is offered when the
// statement starts its line and every result of the function has a known zero
// value.
func (d derefSite) guardFix() (Fix, bool) {
	name, ok := syntax.Unparen(d.sel().X).(*syntax.Name)
	stmt, fn := d.enclosing()
	if !ok || stmt == nil || fn == nil {
		return Fix{}, false
	}
	zeros, ok := resultZeros(fn)
	if !ok {
		return Fix{}, false
	}
	pos := stmtStart(stmt)
	off, ok := d.offset(pos)
	if !ok {
		return Fix{}, false
	}
	indent := d.src[bytes.LastIndexByte(d.src[:off], '\n')+1 : off]
	if len(bytes.TrimLeft(indent, " \t")) != 0 {
		return Fix{}, false
	}
	ret := "return"
	if len(zeros) > 0 {
		ret += " " + strings.Join(zeros, ", ")
	}
	text := string(indent) + "if " + name.Value + " == nil {\n" + string(indent) + "\t" + ret + "\n" + string(indent) + "}\n"
	return Fix{Description: "return early when " + name.Value + " is nil", Replacement: text, InsertBeforeLine: int(pos.Line())}, true
}

// stmtStart returns the position where the source of s begins. The parser
// puts an expression statement at its operand, a call at its open
// parenthesis, and StartPos does not correct that, so the start comes from
// the operand.
func stmtStart(s syntax.Stmt) syntax.Pos {
	if es, ok := s.(*syntax.ExprStmt); ok {
		return syntax.StartPos(es.X)
	}
	return syntax.StartPos(s)
}

// enclosing returns the innermost statement of a statement list that holds
// the selection and the type of the function around it. A lambda in between
// has no statements to guard, so it yields no statement.
func (d derefSite) enclosing() (syntax.Stmt, *syntax.FuncType) {
	var stmt syntax.Stmt
	for i := len(d.stack) - 1; i >= 0; i-- {
		if ft := funcTypeOf(d.stack[i]); ft != nil {
			return stmt, ft
		}
		if _, ok := d.stack[i].(*syntax.LambdaExpr); ok && stmt == nil {
			return nil, nil
		}
		if stmt == nil {
			stmt = d.listStmt(i)
		}
	}
	return nil, nil
}

// listStmt returns the statement at index i of the path when it is an
// element of a statement list, else nil.
func (d derefSite) listStmt(i int) syntax.Stmt {
	s, ok := d.stack[i].(syntax.Stmt)
	if !ok || i == 0 || !inStmtList(d.stack[i-1]) {
		return nil
	}
	return s
}

// funcTypeOf returns the signature of a function declaration or literal, or
// nil for any other node.
func funcTypeOf(n syntax.Node) *syntax.FuncType {
	switch n := n.(type) {
	case *syntax.FuncDecl:
		return n.Type
	case *syntax.FuncLit:
		return n.Type
	}
	return nil
}

// inStmtList reports whether n holds a list of statements.
func inStmtList(n syntax.Node) bool {
	switch n.(type) {
	case *syntax.BlockStmt, *syntax.CaseClause, *syntax.CommClause:
		return true
	}
	return false
}

// resultZeros returns the zero value source of each result of a function.
func resultZeros(ft *syntax.FuncType) ([]string, bool) {
	var out []string
	for _, field := range ft.ResultList {
		zero, ok := zeroLiteral(TypeOf(field.Type))
		if !ok {
			return nil, false
		}
		out = append(out, zero)
	}
	return out, true
}

// zeroLiteral returns source for the zero value of t when one is known to
// be legal: false, "", 0, or nil for a type that has nil. A non-nil pointer
// has no zero value in loGo/s, and a struct or array may hold one.
func zeroLiteral(t types2.Type) (string, bool) {
	if t == nil {
		return "", false
	}
	if _, ok := types2.Unalias(t).(*types2.TypeParam); ok {
		return "", false
	}
	switch u := t.Underlying().(type) {
	case *types2.Basic:
		return basicZero(u)
	case *types2.Slice, *types2.Map, *types2.Chan, *types2.Signature, *types2.Interface, *types2.Option:
		return "nil", true
	}
	return "", false
}

// basicZero returns the zero literal of a basic type.
func basicZero(b *types2.Basic) (string, bool) {
	switch info := b.Info(); {
	case info&types2.IsBoolean != 0:
		return "false", true
	case info&types2.IsString != 0:
		return `""`, true
	case info&types2.IsNumeric != 0:
		return "0", true
	}
	return "", false
}
