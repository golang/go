// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"slices"

	"cmd/compile/internal/syntax"
)

// This file constructs the lowered statement shapes of the M1-S5
// templates. Each builder takes the expressions to splice in, so the same
// builder serves the native tree and the eject-text twin.

// exitSpec describes one ? error path: record the site, then return zero
// values and the error.
type exitSpec struct {
	pos   syntax.Pos
	err   syntax.Expr
	site  *syntax.Name
	zeros []syntax.Expr
}

// buildExit builds the ? error path.
func buildExit(spec exitSpec) []syntax.Stmt {
	trace := exprStmtOf(spec.pos, callOf(spec.pos, rtSel(spec.pos, "TraceErr"), []syntax.Expr{
		spec.err,
		at(&syntax.Operation{Op: syntax.And, X: spec.site}, spec.pos),
	}))
	results := slices.Concat(slices.Clone(spec.zeros), []syntax.Expr{spec.err})
	return []syntax.Stmt{trace, returnOf(spec.pos, results)}
}

// errDefineOf builds lhs..., err := call.
func errDefineOf(pos syntax.Pos, lhs []syntax.Expr, call syntax.Expr, err *syntax.Name) *syntax.AssignStmt {
	withErr := slices.Clone(lhs)
	withErr = append(withErr, err)
	return defineOf(pos, withErr, []syntax.Expr{call})
}

// errCheckOf builds if err != nil exit.
func errCheckOf(pos syntax.Pos, err *syntax.Name, exit []syntax.Stmt) *syntax.IfStmt {
	return ifOf(pos, nil, notNil(pos, err), exit)
}

// errCheckInitOf builds if lhs..., err := call; err != nil exit.
func errCheckInitOf(pos syntax.Pos, lhs []syntax.Expr, call syntax.Expr, err *syntax.Name, exit []syntax.Stmt) *syntax.IfStmt {
	return ifOf(pos, errDefineOf(pos, lhs, call, err), notNil(pos, err), exit)
}

// typedVarDecl builds var name typ.
func typedVarDecl(pos syntax.Pos, name *syntax.Name, typ syntax.Expr) *syntax.DeclStmt {
	return at(&syntax.DeclStmt{DeclList: []syntax.Decl{
		at(&syntax.VarDecl{NameList: []*syntax.Name{name}, Type: typ}, pos),
	}}, pos)
}

// errorVarDecl builds var err error.
func errorVarDecl(pos syntax.Pos, err *syntax.Name) *syntax.DeclStmt {
	return typedVarDecl(pos, err, nameOf(pos, "error"))
}

// blankNames builds n blank names at pos.
func blankNames(pos syntax.Pos, n int) []syntax.Expr {
	out := make([]syntax.Expr, n)
	for i := range out {
		out[i] = nameOf(pos, "_")
	}
	return out
}

// joinAssign builds the defer? error join into the named error result.
// The inner spelling is instantiated fresh at every use.
func joinAssign(pos syntax.Pos, errName, inner string) []syntax.Stmt {
	then := assignOf(pos, []syntax.Expr{nameOf(pos, errName)}, []syntax.Expr{nameOf(pos, inner)})
	join := callOf(pos, rtSel(pos, "Join"), []syntax.Expr{nameOf(pos, errName), nameOf(pos, inner)})
	otherwise := assignOf(pos, []syntax.Expr{nameOf(pos, errName)}, []syntax.Expr{join})
	isNil := at(&syntax.Operation{Op: syntax.Eql, X: nameOf(pos, errName), Y: nilOf(pos)}, pos)
	return []syntax.Stmt{ifElseOf(pos, nil, isNil, []syntax.Stmt{then}, blockOf(pos, []syntax.Stmt{otherwise}))}
}

// guardedJoin wraps stmts in if !ok || err != nil.
func guardedJoin(pos syntax.Pos, okName, errName string, stmts []syntax.Stmt) *syntax.IfStmt {
	cond := at(&syntax.Operation{
		Op: syntax.OrOr,
		X:  at(&syntax.Operation{Op: syntax.Not, X: nameOf(pos, okName)}, pos),
		Y:  notNil(pos, nameOf(pos, errName)),
	}, pos)
	return ifOf(pos, nil, cond, stmts)
}

// deferClosure builds func() { stmts... }. The type carries an explicit
// empty parameter list so the rendered text is exactly func(); a nil
// result list is what the parser produces for a bare func().
func deferClosure(pos syntax.Pos, stmts []syntax.Stmt) *syntax.FuncLit {
	return at(&syntax.FuncLit{
		Type: at(&syntax.FuncType{
			ParamList: []*syntax.Field{},
		}, pos),
		Body: blockOf(pos, stmts),
	}, pos)
}

// okInit builds _logos_ok := false.
func okInit(pos syntax.Pos, ok *syntax.Name) *syntax.AssignStmt {
	return defineOf(pos, []syntax.Expr{ok}, []syntax.Expr{nameOf(pos, "false")})
}

// siteDecl builds var site = _logos_rt.Site{File: base, Line: line}.
func siteDecl(pos syntax.Pos, site *syntax.Name, base string, line uint) *syntax.VarDecl {
	typ := selOf(pos, nameOf(pos, rtAlias), "Site")
	file := at(&syntax.KeyValueExpr{
		Key:   nameOf(pos, "File"),
		Value: litOf(pos, fmt.Sprintf("%q", base), syntax.StringLit),
	}, pos)
	ln := at(&syntax.KeyValueExpr{
		Key:   nameOf(pos, "Line"),
		Value: litOf(pos, fmt.Sprintf("%d", line), syntax.IntLit),
	}, pos)
	return at(&syntax.VarDecl{
		NameList: []*syntax.Name{site},
		Values: at(&syntax.CompositeLit{
			Type:     typ,
			ElemList: []syntax.Expr{file, ln},
			// The unified IR writer reads NKeys == 0 as a positional
			// literal and types each element as a value; both elements
			// here are keyed, as the parser would have counted them.
			NKeys:  2,
			Rbrace: pos,
		}, pos),
	}, pos)
}

// rtImport builds the single support import of lowered files.
func rtImport(pos syntax.Pos) *syntax.ImportDecl {
	return at(&syntax.ImportDecl{
		LocalPkgName: nameOf(pos, rtAlias),
		Path:         litOf(pos, fmt.Sprintf("%q", rtImportPath), syntax.StringLit),
	}, pos)
}
