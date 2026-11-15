// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"slices"

	"cmd/compile/internal/syntax"
)

type exitSpec struct {
	pos   syntax.Pos
	err   syntax.Expr
	site  *syntax.Name
	zeros []syntax.Expr
}

func buildExit(spec exitSpec) []syntax.Stmt {
	trace := exprStmtOf(spec.pos, callOf(spec.pos, rtSel(spec.pos, "TraceErr"), []syntax.Expr{
		spec.err,
		at(&syntax.Operation{Op: syntax.And, X: spec.site}, spec.pos),
	}))
	results := slices.Concat(slices.Clone(spec.zeros), []syntax.Expr{spec.err})
	return []syntax.Stmt{trace, returnOf(spec.pos, results)}
}

func errDefineOf(pos syntax.Pos, lhs []syntax.Expr, call syntax.Expr, err *syntax.Name) *syntax.AssignStmt {
	withErr := slices.Clone(lhs)
	withErr = append(withErr, err)
	return defineOf(pos, withErr, []syntax.Expr{call})
}

func errCheckOf(pos syntax.Pos, err *syntax.Name, exit []syntax.Stmt) *syntax.IfStmt {
	return ifOf(pos, nil, notNil(pos, err), exit)
}

func errCheckInitOf(pos syntax.Pos, lhs []syntax.Expr, call syntax.Expr, err *syntax.Name, exit []syntax.Stmt) *syntax.IfStmt {
	return ifOf(pos, errDefineOf(pos, lhs, call, err), notNil(pos, err), exit)
}

func closeAtExit(check *syntax.IfStmt, exitPos syntax.Pos) *syntax.IfStmt {
	check.Then.Rbrace = exitPos
	return check
}

func typedVarDecl(pos syntax.Pos, name *syntax.Name, typ syntax.Expr) *syntax.DeclStmt {
	return at(&syntax.DeclStmt{DeclList: []syntax.Decl{
		at(&syntax.VarDecl{NameList: []*syntax.Name{name}, Type: typ}, pos),
	}}, pos)
}

func errorVarDecl(pos syntax.Pos, err *syntax.Name) *syntax.DeclStmt {
	return typedVarDecl(pos, err, nameOf(pos, "error"))
}

func blankNames(pos syntax.Pos, n int) []syntax.Expr {
	out := make([]syntax.Expr, n)
	for i := range out {
		out[i] = nameOf(pos, "_")
	}
	return out
}

func joinAssign(pos syntax.Pos, errName, inner string) []syntax.Stmt {
	then := assignOf(pos, []syntax.Expr{nameOf(pos, errName)}, []syntax.Expr{nameOf(pos, inner)})
	join := callOf(pos, rtSel(pos, "Join"), []syntax.Expr{nameOf(pos, errName), nameOf(pos, inner)})
	otherwise := assignOf(pos, []syntax.Expr{nameOf(pos, errName)}, []syntax.Expr{join})
	isNil := at(&syntax.Operation{Op: syntax.Eql, X: nameOf(pos, errName), Y: nilOf(pos)}, pos)
	return []syntax.Stmt{ifElseOf(pos, nil, isNil, []syntax.Stmt{then}, blockOf(pos, []syntax.Stmt{otherwise}))}
}

func guardedJoin(pos syntax.Pos, okName, errName string, stmts []syntax.Stmt) *syntax.IfStmt {
	cond := at(&syntax.Operation{
		Op: syntax.OrOr,
		X:  at(&syntax.Operation{Op: syntax.Not, X: nameOf(pos, okName)}, pos),
		Y:  notNil(pos, nameOf(pos, errName)),
	}, pos)
	return ifOf(pos, nil, cond, stmts)
}

func deferClosure(pos syntax.Pos, stmts []syntax.Stmt) *syntax.FuncLit {
	return at(&syntax.FuncLit{
		Type: at(&syntax.FuncType{
			ParamList: []*syntax.Field{},
		}, pos),
		Body: blockOf(pos, stmts),
	}, pos)
}

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
			NKeys:    2,
			Rbrace:   pos,
		}, pos),
	}, pos)
}

func rtImport(pos syntax.Pos) *syntax.ImportDecl {
	return at(&syntax.ImportDecl{
		LocalPkgName: nameOf(pos, rtAlias),
		Path:         litOf(pos, fmt.Sprintf("%q", rtImportPath), syntax.StringLit),
	}, pos)
}
