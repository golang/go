// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"fmt"
	"go/constant"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file builds the synthetic syntax nodes of the M1-S5 lowering. Each
// template constructor takes ordinary expression arguments, so callers run
// it once with the user nodes for the native tree and once with marker
// names for the eject text. Markers print verbatim and are replaced with
// the original source text after printing.

// rtImportPath is the only support import lowered files may carry.
const rtImportPath = "logos/rt"

// rtAlias is the reserved local name of the support import.
const rtAlias = "_logos_rt"

// markerPrefix tags placeholder names that renderText replaces.
const markerPrefix = "\x00logosP"

// textSub replaces one printed marker with source text.
type textSub struct {
	marker string
	text   string
}

// markerText returns a placeholder name standing for text and records it.
// The name carries pos: renderText prints through the syntax printer,
// which requires known positions on every node.
func markerText(subs *[]textSub, seq *int, pos syntax.Pos, text string) *syntax.Name {
	m := fmt.Sprintf("%s%d\x00", markerPrefix, *seq)
	*seq++
	*subs = append(*subs, textSub{marker: m, text: text})
	return nameOf(pos, m)
}

// renderText prints stmts and replaces every marker with source text.
func renderText(stmts []syntax.Stmt, subs []textSub) string {
	var buf bytes.Buffer
	for _, s := range stmts {
		_, _ = syntax.Fprint(&buf, s, 0)
		buf.WriteByte('\n')
	}
	out := buf.String()
	if len(subs) == 0 {
		return out
	}
	pairs := make([]string, 0, 2*len(subs))
	for _, sub := range subs {
		pairs = append(pairs, sub.marker, sub.text)
	}
	return strings.NewReplacer(pairs...).Replace(out)
}

// leftmostOf returns the position of the leftmost token of e. Selector,
// index, slice, assertion, and call nodes sit at an interior token, so
// the start must descend to the operand.
func leftmostOf(e syntax.Expr) syntax.Pos {
	for {
		switch n := syntax.Unparen(e).(type) {
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

// renderExpr prints one expression without ShortForm truncation.
func renderExpr(e syntax.Expr) string {
	var buf bytes.Buffer
	_, _ = syntax.Fprint(&buf, e, syntax.LineForm)
	return buf.String()
}

// at returns n positioned at pos.
func at[T syntax.Node](n T, pos syntax.Pos) T {
	n.SetPos(pos)
	return n
}

// nameOf builds a name at pos.
func nameOf(pos syntax.Pos, value string) *syntax.Name {
	return at(syntax.NewName(pos, value), pos)
}

// litOf builds a basic literal at pos.
func litOf(pos syntax.Pos, value string, kind syntax.LitKind) *syntax.BasicLit {
	return at(&syntax.BasicLit{Value: value, Kind: kind}, pos)
}

// nilOf builds a nil name at pos.
func nilOf(pos syntax.Pos) *syntax.Name {
	return nameOf(pos, "nil")
}

// callOf builds Fun(args...) at pos.
func callOf(pos syntax.Pos, fun syntax.Expr, args []syntax.Expr) *syntax.CallExpr {
	return at(&syntax.CallExpr{Fun: fun, ArgList: args}, pos)
}

// selOf builds X.Sel at pos.
func selOf(pos syntax.Pos, x syntax.Expr, sel string) *syntax.SelectorExpr {
	return at(&syntax.SelectorExpr{X: x, Sel: nameOf(pos, sel)}, pos)
}

// notNil builds X != nil at pos.
func notNil(pos syntax.Pos, x syntax.Expr) *syntax.Operation {
	return at(&syntax.Operation{Op: syntax.Neq, X: x, Y: nilOf(pos)}, pos)
}

// assignOf builds Lhs = Rhs at pos.
func assignOf(pos syntax.Pos, lhs, rhs []syntax.Expr) *syntax.AssignStmt {
	return at(&syntax.AssignStmt{Lhs: listOf(pos, lhs), Rhs: listOf(pos, rhs)}, pos)
}

// defineOf builds Lhs := Rhs at pos.
func defineOf(pos syntax.Pos, lhs, rhs []syntax.Expr) *syntax.AssignStmt {
	return at(&syntax.AssignStmt{Op: syntax.Def, Lhs: listOf(pos, lhs), Rhs: listOf(pos, rhs)}, pos)
}

// listOf packs exprs for an assignment side at pos. A multi-name list
// renders a ListExpr, which must carry pos like every printed node.
func listOf(pos syntax.Pos, exprs []syntax.Expr) syntax.Expr {
	if len(exprs) == 1 {
		return exprs[0]
	}
	return at(&syntax.ListExpr{ElemList: exprs}, pos)
}

// exprStmtOf builds an expression statement at pos.
func exprStmtOf(pos syntax.Pos, x syntax.Expr) *syntax.ExprStmt {
	return at(&syntax.ExprStmt{X: x}, pos)
}

// blockOf builds a statement block at pos. The closing brace has no source
// text, so Rbrace takes pos as well: the unified IR writer closes the
// block's scope at Rbrace, and a zero position there is the "unknown scope
// position" ICE.
func blockOf(pos syntax.Pos, list []syntax.Stmt) *syntax.BlockStmt {
	return at(&syntax.BlockStmt{List: list, Rbrace: pos}, pos)
}

// ifOf builds if Init; Cond Then at pos.
func ifOf(pos syntax.Pos, init syntax.SimpleStmt, cond syntax.Expr, then []syntax.Stmt) *syntax.IfStmt {
	return at(&syntax.IfStmt{Init: init, Cond: cond, Then: blockOf(pos, then)}, pos)
}

// ifElseOf builds if Init; Cond Then Else at pos.
func ifElseOf(pos syntax.Pos, init syntax.SimpleStmt, cond syntax.Expr, then []syntax.Stmt, els syntax.Stmt) *syntax.IfStmt {
	return at(&syntax.IfStmt{Init: init, Cond: cond, Then: blockOf(pos, then), Else: els}, pos)
}

// returnOf builds a return statement at pos.
func returnOf(pos syntax.Pos, results []syntax.Expr) *syntax.ReturnStmt {
	var r syntax.Expr
	if len(results) > 0 {
		r = listOf(pos, results)
	}
	return at(&syntax.ReturnStmt{Results: r}, pos)
}

// deferOf builds a defer statement at pos.
func deferOf(pos syntax.Pos, call syntax.Expr) *syntax.CallStmt {
	return at(&syntax.CallStmt{Tok: syntax.Defer, Call: call}, pos)
}

// rtSel builds _logos_rt.Sel at pos.
func rtSel(pos syntax.Pos, sel string) *syntax.SelectorExpr {
	return selOf(pos, nameOf(pos, rtAlias), sel)
}

// zeroForType builds the zero value of typ at pos. typeExpr supplies the
// source spelling of struct and array types; type parameters lower to
// *new(T).
func zeroForType(typ types2.Type, typeExpr syntax.Expr, pos syntax.Pos) syntax.Expr {
	typ = types2.Unalias(typ)
	if _, ok := typ.(*types2.TypeParam); ok {
		return at(&syntax.Operation{
			Op: syntax.Mul,
			X:  callOf(pos, nameOf(pos, "new"), []syntax.Expr{cloneType(typeExpr, pos)}),
		}, pos)
	}
	if b, ok := typ.(*types2.Basic); ok {
		return zeroBasic(b, pos)
	}
	switch typ.Underlying().(type) {
	case *types2.Struct, *types2.Array:
		return at(&syntax.CompositeLit{Type: cloneType(typeExpr, pos), Rbrace: pos}, pos)
	default:
		return nilOf(pos)
	}
}

// zeroBasic builds the zero value of a basic type at pos.
func zeroBasic(b *types2.Basic, pos syntax.Pos) syntax.Expr {
	switch {
	case b.Kind() == types2.Bool:
		return nameOf(pos, "false")
	case b.Kind() == types2.String:
		return litOf(pos, `""`, syntax.StringLit)
	case b.Kind() == types2.UnsafePointer:
		return nilOf(pos)
	default:
		return litOf(pos, "0", syntax.IntLit)
	}
}

// cloneType copies a type expression by printing and re-parsing it, then
// moves every node to pos. The re-parse gives the clone its own nodes. A
// type that does not respell becomes any.
func cloneType(e syntax.Expr, pos syntax.Pos) syntax.Expr {
	if decl, ok := respellType(e, pos); ok {
		return decl
	}
	return nameOf(pos, "any")
}

// mustCloneType is cloneType for a declared type, where a stand-in would
// give the variable the wrong type; it stops the compiler instead.
func mustCloneType(e syntax.Expr, pos syntax.Pos) syntax.Expr {
	decl, ok := respellType(e, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q", renderExpr(e))
	}
	return decl
}

// respellType prints and re-parses e and moves every node of the copy to
// pos. It reports false when the printed text is not a type.
func respellType(e syntax.Expr, pos syntax.Pos) (syntax.Expr, bool) {
	decl, ok := reparseType(renderExpr(e))
	if !ok {
		return nil, false
	}
	syntax.Inspect(decl, func(n syntax.Node) bool {
		if n != nil {
			n.SetPos(pos)
		}
		return true
	})
	return decl, true
}

// reparseType parses "package p;var _ T" and returns the type.
func reparseType(text string) (syntax.Expr, bool) {
	file, err := syntax.Parse(syntax.NewFileBase("logos-clone"), strings.NewReader("package logosclone\nvar _ "+text), nil, nil, 0)
	if err != nil || file == nil || len(file.DeclList) != 1 {
		return nil, false
	}
	decl, ok := file.DeclList[0].(*syntax.VarDecl)
	if !ok || decl.Type == nil {
		return nil, false
	}
	return decl.Type, true
}

// typeConst reports the constant value of e.
func typeConst(info *types2.Info, e syntax.Expr) (constant.Value, bool) {
	if info.Types != nil {
		if tv, ok := info.Types[e]; ok && tv.Value != nil {
			return tv.Value, true
		}
		return nil, false
	}
	if tv := e.GetTypeInfo(); tv.Value != nil {
		return tv.Value, true
	}
	return nil, false
}
