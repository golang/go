// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"fmt"
	"go/token"
	"reflect"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers the M2 nodes (SPEC Sec 2.3, T-203.1 and T-203.3):
//
//   - a switch expression becomes an immediately invoked function
//     literal whose switch statement returns the chosen arm, and
//   - the inline assertions of `type S struct : I, J { ... }` become one
//     `var _ I = (*S)(nil)` declaration each after the type declaration.
//
// A closed union itself needs no lowering: it is an interface whose type
// set the unified IR reader folds, so the run-time representation is the
// upstream interface. The recheck of the lowered tree accepts the unions
// that remain in value positions through Config.LogosUnions.
//
// Every rewrite builds the native tree and the eject text together. The
// eject text uses small splices (an insertion before `switch`, one
// replacement per `=>`, an insertion after the closing brace) instead of
// one replacement of the whole expression, so the splices of nested
// constructs inside an arm stay valid beside them.

// lowerUnions rewrites every switch expression and asserted struct of the
// file. It runs before the statement lowering, which then sees ordinary Go
// where these nodes stood.
func (fl *fileLower) lowerUnions() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case nil:
			return true
		case *syntax.File:
			n.DeclList = fl.lowerAssertedDecls(n.DeclList)
		case *syntax.DeclStmt:
			n.DeclList = fl.lowerAssertedDecls(n.DeclList)
		}
		replaceSwitchExprs(n, fl.lowerSwitchExpr)
		return true
	})
}

// replaceSwitchExprs replaces each *syntax.SwitchExpr that n holds
// directly, as a field or as an element of a list field, with repl of it.
// The caller walks n's children after the call, so it reaches the
// replacement and everything the replacement carries.
func replaceSwitchExprs(n syntax.Node, repl func(*syntax.SwitchExpr) syntax.Expr) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	typ := v.Type()
	for i := range v.NumField() {
		if typ.Field(i).IsExported() {
			replaceSwitchField(v.Field(i), repl)
		}
	}
}

// replaceSwitchField replaces a *syntax.SwitchExpr held by the interface
// value f, or by an element of the slice or array f.
func replaceSwitchField(f reflect.Value, repl func(*syntax.SwitchExpr) syntax.Expr) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() {
			return
		}
		if se, ok := f.Elem().Interface().(*syntax.SwitchExpr); ok {
			f.Set(reflect.ValueOf(repl(se)))
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			replaceSwitchField(f.Index(i), repl)
		}
	}
}

// lowerSwitchExpr lowers one switch expression to
//
//	func() T { switch ... { case ...: return arm; ... }; panic(...) }()
//
// where T is the checked common type of the arms. The panic follows the
// switch only when no default arm ends it: a type switch expression may
// be exhaustive over a union and nil without one, and run-time stores
// that bypass the type system can still reach the end.
func (fl *fileLower) lowerSwitchExpr(e *syntax.SwitchExpr) syntax.Expr {
	pos := e.Pos()
	typ := fl.pkg.info.TypeOf(e)
	if typ == nil {
		base.Fatalf("logos: switch expression at %s has no checked type", pos.String())
	}
	text, ok := fl.spellType(typ)
	if !ok {
		base.Fatalf("logos: switch expression at %s has type %s from a package this file does not import", pos.String(), text)
	}
	resultType, ok := typeExprAt(text, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q of the switch expression at %s", text, pos.String())
	}

	clauses := make([]*syntax.CaseClause, len(e.Arms))
	for i, arm := range e.Arms {
		result := at(&syntax.ReturnStmt{Results: arm.Body}, arm.Arrow)
		clauses[i] = at(&syntax.CaseClause{
			Cases: arm.Cases,
			Body:  []syntax.Stmt{result},
			Colon: arm.Arrow,
		}, arm.Pos())
	}
	body := []syntax.Stmt{
		at(&syntax.SwitchStmt{Init: e.Init, Tag: e.Tag, Body: clauses, Rbrace: e.Rbrace}, pos),
	}
	unreachable := fmt.Sprintf("%q", fmt.Sprintf("logos: switch expression at %s:%d matched no arm", fl.baseName, pos.Line()))
	if !hasDefaultArm(e) {
		body = append(body, exprStmtOf(e.Rbrace, callOf(e.Rbrace, nameOf(e.Rbrace, "panic"), []syntax.Expr{
			litOf(e.Rbrace, unreachable, syntax.StringLit),
		})))
	}

	signature := at(&syntax.FuncType{
		ParamList:  []*syntax.Field{},
		ResultList: []*syntax.Field{at(&syntax.Field{Type: resultType}, pos)},
	}, pos)
	lit := at(&syntax.FuncLit{Type: signature, Body: at(&syntax.BlockStmt{List: body, Rbrace: e.Rbrace}, pos)}, pos)

	fl.spliceSwitchExpr(e, text, unreachable)
	return callOf(pos, lit, nil)
}

// hasDefaultArm reports whether e has a default arm.
func hasDefaultArm(e *syntax.SwitchExpr) bool {
	for _, arm := range e.Arms {
		if arm.Cases == nil {
			return true
		}
	}
	return false
}

// spliceSwitchExpr records the eject edits of one switch expression: the
// `switch` keyword gains the function literal head, each `=>` becomes
// `: return` (the span up to the result when a line break separates
// them, because `return` ends its line), and the final brace gains the
// closing of the literal. Every edit replaces a token instead of
// inserting beside one, so an enclosing edit that spans the whole
// expression absorbs all of them, also when it starts at `switch` or ends
// at the brace.
func (fl *fileLower) spliceSwitchExpr(e *syntax.SwitchExpr, text, unreachable string) {
	head := fl.off(e.Pos())
	fl.add(head, head+len("switch"), "func() "+text+" { switch")
	for _, arm := range e.Arms {
		arrow := fl.off(arm.Arrow)
		after := arrow + len("=>")
		result := firstTokenStart(fl.src, after)
		if bytes.IndexByte(fl.src[after:result], '\n') >= 0 {
			fl.add(arrow, result, ": return ")
		} else {
			fl.add(arrow, after, ": return")
		}
	}
	brace := fl.off(e.Rbrace)
	closing := "} }()"
	if !hasDefaultArm(e) {
		closing = "}; panic(" + unreachable + ") }()"
	}
	fl.add(brace, brace+1, closing)
}

// firstTokenStart returns the offset of the first token at or after from
// that is neither a comment nor an automatic semicolon, or the end of src.
func firstTokenStart(src []byte, from int) int {
	start := len(src)
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		start = t.start
		return false
	})
	return start
}

// importNames maps each package the file imports to the name that denotes
// it in the file; a dot import maps to the empty name. Blank imports
// denote nothing and stay out.
func (fl *fileLower) importNames() map[*types2.Package]string {
	if fl.names != nil {
		return fl.names
	}
	fl.names = make(map[*types2.Package]string)
	info := fl.pkg.info
	for _, decl := range fl.file.DeclList {
		imp, ok := decl.(*syntax.ImportDecl)
		if !ok {
			continue
		}
		var obj types2.Object
		if imp.LocalPkgName != nil {
			obj = info.Defs[imp.LocalPkgName]
		} else {
			obj = info.Implicits[imp]
		}
		name, ok := obj.(*types2.PkgName)
		if !ok || name.Name() == "_" {
			continue
		}
		spelling := name.Name()
		if spelling == "." {
			spelling = ""
		}
		fl.names[name.Imported()] = spelling
	}
	return fl.names
}

// spellType renders typ as Go source in this file: names of the package
// itself stand bare, and every other package by its import name. It
// reports false when typ mentions a package the file does not import.
func (fl *fileLower) spellType(typ types2.Type) (string, bool) {
	complete := true
	qualify := func(p *types2.Package) string {
		if p == fl.pkg.tpkg {
			return ""
		}
		name, found := fl.importNames()[p]
		if !found {
			complete = false
			return p.Name()
		}
		return name
	}
	text := types2.TypeString(typ, qualify)
	return text, complete
}

// typeExprAt parses text as a type and moves every node of the result to
// pos. It reports false when text is not a type.
func typeExprAt(text string, pos syntax.Pos) (syntax.Expr, bool) {
	typ, ok := reparseType(text)
	if !ok {
		return nil, false
	}
	syntax.Inspect(typ, func(n syntax.Node) bool {
		if n != nil {
			n.SetPos(pos)
		}
		return true
	})
	return typ, true
}

// lowerAssertedDecls unwraps every asserted struct of list and adds the
// assertion declarations after the type declaration, or after its group.
// It returns the declarations to use in place of list.
func (fl *fileLower) lowerAssertedDecls(list []syntax.Decl) []syntax.Decl {
	out := make([]syntax.Decl, 0, len(list))
	for _, decl := range list {
		out = append(out, decl)
		td, ok := decl.(*syntax.TypeDecl)
		if !ok {
			continue
		}
		asserted, ok := td.Type.(*syntax.AssertedStruct)
		if !ok {
			continue
		}
		td.Type = asserted.Struct
		out = append(out, fl.lowerAsserts(td, asserted)...)
	}
	return out
}

// lowerAsserts builds `var _ I = (*S)(nil)` for each assertion of the
// declaration td and records the eject edits: the assertion list leaves
// the struct header, and the declarations follow the type declaration.
func (fl *fileLower) lowerAsserts(td *syntax.TypeDecl, asserted *syntax.AssertedStruct) []syntax.Decl {
	pos := td.Pos()
	name := td.Name.Value
	decls := make([]syntax.Decl, len(asserted.Asserts))
	var text string
	for i, iface := range asserted.Asserts {
		pointer := at(&syntax.ParenExpr{X: at(&syntax.Operation{Op: syntax.Mul, X: nameOf(pos, name)}, pos)}, pos)
		decls[i] = at(&syntax.VarDecl{
			NameList: []*syntax.Name{nameOf(pos, "_")},
			Type:     iface,
			Values:   callOf(pos, pointer, []syntax.Expr{nilOf(pos)}),
		}, pos)
		text += "\n\nvar _ " + renderExpr(iface) + " = (*" + name + ")(nil)"
	}

	fl.add(fl.off(asserted.Colon), fl.off(asserted.Lbrace), "")
	end := stmtEnd(fl.src, fl.off(pos))
	if td.Group != nil {
		end = groupEnd(fl.src, fl.off(pos))
	}
	fl.add(end, end, text)
	return decls
}

// groupEnd returns the offset just past the parenthesis that closes the
// declaration group which holds the declaration starting at from.
func groupEnd(src []byte, from int) int {
	end := len(src)
	depth := 0
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RBRACK, token.RBRACE:
			depth--
		case token.RPAREN:
			if depth == 0 {
				end = t.end
				return false
			}
			depth--
		}
		return true
	})
	return end
}
