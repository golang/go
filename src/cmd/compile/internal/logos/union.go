// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"fmt"
	"go/token"
	"reflect"
	"strings"

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
	fl.lowerUnionMembers()
	fl.flushTypeImports()
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
		base.Fatalf("logos: switch expression at %s has type %s that cannot be spelled in this file", pos.String(), text)
	}
	resultType, ok := typeExprAt(text, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q of the switch expression at %s", text, pos.String())
	}

	results := fl.expandArmCases(e)
	clauses := make([]*syntax.CaseClause, 0, len(e.Arms))
	for i, arm := range e.Arms {
		if results[i].drop {
			continue
		}
		var result syntax.Stmt = at(&syntax.ReturnStmt{Results: arm.Body}, arm.Arrow)
		if b := results[i].bind; b != nil {
			result = blockOf(arm.Arrow, append(b.stmts(arm.Arrow), result))
		}
		clauses = append(clauses, at(&syntax.CaseClause{
			Cases: arm.Cases,
			Body:  []syntax.Stmt{result},
			Colon: arm.Arrow,
		}, arm.Pos()))
	}
	body := []syntax.Stmt{
		// Init is guaranteed try-free: the checker rejects `?` in a
		// switch-expression init, since this literal has no error
		// result to propagate to.
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

	fl.spliceSwitchExpr(e, text, unreachable, results)
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
// at the brace. A bound arm opens its block at `=>` and closes it before
// the next arm, or before the final brace when it is the last arm. A
// dropped arm is deleted, from its `case` to the next arm or the brace.
func (fl *fileLower) spliceSwitchExpr(e *syntax.SwitchExpr, text, unreachable string, results []siteResult) {
	head := fl.off(e.Pos())
	fl.add(head, head+len("switch"), "func() "+text+" { switch")
	closeLast := false
	for i, arm := range e.Arms {
		next := fl.off(e.Rbrace)
		if i+1 < len(e.Arms) {
			next = fl.off(e.Arms[i+1].Pos())
		}
		if results[i].drop {
			fl.dropSpan(span{fl.off(arm.Pos()), next})
			continue
		}
		arrow := fl.off(arm.Arrow)
		after := arrow + len("=>")
		result := firstTokenStart(fl.src, after)
		open := ": "
		b := results[i].bind
		if b != nil {
			open += b.open() + " "
		}
		if bytes.IndexByte(fl.src[after:result], '\n') >= 0 {
			fl.add(arrow, result, open+"return ")
		} else {
			fl.add(arrow, after, open+"return")
		}
		if b == nil {
			continue
		}
		if i+1 < len(e.Arms) {
			fl.add(next, next, "}\n")
		} else {
			closeLast = true
		}
	}
	brace := fl.off(e.Rbrace)
	closing := "} }()"
	if !hasDefaultArm(e) {
		closing = "}; panic(" + unreachable + ") }()"
	}
	if closeLast {
		closing = "}\n" + closing
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

// typeImportSet collects the imports spellType synthesizes for one file:
// every package of a spelled switch-expression result type that the file
// does not import, mapped to the qualifier used for it, in first-use
// order. Lowering runs file by file, so the set is filled while the
// switch walk spells result types and drained at the end of lowerUnions.
type typeImportSet struct {
	qual  map[*types2.Package]string
	order []*types2.Package
}

// logosTypeImports holds the pending synthesized imports per file under
// lowering. It lives outside fileLower so the import machinery stays in
// this M2 file; lower.go keeps only its hook line.
var logosTypeImports = make(map[*syntax.File]*typeImportSet)

// spellType renders typ as Go source in this file: names of the package
// itself stand bare, and names from other packages always use a
// synthesized import (see synthQualifier) recorded for flushTypeImports.
// The qualifier is always fresh because the file's own import name may be
// shadowed by a local at the switch site (or be a dot import with no name
// at all), while a reserved qualifier resolves everywhere. It reports
// false only when typ mentions an unexported name of another package,
// which no import can spell.
func (fl *fileLower) spellType(typ types2.Type) (string, bool) {
	qualify := func(p *types2.Package) string {
		if p == fl.pkg.tpkg {
			return ""
		}
		// packagePrefix never calls the qualifier with a nil package,
		// so p is always a real package here.
		return fl.synthQualifier(p)
	}
	text := types2.TypeString(typ, qualify)
	return text, spellableType(typ, fl.pkg.tpkg)
}

// synthQualifier returns the qualifier spelling package p in this file,
// recording a synthesized import on first use. The qualifier is a fresh
// reserved name, so it collides with neither user identifiers (which
// cannot use the reserved prefix) nor the file's imports.
func (fl *fileLower) synthQualifier(p *types2.Package) string {
	set := logosTypeImports[fl.file]
	if set == nil {
		set = &typeImportSet{qual: make(map[*types2.Package]string)}
		logosTypeImports[fl.file] = set
	}
	if q, found := set.qual[p]; found {
		return q
	}
	used := map[string]bool{rtAlias: true}
	for _, name := range fl.importNames() {
		used[name] = true
	}
	for _, q := range set.qual {
		used[q] = true
	}
	var q string
	for i := 0; ; i++ {
		q = fmt.Sprintf("_logos_t%d", i)
		if !used[q] {
			break
		}
	}
	set.qual[p] = q
	set.order = append(set.order, p)
	return q
}

// flushTypeImports prepends one import declaration per synthesized package
// and records the matching eject-text header edit. It runs at the end of
// lowerUnions, so one import serves every switch expression of the file.
// The header splice composes with addHeaderSplice: both are zero-width
// inserts after the package clause.
func (fl *fileLower) flushTypeImports() {
	set := logosTypeImports[fl.file]
	delete(logosTypeImports, fl.file)
	if set == nil || len(set.order) == 0 {
		return
	}
	pos := fl.file.Pos()
	decls := make([]syntax.Decl, 0, len(set.order))
	var text string
	for _, p := range set.order {
		q := set.qual[p]
		decls = append(decls, at(&syntax.ImportDecl{
			LocalPkgName: nameOf(pos, q),
			Path:         litOf(pos, fmt.Sprintf("%q", p.Path()), syntax.StringLit),
		}, pos))
		text += "\n\nimport " + q + " " + fmt.Sprintf("%q", p.Path())
	}
	fl.file.DeclList = append(decls, fl.file.DeclList...)
	off := fl.off(fl.file.PkgName.Pos())
	ins := -1
	scanSegment(fl.src, off, len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		ins = t.end
		return false
	})
	if ins < 0 {
		ins = off
	}
	fl.add(ins, ins, text)
}

// spellableType reports whether typ can be spelled in generated source:
// every package-level name it mentions is either declared in self
// (spelled bare) or exported (spelled through an import, synthesized if
// needed). Basic keywords cannot be shadowed in a type position, and type
// parameters spell bare.
func spellableType(typ types2.Type, self *types2.Package) bool {
	seen := make(map[types2.Type]bool)
	var walk func(t types2.Type) bool
	walkTuple := func(t *types2.Tuple) bool {
		for i := range t.Len() {
			if !walk(t.At(i).Type()) {
				return false
			}
		}
		return true
	}
	walk = func(t types2.Type) bool {
		if t == nil || seen[t] {
			return true
		}
		seen[t] = true
		switch t := t.(type) {
		case *types2.Basic, *types2.TypeParam:
			return true
		case *types2.Alias:
			// The alias name and its arguments are what the text spells;
			// the expansion never appears, so only they matter.
			if obj := t.Obj(); obj != nil && !spellableObject(obj, self) {
				return false
			}
			if args := t.TypeArgs(); args != nil {
				for i := range args.Len() {
					if !walk(args.At(i)) {
						return false
					}
				}
			}
			return true
		case *types2.Pointer:
			return walk(t.Elem())
		case *types2.Slice:
			return walk(t.Elem())
		case *types2.Array:
			return walk(t.Elem())
		case *types2.Map:
			return walk(t.Key()) && walk(t.Elem())
		case *types2.Chan:
			return walk(t.Elem())
		case *types2.Signature:
			return walkTuple(t.Params()) && walkTuple(t.Results())
		case *types2.Tuple:
			return walkTuple(t)
		case *types2.Named:
			if obj := t.Obj(); obj != nil && !spellableObject(obj, self) {
				return false
			}
			if args := t.TypeArgs(); args != nil {
				for i := range args.Len() {
					if !walk(args.At(i)) {
						return false
					}
				}
			}
			return true
		case *types2.Interface:
			for i := range t.NumMethods() {
				if !walk(t.Method(i).Type()) {
					return false
				}
			}
			for i := range t.NumEmbeddeds() {
				if !walk(t.EmbeddedType(i)) {
					return false
				}
			}
			return true
		case *types2.Struct:
			for i := range t.NumFields() {
				if !walk(t.Field(i).Type()) {
					return false
				}
			}
			return true
		default:
			return true
		}
	}
	return walk(typ)
}

// spellableObject reports whether the package-level name obj can be
// spelled in generated source: universe and own-package names stand bare,
// other packages' names need an export.
func spellableObject(obj *types2.TypeName, self *types2.Package) bool {
	pkg := obj.Pkg()
	return pkg == nil || pkg == self || obj.Exported()
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

// M2-S4: lowering of assertions to unions (SPEC Sec 2.3, T-203.1).
//
// A union value is an upstream interface at run time, so a plain type
// switch or assertion against a methodless union would match or succeed
// for every dynamic type. The lowering restores membership checking:
//
//   - per union used in an assertion or case position, one helper triple:
//     func _logos_is_U(v any) bool,
//     func _logos_assert_U(x any) U, and
//     func _logos_assert2_U(x any) (U, bool);
//   - x.(U) becomes _logos_assert_U(x),
//     v, ok := x.(U) becomes v, ok := _logos_assert2_U(x), and
//   - case U: becomes case <members>: with v := v.(U) first in the clause.
//
// The helpers are emitted once per package: Lower visits the files of a
// package in order, and the first file that names a union carries its
// declarations. Every rewrite builds the native tree and the eject text
// together, like the switch-expression lowering above.

// unionUse is one closed union named in an assertion or case position.
type unionUse struct {
	suffix  string        // helper-name suffix, unique per package
	union   string        // Union field: "<pkg>.<name>"
	members []types2.Type // direct members in declaration order
	texts   []string      // spelled members, aligned with members
	utext   string        // spelled union type
}

// isName returns the membership-test helper name of u.
func (u *unionUse) isName() string     { return "_logos_is_" + u.suffix }
func (u *unionUse) assertName() string { return "_logos_assert_" + u.suffix }
func (u *unionUse) assert2Name() string {
	return "_logos_assert2_" + u.suffix
}

// unionPkgState tracks the unions one package already emitted helpers for.
type unionPkgState struct {
	byKey map[string]*unionUse
	names map[string]bool
}

// logosUnionPkgs records the helper suffixes emitted per package under
// lowering.
var logosUnionPkgs = make(map[*types2.Package]*unionPkgState)

// unionLower carries the per-file state of the M2-S4 lowering.
type unionLower struct {
	fl *fileLower
}

// logosUnionUses holds, per file under lowering, the unions first named
// there; the file emits their helpers. lowerUnionMembers drains it.
var logosUnionUses = make(map[*syntax.File][]*unionUse)

// lowerUnionMembers lowers union assertions and union case types of one
// file, then emits the helpers of the unions the file introduced. Helpers
// come last, so no walk ever visits generated declarations.
func (fl *fileLower) lowerUnionMembers() {
	ul := &unionLower{fl: fl}
	ul.lowerUnionAsserts()
	ul.lowerUnionCases()
	uses := logosUnionUses[fl.file]
	delete(logosUnionUses, fl.file)
	for _, u := range uses {
		ul.emitHelpers(u)
	}
}

// unionTerms returns the direct members of t in declaration order when t
// is a union of non-~ terms, and reports false otherwise. Members that
// are themselves unions stay whole: a plain case naming them matches
// their members at run time.
func unionTerms(t types2.Type) ([]types2.Type, bool) {
	if t == nil {
		return nil, false
	}
	iface, ok := types2.Unalias(t).Underlying().(*types2.Interface)
	if !ok {
		return nil, false
	}
	var out []types2.Type
	for i := range iface.NumEmbeddeds() {
		u, ok := iface.EmbeddedType(i).(*types2.Union)
		if !ok {
			continue
		}
		for j := range u.Len() {
			term := u.Term(j)
			if term.Tilde() {
				return nil, false
			}
			out = append(out, term.Type())
		}
	}
	if len(out) == 0 {
		return nil, false
	}
	return out, true
}

// mentionsTypeParam reports whether t mentions a type parameter. The
// package-level helpers cannot name one, so such unions are not lowered.
func mentionsTypeParam(t types2.Type) bool {
	seen := make(map[types2.Type]bool)
	var walk func(t types2.Type) bool
	walkTuple := func(t *types2.Tuple) bool {
		for i := range t.Len() {
			if walk(t.At(i).Type()) {
				return true
			}
		}
		return false
	}
	walk = func(t types2.Type) bool {
		if t == nil || seen[t] {
			return false
		}
		seen[t] = true
		if _, ok := types2.Unalias(t).(*types2.TypeParam); ok {
			return true
		}
		switch t := t.(type) {
		case *types2.Basic:
			return false
		case *types2.Alias:
			if args := t.TypeArgs(); args != nil {
				for i := range args.Len() {
					if walk(args.At(i)) {
						return true
					}
				}
			}
			return false
		case *types2.Pointer:
			return walk(t.Elem())
		case *types2.Slice:
			return walk(t.Elem())
		case *types2.Array:
			return walk(t.Elem())
		case *types2.Map:
			return walk(t.Key()) || walk(t.Elem())
		case *types2.Chan:
			return walk(t.Elem())
		case *types2.Signature:
			return walkTuple(t.Params()) || walkTuple(t.Results())
		case *types2.Tuple:
			return walkTuple(t)
		case *types2.Named:
			if args := t.TypeArgs(); args != nil {
				for i := range args.Len() {
					if walk(args.At(i)) {
						return true
					}
				}
			}
			return false
		case *types2.Interface:
			for i := range t.NumMethods() {
				if walk(t.Method(i).Type()) {
					return true
				}
			}
			for i := range t.NumEmbeddeds() {
				if walk(t.EmbeddedType(i)) {
					return true
				}
			}
			return false
		case *types2.Struct:
			for i := range t.NumFields() {
				if walk(t.Field(i).Type()) {
					return true
				}
			}
			return false
		case *types2.Union:
			for i := range t.Len() {
				if walk(t.Term(i).Type()) {
					return true
				}
			}
			return false
		default:
			return false
		}
	}
	return walk(t)
}

// sanitizeUnionName maps arbitrary text to a helper-name suffix.
func sanitizeUnionName(text string) string {
	var b strings.Builder
	for _, r := range text {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "U"
	}
	return b.String()
}

// useOf returns the unionUse for t, registering it for helper emission.
// It returns nil when t is not a lowerable closed union.
func (ul *unionLower) useOf(t types2.Type, pos syntax.Pos) *unionUse {
	if t == nil {
		return nil
	}
	fl := ul.fl
	key := types2.TypeString(t, func(p *types2.Package) string {
		if p == nil {
			return ""
		}
		return p.Path()
	})
	st := logosUnionPkgs[fl.pkg.tpkg]
	if st == nil {
		st = &unionPkgState{byKey: make(map[string]*unionUse), names: make(map[string]bool)}
		logosUnionPkgs[fl.pkg.tpkg] = st
	}
	if u, ok := st.byKey[key]; ok {
		return u
	}
	members, ok := unionTerms(t)
	if !ok || mentionsTypeParam(t) {
		return nil
	}
	for _, m := range members {
		if mentionsTypeParam(m) {
			return nil
		}
	}
	utext, ok := fl.spellType(t)
	if !ok {
		base.Fatalf("logos: union type %s at %s cannot be spelled in this file", key, pos.String())
	}
	texts := make([]string, len(members))
	for i, m := range members {
		mt, ok := fl.spellType(m)
		if !ok {
			base.Fatalf("logos: union member %s at %s cannot be spelled in this file", key, pos.String())
		}
		texts[i] = mt
	}
	var stem, ustr string
	if named, ok := types2.Unalias(t).(*types2.Named); ok {
		obj := named.Obj()
		stem = obj.Name()
		if pkg := obj.Pkg(); pkg != nil {
			ustr = pkg.Name() + "." + obj.Name()
		} else {
			ustr = obj.Name()
		}
	} else {
		stem = sanitizeUnionName(utext)
		ustr = utext
	}
	suffix := stem
	for i := 2; st.names[suffix]; i++ {
		suffix = fmt.Sprintf("%s_%d", stem, i)
	}
	st.names[suffix] = true
	u := &unionUse{suffix: suffix, union: ustr, members: members, texts: texts, utext: utext}
	st.byKey[key] = u
	logosUnionUses[fl.file] = append(logosUnionUses[fl.file], u)
	return u
}

// emitHelpers appends the helper triple of u to the file, in the native
// tree and the eject text.
func (ul *unionLower) emitHelpers(u *unionUse) {
	fl := ul.fl
	pos := fl.file.Pos()
	fl.needRT = true // _logos_assert_U panics with _logos_rt.UnionAssertionError
	texpr := func(text string) syntax.Expr {
		e, ok := typeExprAt(text, pos)
		if !ok {
			base.Fatalf("logos: cannot respell type %q", text)
		}
		return e
	}
	anyField := func(name string) *syntax.Field {
		return at(&syntax.Field{Name: nameOf(pos, name), Type: nameOf(pos, "any")}, pos)
	}
	decl := func(name string, params, results []*syntax.Field, body []syntax.Stmt) *syntax.FuncDecl {
		return at(&syntax.FuncDecl{
			Name: nameOf(pos, name),
			Type: at(&syntax.FuncType{ParamList: params, ResultList: results}, pos),
			Body: blockOf(pos, body),
		}, pos)
	}
	memberCases := func() []syntax.Expr {
		out := make([]syntax.Expr, len(u.texts))
		for i, mt := range u.texts {
			out[i] = texpr(mt)
		}
		return out
	}
	isCall := func(x syntax.Expr) *syntax.CallExpr {
		return callOf(pos, nameOf(pos, u.isName()), []syntax.Expr{x})
	}
	unionAssert := func(x syntax.Expr) *syntax.AssertExpr {
		return at(&syntax.AssertExpr{X: x, Type: texpr(u.utext)}, pos)
	}

	// func _logos_is_U(v any) bool {
	//   switch v.(type) { case <members>: return true }
	//   return false
	// }
	isBody := []syntax.Stmt{
		at(&syntax.SwitchStmt{
			Tag: at(&syntax.TypeSwitchGuard{X: nameOf(pos, "v")}, pos),
			Body: []*syntax.CaseClause{at(&syntax.CaseClause{
				Cases: listOf(pos, memberCases()),
				Body:  []syntax.Stmt{returnOf(pos, []syntax.Expr{nameOf(pos, "true")})},
				Colon: pos,
			}, pos)},
			Rbrace: pos,
		}, pos),
		returnOf(pos, []syntax.Expr{nameOf(pos, "false")}),
	}
	isFn := decl(u.isName(), []*syntax.Field{anyField("v")},
		[]*syntax.Field{at(&syntax.Field{Type: nameOf(pos, "bool")}, pos)}, isBody)

	// func _logos_assert_U(x any) U {
	//   if !_logos_is_U(x) {
	//     panic(&_logos_rt.UnionAssertionError{Value: x, Union: "<pkg>.U"})
	//   }
	//   return x.(U)
	// }
	errLit := at(&syntax.CompositeLit{
		Type: rtSel(pos, "UnionAssertionError"),
		ElemList: []syntax.Expr{
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Value"), Value: nameOf(pos, "x")}, pos),
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Union"), Value: litOf(pos, fmt.Sprintf("%q", u.union), syntax.StringLit)}, pos),
		},
		NKeys:  2,
		Rbrace: pos,
	}, pos)
	panicCall := callOf(pos, nameOf(pos, "panic"), []syntax.Expr{
		at(&syntax.Operation{Op: syntax.And, X: errLit}, pos),
	})
	check := ifOf(pos, nil,
		at(&syntax.Operation{Op: syntax.Not, X: isCall(nameOf(pos, "x"))}, pos),
		[]syntax.Stmt{exprStmtOf(pos, panicCall)})
	assertFn := decl(u.assertName(), []*syntax.Field{anyField("x")},
		[]*syntax.Field{at(&syntax.Field{Type: texpr(u.utext)}, pos)},
		[]syntax.Stmt{check, returnOf(pos, []syntax.Expr{unionAssert(nameOf(pos, "x"))})})

	// func _logos_assert2_U(x any) (U, bool) {
	//   if _logos_is_U(x) { return x.(U), true }
	//   var z U
	//   return z, false
	// }
	hit := returnOf(pos, []syntax.Expr{unionAssert(nameOf(pos, "x")), nameOf(pos, "true")})
	miss := []syntax.Stmt{
		typedVarDecl(pos, nameOf(pos, "z"), texpr(u.utext)),
		returnOf(pos, []syntax.Expr{nameOf(pos, "z"), nameOf(pos, "false")}),
	}
	assert2Fn := decl(u.assert2Name(), []*syntax.Field{anyField("x")},
		[]*syntax.Field{
			at(&syntax.Field{Type: texpr(u.utext)}, pos),
			at(&syntax.Field{Type: nameOf(pos, "bool")}, pos),
		},
		[]syntax.Stmt{ifOf(pos, nil, isCall(nameOf(pos, "x")), []syntax.Stmt{hit}), miss[0], miss[1]})

	fl.file.DeclList = append(fl.file.DeclList, isFn, assertFn, assert2Fn)
	var text strings.Builder
	for _, d := range []syntax.Decl{isFn, assertFn, assert2Fn} {
		text.WriteString("\n\n")
		var buf bytes.Buffer
		_, _ = syntax.Fprint(&buf, d, 0)
		text.WriteString(buf.String())
		text.WriteString("\n")
	}
	fl.add(fl.orig, fl.orig, text.String())
}

// lowerUnionAsserts rewrites x.(U) and v, ok := x.(U) for closed unions.
func (ul *unionLower) lowerUnionAsserts() {
	fl := ul.fl
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.AssignStmt:
			ul.lowerCommaOkAssign(n)
		case *syntax.VarDecl:
			ul.lowerCommaOkVar(n)
		}
		replaceUnionAssertExprs(n, ul.lowerSingleAssert)
		return true
	})
}

// unparenAssert unwraps parentheses around an assertion operand.
func unparenAssert(e syntax.Expr) (*syntax.AssertExpr, bool) {
	for {
		p, ok := e.(*syntax.ParenExpr)
		if !ok {
			break
		}
		e = p.X
	}
	a, ok := e.(*syntax.AssertExpr)
	return a, ok
}

// commaOkTarget returns the assertion and union of value when value is a
// single x.(U) naming a lowerable closed union.
func (ul *unionLower) commaOkTarget(value syntax.Expr) (*syntax.AssertExpr, *unionUse) {
	a, ok := unparenAssert(value)
	if !ok {
		return nil, nil
	}
	u := ul.useOf(ul.fl.pkg.info.TypeOf(a.Type), a.Pos())
	if u == nil {
		return nil, nil
	}
	return a, u
}

// lowerCommaOkAssign rewrites v, ok := x.(U) to v, ok := _logos_assert2_U(x).
func (ul *unionLower) lowerCommaOkAssign(s *syntax.AssignStmt) {
	if s.Rhs == nil {
		return
	}
	if len(syntax.UnpackListExpr(s.Lhs)) != 2 {
		return
	}
	if a, u := ul.commaOkTarget(s.Rhs); a != nil {
		s.Rhs = ul.assertCall(a, u.assert2Name())
	}
}

// lowerCommaOkVar rewrites var v, ok = x.(U) to var v, ok = _logos_assert2_U(x).
func (ul *unionLower) lowerCommaOkVar(d *syntax.VarDecl) {
	if len(d.NameList) != 2 || d.Values == nil {
		return
	}
	if a, u := ul.commaOkTarget(d.Values); a != nil {
		d.Values = ul.assertCall(a, u.assert2Name())
	}
}

// lowerSingleAssert rewrites x.(U) to _logos_assert_U(x), or returns nil
// when the assertion does not target a lowerable closed union.
func (ul *unionLower) lowerSingleAssert(a *syntax.AssertExpr) syntax.Expr {
	u := ul.useOf(ul.fl.pkg.info.TypeOf(a.Type), a.Pos())
	if u == nil {
		return nil
	}
	return ul.assertCall(a, u.assertName())
}

// assertCall builds name(x) and records the eject splices turning the
// source x.(U) into that call: the operand text stays verbatim.
func (ul *unionLower) assertCall(a *syntax.AssertExpr, name string) syntax.Expr {
	fl := ul.fl
	pos := a.Pos()
	xStart := fl.off(a.X.Pos())
	dot, end, ok := splitAssert(fl.src, xStart)
	if !ok {
		base.Fatalf("logos: cannot split assertion at %s", pos.String())
	}
	fl.add(xStart, xStart, name+"(")
	fl.add(dot, end, ")")
	return callOf(pos, nameOf(pos, name), []syntax.Expr{a.X})
}

// splitAssert locates the `.(T)` of the assertion whose operand starts at
// xStart: the dot starting it and the end of the type. A chained
// assertion x.(A).(B) reports the outer span. Brackets of any kind nest.
func splitAssert(src []byte, xStart int) (dot, end int, ok bool) {
	var toks []tok
	scanSegment(src, xStart, len(src), func(t tok) bool {
		if !t.comment && !t.auto {
			toks = append(toks, t)
		}
		return true
	})
	depth := 0
	for i := 0; i < len(toks); i++ {
		t := toks[i]
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case token.PERIOD:
			if depth != 0 || i+1 >= len(toks) || toks[i+1].kind != token.LPAREN {
				continue
			}
			d, e, next, good := matchAssertParen(toks, i)
			if !good {
				return 0, 0, false
			}
			dot, end = d, e
			i = next
			// A chained assertion follows the close with `.(`
			// and nothing between; anything else ends the span.
			// Step back so the loop lands on the period again.
			if i+1 < len(toks) && toks[i].kind == token.PERIOD && toks[i+1].kind == token.LPAREN {
				i--
				continue
			}
			return dot, end, true
		}
	}
	return 0, 0, false
}

// matchAssertParen matches the parenthesis opened by the assertion dot at
// toks[i].(toks[i+1], returning the dot, the end of the close, and the
// index after it.
func matchAssertParen(toks []tok, i int) (dot, end, next int, ok bool) {
	depth := 0
	for j := i + 1; j < len(toks); j++ {
		switch toks[j].kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
			if depth == 0 {
				return toks[i].start, toks[j].end, j + 1, true
			}
		}
	}
	return 0, 0, 0, false
}

// replaceUnionAssertExprs replaces each *syntax.AssertExpr that n holds
// directly, as a field or as an element of a list field, with repl of it.
// A nil result leaves the assertion in place.
func replaceUnionAssertExprs(n syntax.Node, repl func(*syntax.AssertExpr) syntax.Expr) {
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
			replaceUnionAssertField(v.Field(i), repl)
		}
	}
}

// replaceUnionAssertField replaces a *syntax.AssertExpr held by the
// interface value f, or by an element of the slice or array f.
func replaceUnionAssertField(f reflect.Value, repl func(*syntax.AssertExpr) syntax.Expr) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() {
			return
		}
		if a, ok := f.Elem().Interface().(*syntax.AssertExpr); ok {
			if r := repl(a); r != nil {
				f.Set(reflect.ValueOf(r))
			}
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			replaceUnionAssertField(f.Index(i), repl)
		}
	}
}

// unionBind binds a type-switch guard variable to a closed union inside
// one clause or arm. The guard already declares name in the clause scope,
// so the rebind opens a nested block; a := in the clause itself would
// declare nothing.
type unionBind struct {
	name string
	use  *unionUse
}

// open returns the text that opens the block of the binding. The block
// starts with the rebind and a read of the variable, so the unused-variable
// check never fires on it.
func (b *unionBind) open() string {
	return "{ " + b.name + " := " + b.use.assertName() + "(" + b.name + "); _ = " + b.name + ";"
}

// stmts returns the native statements of the binding at pos, in the order
// open spells them.
func (b *unionBind) stmts(pos syntax.Pos) []syntax.Stmt {
	call := callOf(pos, nameOf(pos, b.use.assertName()), []syntax.Expr{nameOf(pos, b.name)})
	return []syntax.Stmt{
		defineOf(pos, []syntax.Expr{nameOf(pos, b.name)}, []syntax.Expr{call}),
		assignOf(pos, []syntax.Expr{nameOf(pos, "_")}, []syntax.Expr{nameOf(pos, b.name)}),
	}
}

// caseSite is one case list of a type switch: the list of a clause or of
// an arm, with the token that ends it. cases is nil for a default.
type caseSite struct {
	cases syntax.Expr
	colon syntax.Pos
}

// lowerUnionCases expands the union case lists of type switch statements.
// Switch expressions are expanded when they are lowered, in expandArmCases.
func (ul *unionLower) lowerUnionCases() {
	fl := ul.fl
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		sw, ok := n.(*syntax.SwitchStmt)
		if !ok {
			return true
		}
		guard, ok := sw.Tag.(*syntax.TypeSwitchGuard)
		if !ok {
			return true
		}
		ul.expandSwitchStmt(sw, guard)
		return true
	})
}

// expandSwitchStmt expands the union clauses of one type switch statement.
// A clause whose entries are all matched by earlier clauses is deleted. A
// clause that binds the guard runs its body in a block that closes before
// the next clause keyword, or before the closing brace of the switch.
func (ul *unionLower) expandSwitchStmt(sw *syntax.SwitchStmt, guard *syntax.TypeSwitchGuard) {
	fl := ul.fl
	sites := make([]caseSite, len(sw.Body))
	for i, cc := range sw.Body {
		sites[i] = caseSite{cases: cc.Cases, colon: cc.Colon}
	}
	results := ul.expandCaseSites(guard, sites)
	kept := make([]*syntax.CaseClause, 0, len(sw.Body))
	for i, cc := range sw.Body {
		end := fl.off(sw.Rbrace)
		if i+1 < len(sw.Body) {
			end = fl.off(sw.Body[i+1].Pos())
		}
		r := results[i]
		if r.drop {
			fl.dropSpan(span{fl.off(cc.Pos()), end})
			continue
		}
		kept = append(kept, cc)
		if r.cases != nil {
			cc.Cases = r.cases
		}
		if r.bind != nil {
			fl.wrapClause(cc, r.bind, end)
		}
	}
	sw.Body = kept
}

// wrapClause moves the body of cc into a block that opens with the binding
// b and closes at end, the offset of the next clause keyword or of the
// switch's closing brace.
func (fl *fileLower) wrapClause(cc *syntax.CaseClause, b *unionBind, end int) {
	stmts := append(b.stmts(cc.Colon), cc.Body...)
	cc.Body = []syntax.Stmt{blockOf(cc.Colon, stmts)}
	colonEnd := fl.off(cc.Colon) + len(":")
	fl.add(colonEnd, colonEnd, " "+b.open())
	fl.add(end, end, "}\n")
}

// expandArmCases expands the union arms of a type switch expression before
// the arms are lowered. It rewrites each arm's case list in place and
// returns the decision for each arm; a value switch expression has none.
func (fl *fileLower) expandArmCases(e *syntax.SwitchExpr) []siteResult {
	guard, ok := e.Tag.(*syntax.TypeSwitchGuard)
	if !ok {
		return make([]siteResult, len(e.Arms))
	}
	sites := make([]caseSite, len(e.Arms))
	for i, arm := range e.Arms {
		sites[i] = caseSite{cases: arm.Cases, colon: arm.Arrow}
	}
	results := (&unionLower{fl: fl}).expandCaseSites(guard, sites)
	for i, arm := range e.Arms {
		if results[i].cases != nil {
			arm.Cases = results[i].cases
		}
	}
	return results
}

// siteResult is the decision for one case list of a type switch.
type siteResult struct {
	cases syntax.Expr // the expanded list; nil when the list keeps its text
	bind  *unionBind  // the union the guard binds in this site; nil when none
	drop  bool        // every entry is matched earlier: the site is deleted
}

// dropSpan deletes the source text of s. Splices inside s describe text
// that is removed, so they are discarded; zero-width splices on the
// boundary stay, so an insertion beside the span keeps its place.
func (fl *fileLower) dropSpan(s span) {
	_ = fl.dropInner(s)
	fl.add(s.start, s.end, "")
}

// expandCaseSites rewrites the case lists of one type switch, in order, and
// returns the decision for each site. A union entry expands to its members
// not named by an earlier site. A plain entry named by an earlier site is
// dropped: first match wins, and a repeated case does not compile. A site
// left with no entries is deleted. A kept `case U:` would match non-members
// through the run-time interface, so no union entry survives as written.
func (ul *unionLower) expandCaseSites(guard *syntax.TypeSwitchGuard, sites []caseSite) []siteResult {
	var seen []types2.Type
	results := make([]siteResult, len(sites))
	for i, site := range sites {
		if site.cases == nil {
			continue
		}
		results[i] = ul.expandCaseList(guard, site, caseElems(site.cases), &seen)
	}
	return results
}

// expandCaseList decides one case list; see expandCaseSites. seen collects
// the types named so far. The binding of the guard variable exists only
// when the list names exactly one union and no kept entry that a member
// test cannot cover.
func (ul *unionLower) expandCaseList(guard *syntax.TypeSwitchGuard, site caseSite, elems []syntax.Expr, seen *[]types2.Type) siteResult {
	fl := ul.fl
	changed := false
	for _, e := range elems {
		t := fl.pkg.info.TypeOf(e)
		if _, ok := unionTerms(t); ok || (t != nil && seenIn(*seen, t)) {
			changed = true
			break
		}
	}
	if !changed {
		for _, e := range elems {
			if t := fl.pkg.info.TypeOf(e); t != nil {
				*seen = append(*seen, t)
			}
		}
		return siteResult{}
	}
	spans := caseElemSpans(fl.src, fl.off(elems[0].Pos()), fl.off(site.colon))
	var out []syntax.Expr
	var texts []string
	var first *unionUse
	insertOK := guard.Lhs != nil
	unions := 0
	for i, e := range elems {
		t := fl.pkg.info.TypeOf(e)
		var text string
		if i < len(spans) {
			text = string(fl.src[spans[i].start:spans[i].end])
		} else {
			text = renderExpr(e)
		}
		if members, ok := unionTerms(t); ok {
			u := ul.useOf(t, e.Pos())
			if u == nil {
				// Not lowerable (type parameters): keep verbatim.
				out = append(out, e)
				texts = append(texts, text)
				insertOK = false
				continue
			}
			unions++
			if first == nil {
				first = u
			} else {
				insertOK = false
			}
			for k, m := range members {
				if seenIn(*seen, m) {
					continue
				}
				*seen = append(*seen, m)
				me, ok := typeExprAt(u.texts[k], e.Pos())
				if !ok {
					base.Fatalf("logos: cannot respell type %q", u.texts[k])
				}
				out = append(out, me)
				texts = append(texts, u.texts[k])
			}
			continue
		}
		if t != nil && seenIn(*seen, t) {
			continue
		}
		out = append(out, e)
		texts = append(texts, text)
		if name, ok := syntax.Unparen(e).(*syntax.Name); ok && name.Value == "nil" {
			insertOK = false
		} else if first == nil || !memberCovered(first, t) {
			insertOK = false
		}
		if t != nil {
			*seen = append(*seen, t)
		}
	}
	if len(out) == 0 {
		// Every entry is matched earlier, so the site can never be the
		// first match. Delete it rather than keep a case that could match
		// non-members.
		return siteResult{drop: true}
	}
	fl.add(spans[0].start, spans[len(spans)-1].end, strings.Join(texts, ", "))
	var bind *unionBind
	if insertOK && unions == 1 {
		bind = &unionBind{name: guard.Lhs.Value, use: first}
	}
	return siteResult{cases: listOf(site.cases.Pos(), out), bind: bind}
}

// seenIn reports whether t is identical to one of the types in seen.
func seenIn(seen []types2.Type, t types2.Type) bool {
	for _, s := range seen {
		if types2.Identical(s, t) {
			return true
		}
	}
	return false
}

// caseElems unwraps a case list to its elements.
func caseElems(c syntax.Expr) []syntax.Expr {
	if l, ok := syntax.Unparen(c).(*syntax.ListExpr); ok {
		return l.ElemList
	}
	return []syntax.Expr{c}
}

// memberCovered reports whether t denotes a member of u.
func memberCovered(u *unionUse, t types2.Type) bool {
	if t == nil {
		return false
	}
	for _, m := range u.members {
		if types2.Identical(m, t) {
			return true
		}
	}
	return false
}

// caseElemSpans splits the case list ending at colonOff into one span per
// element, trimmed of surrounding white space.
func caseElemSpans(src []byte, firstStart, colonOff int) []span {
	var spans []span
	depth := 0
	cur := firstStart
	scanSegment(src, firstStart, colonOff, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case token.COMMA:
			if depth == 0 {
				spans = append(spans, trimSpan(src, span{cur, t.start}))
				cur = t.end
			}
		}
		return true
	})
	spans = append(spans, trimSpan(src, span{cur, colonOff}))
	return spans
}
