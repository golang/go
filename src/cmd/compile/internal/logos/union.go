// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"fmt"
	"go/token"
	"reflect"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

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
}

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

func (fl *fileLower) lowerSwitchExpr(e *syntax.SwitchExpr) syntax.Expr {
	if isTryInit(e) {
		base.Fatalf("logos: switch expression at %s reached lowering with a try init", e.Pos().String())
	}
	pos := e.Pos()
	typ := fl.pkg.info.TypeOf(e)
	if typ == nil {
		base.Fatalf("logos: switch expression at %s has no checked type", pos.String())
	}
	resultType, text, ok := fl.spellLoweredBoth(typ, pos)
	if !ok {
		base.Fatalf("logos: switch expression at %s has type %s that cannot be spelled in this file", pos.String(), typ)
	}

	closure := fl.newGroupClosure(e)
	wrap := 0
	if closure != nil {
		wrap = fl.nextWrapID()
	}
	results := fl.expandArmCases(e)
	clauses := make([]*syntax.CaseClause, 0, len(e.Arms))
	for i, arm := range e.Arms {
		if results[i].drop {
			continue
		}
		ret := at(&syntax.ReturnStmt{Results: arm.Body}, arm.Arrow)
		results[i].prefix = closure.prefix(ret)
		fl.recordArmResult(ret, typ)
		var result syntax.Stmt = ret
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

	fl.spliceSwitchExpr(e, switchExprText{text: text, unreachable: unreachable, results: results, closure: closure, wrap: wrap})
	call := callOf(pos, lit, nil)
	fl.carryType(call, e)
	closure.bind(call, span{start: fl.off(e.Pos()), end: fl.off(e.Rbrace) + 1}, wrap)
	return call
}

func hasDefaultArm(e *syntax.SwitchExpr) bool {
	for _, arm := range e.Arms {
		if arm.Cases == nil {
			return true
		}
	}
	return false
}

type switchExprText struct {
	text        string
	unreachable string
	results     []siteResult
	closure     *groupClosure
	wrap        int
}

func (fl *fileLower) spliceSwitchExpr(e *syntax.SwitchExpr, output switchExprText) {
	text, results, closure := output.text, output.results, output.closure
	head := fl.off(e.Pos())
	if closure == nil {
		fl.add(head, head+len("switch"), "func() "+text+" { switch")
	} else {
		fl.addOwned(output.wrap, head, head, closure.head("func() "+text, []string{text})+" { "+closure.prefix(nil))
	}
	lastBindingEnd := ""
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
			fl.add(arrow, result, open+results[i].prefix+"return ")
		} else {
			fl.add(arrow, after, open+results[i].prefix+"return")
		}
		if closure != nil {
			closure.armEnd(fl.exprEndNested(result))
		}
		if b == nil {
			continue
		}
		if i+1 < len(e.Arms) {
			fl.add(next, next, "}\n")
		} else {
			lastBindingEnd = "}\n"
		}
	}
	brace := fl.off(e.Rbrace)
	if closure == nil {
		fl.add(brace, brace+1, lastBindingEnd+"}"+output.afterSwitch(e))
		return
	}
	fl.addAfterExpression(brace, lastBindingEnd)
	fl.addOwned(output.wrap, brace+1, brace+1, output.afterSwitch(e))
}

func (output switchExprText) afterSwitch(e *syntax.SwitchExpr) string {
	if hasDefaultArm(e) {
		return " }()"
	}
	return "; panic(" + output.unreachable + ") }()"
}

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

func (fl *fileLower) spellType(typ types2.Type) (string, bool) {
	ok := spellableType(typ, fl.pkg.tpkg)
	qualify := func(p *types2.Package) string {
		if p == fl.pkg.tpkg {
			return ""
		}
		if !ok {
			return p.Name()
		}
		return fl.synthQualifier(p)
	}
	text := types2.TypeString(typ, qualify)
	return text, ok
}

func (fl *fileLower) synthQualifier(p *types2.Package) string {
	reg := fl.getGenImports()
	reg.pkgs[p.Path()] = p
	if q, found := reg.qual[p.Path()]; found {
		return q
	}
	for prior, q := range fl.importNames() {
		if prior.Path() == p.Path() && q != "" {
			reg.qual[p.Path()] = q
			reg.used[q] = true
			if ownedEjectImports != nil {
				if ownedEjectImports[fl.file] == nil {
					ownedEjectImports[fl.file] = make(map[string]string)
				}
				ownedEjectImports[fl.file][p.Path()] = q
			}
			return q
		}
	}
	if existing, ok := fl.existingImportName(p.Path()); ok && existing != "" {
		reg.qual[p.Path()] = existing
		reg.used[existing] = true
		if ownedEjectImports != nil {
			if ownedEjectImports[fl.file] == nil {
				ownedEjectImports[fl.file] = make(map[string]string)
			}
			ownedEjectImports[fl.file][p.Path()] = existing
		}
		return existing
	}
	q := interpUnsafePkg
	if p != types2.Unsafe {
		for i := 0; ; i++ {
			q = fmt.Sprintf("_logos_t%d", i)
			if !reg.used[q] {
				break
			}
		}
	}
	reg.used[q] = true
	reg.qual[p.Path()] = q
	reg.order = append(reg.order, p.Path())
	if ownedEjectImports != nil {
		if ownedEjectImports[fl.file] == nil {
			ownedEjectImports[fl.file] = make(map[string]string)
		}
		ownedEjectImports[fl.file][p.Path()] = q
	}
	return q
}

func containsIdent(s, ident string) bool {
	idx := 0
	for {
		i := strings.Index(s[idx:], ident)
		if i < 0 {
			return false
		}
		pos := idx + i
		beforeOk := pos == 0 || !isIdentChar(s[pos-1])
		afterPos := pos + len(ident)
		afterOk := afterPos >= len(s) || !isIdentChar(s[afterPos])
		if beforeOk && afterOk {
			return true
		}
		idx = pos + len(ident)
	}
}

func isIdentChar(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9') || b == '_'
}

func (fl *fileLower) flushTypeImports() {
	reg := fl.genImps
	if reg == nil || len(reg.order) == 0 {
		return
	}
	nativeUsed := fl.cfNamesInUse()
	var ejectTexts []string
	if fl.constText != "" {
		ejectTexts = append(ejectTexts, fl.constText)
	}
	for _, sp := range fl.splices {
		if len(sp.text) > 0 {
			ejectTexts = append(ejectTexts, string(sp.text))
		}
	}
	aliasInEject := func(alias string) bool {
		for _, text := range ejectTexts {
			if containsIdent(text, alias) {
				return true
			}
		}
		return false
	}

	pos := fl.file.Pos()
	var decls []syntax.Decl
	var text string
	seen := make(map[string]bool)
	for _, p := range reg.order {
		if seen[p] {
			continue
		}
		seen[p] = true
		q := reg.qual[p]
		if q == "" {
			continue
		}
		if existing, ok := fl.existingImportName(p); ok && existing == q {
			continue
		}
		if !nativeUsed[q] && !aliasInEject(q) {
			continue
		}
		decls = append(decls, at(&syntax.ImportDecl{
			LocalPkgName: nameOf(pos, q),
			Path:         litOf(pos, fmt.Sprintf("%q", p), syntax.StringLit),
		}, pos))
		text += "\n\nimport " + q + " " + fmt.Sprintf("%q", p)
	}
	if len(decls) > 0 {
		fl.file.DeclList = append(decls, fl.file.DeclList...)
	}
	if text != "" {
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
}

var ownedTypeBridge func(fl *fileLower, t types2.Type, pos syntax.Pos) (native syntax.Expr, eject string, ok bool)

var ownedFieldBridge func(fl *fileLower, e *syntax.OptSelectExpr, defNative syntax.Expr, defEject string) (nativeAcc syntax.Expr, ejectSuffix string, ok bool)

func (fl *fileLower) spellTypeBoth(typ types2.Type, pos syntax.Pos) (native syntax.Expr, eject string, ok bool) {
	if ownedTypeBridge != nil {
		if n, e, ok := ownedTypeBridge(fl, typ, pos); ok {
			return n, e, true
		}
	}
	s, ok := fl.spellType(typ)
	if !ok {
		return nil, "", false
	}
	n, ok := typeExprAt(s, pos)
	if !ok {
		return nil, "", false
	}
	return n, s, true
}

func spellableType(typ types2.Type, self *types2.Package) bool {
	return typeNamesAllow(typ, func(obj *types2.TypeName) bool { return spellableObject(obj, self) })
}

func hoistableType(typ types2.Type, self *types2.Package) bool {
	return typeNamesAllow(typ, func(obj *types2.TypeName) bool {
		return spellableObject(obj, self) && (obj.Pkg() == nil || obj.Parent() == obj.Pkg().Scope())
	})
}

func typeNamesAllow(typ types2.Type, allow func(*types2.TypeName) bool) bool {
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
			if obj := t.Obj(); obj != nil && !allow(obj) {
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
			if obj := t.Obj(); obj != nil && !allow(obj) {
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

func spellableObject(obj *types2.TypeName, self *types2.Package) bool {
	pkg := obj.Pkg()
	return pkg == nil || pkg == self || obj.Exported()
}

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

func (fl *fileLower) lowerAsserts(td *syntax.TypeDecl, asserted *syntax.AssertedStruct) []syntax.Decl {
	pos := td.Pos()
	decls := make([]syntax.Decl, len(asserted.Asserts))
	var text string
	for i, iface := range asserted.Asserts {
		var line string
		decls[i], line = fl.assertDecl(td, iface)
		text += "\n\n" + line
	}

	fl.add(fl.off(asserted.Colon), fl.off(asserted.Lbrace), "")
	end := stmtEnd(fl.src, fl.off(pos))
	if td.Group != nil {
		end = groupEnd(fl.src, fl.off(pos))
	}
	fl.add(end, end, text)
	return decls
}

func (fl *fileLower) assertDecl(td *syntax.TypeDecl, iface syntax.Expr) (syntax.Decl, string) {
	pos := td.Pos()
	self := selfShape(td, pos)
	pointer := at(&syntax.ParenExpr{X: at(&syntax.Operation{Op: syntax.Mul, X: self}, pos)}, pos)
	check := at(&syntax.VarDecl{
		NameList: []*syntax.Name{nameOf(pos, "_")},
		Type:     iface,
		Values:   callOf(pos, pointer, []syntax.Expr{nilOf(pos)}),
	}, pos)
	line := "var _ " + renderExpr(iface) + " = (*" + renderExpr(self) + ")(nil)"
	if len(td.TParamList) == 0 {
		return check, line
	}
	fn := at(&syntax.FuncDecl{
		Name:       nameOf(pos, "_"),
		TParamList: sharedTParams(td, pos),
		Type:       at(&syntax.FuncType{}, pos),
		Body:       blockOf(pos, []syntax.Stmt{at(&syntax.DeclStmt{DeclList: []syntax.Decl{check}}, pos)}),
	}, pos)
	return fn, "func _" + fl.tparamText(td) + "() {\n\t" + line + "\n}"
}

func sharedTParams(td *syntax.TypeDecl, pos syntax.Pos) []*syntax.Field {
	out := make([]*syntax.Field, len(td.TParamList))
	for i, p := range td.TParamList {
		out[i] = at(&syntax.Field{Name: nameOf(pos, p.Name.Value), Type: p.Type}, pos)
	}
	return out
}

func (fl *fileLower) tparamText(td *syntax.TypeDecl) string {
	parts := make([]string, len(td.TParamList))
	for i, p := range td.TParamList {
		start := fl.off(leftmostOf(p.Type))
		parts[i] = p.Name.Value + " " + strings.TrimSpace(string(fl.src[start:fl.exprEndNested(start)]))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

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

type unionUse struct {
	suffix  string
	union   string
	members []types2.Type
	texts   []string
	utext   string
	nnodes  []syntax.Expr
	nnode   syntax.Expr
	from    bool
	locals  []string
}

func (u *unionUse) isName() string     { return "_logos_is_" + u.suffix }
func (u *unionUse) assertName() string { return "_logos_assert_" + u.suffix }
func (u *unionUse) assert2Name() string {
	return "_logos_assert2_" + u.suffix
}

func (u *unionUse) fromName() string  { return "_logos_assertfrom_" + u.suffix }
func (u *unionUse) from2Name() string { return "_logos_assertfrom2_" + u.suffix }

func (u *unionUse) tparams(pos syntax.Pos, extra ...string) []*syntax.Field {
	var out []*syntax.Field
	for _, name := range append(slices.Clone(u.locals), extra...) {
		out = append(out, at(&syntax.Field{Name: nameOf(pos, name), Type: nameOf(pos, "any")}, pos))
	}
	return out
}

func (u *unionUse) ref(pos syntax.Pos, name string, extra syntax.Expr, extraText string) (syntax.Expr, string) {
	args := make([]syntax.Expr, 0, len(u.locals)+1)
	texts := slices.Clone(u.locals)
	for _, local := range u.locals {
		args = append(args, nameOf(pos, local))
	}
	if extra != nil {
		args = append(args, extra)
		texts = append(texts, extraText)
	}
	if len(args) == 0 {
		return nameOf(pos, name), name
	}
	return at(&syntax.IndexExpr{X: nameOf(pos, name), Index: listOf(pos, args)}, pos), name + "[" + strings.Join(texts, ", ") + "]"
}

func localTypeNames(ts []types2.Type) []*types2.TypeName {
	var out []*types2.TypeName
	for _, t := range ts {
		typeNamesAllow(t, func(obj *types2.TypeName) bool {
			if obj.Pkg() != nil && obj.Parent() != obj.Pkg().Scope() && !slices.Contains(out, obj) {
				out = append(out, obj)
			}
			return true
		})
	}
	return out
}

type unionPkgState struct {
	byFile map[*syntax.File]map[string]*unionUse
	names  map[string]bool
}

var logosUnionPkgs = make(map[*types2.Package]*unionPkgState)

var logosUnionUses = make(map[*syntax.File][]*unionUse)

func (fl *fileLower) lowerUnionMembers() {
	fl.lowerUnionAsserts()
	fl.lowerUnionCases()
	uses := logosUnionUses[fl.file]
	delete(logosUnionUses, fl.file)
	for _, u := range uses {
		fl.emitHelpers(u)
	}
}

func bareUnions(pkg *types2.Package) []string {
	var out []string
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		tn, ok := scope.Lookup(name).(*types2.TypeName)
		if !ok || tn.IsAlias() {
			continue
		}
		n, ok := tn.Type().(*types2.Named)
		if !ok {
			continue
		}
		if _, isUnion := types2.LogosUnionTerms(n); isUnion && types2.LogosOptionIsBare(n) {
			out = append(out, name)
		}
	}
	return out
}

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

func (fl *fileLower) useOf(t types2.Type, pos syntax.Pos) *unionUse {
	if t == nil {
		return nil
	}
	members, ok := types2.LogosUnionTerms(t)
	if !ok || mentionsTypeParam(t) {
		return nil
	}
	for _, m := range members {
		if mentionsTypeParam(m) {
			return nil
		}
	}
	locals := localTypeNames(append([]types2.Type{t}, members...))
	key := types2.TypeString(t, func(p *types2.Package) string {
		if p == nil {
			return ""
		}
		return p.Path()
	})
	for _, obj := range locals {
		key += fmt.Sprintf("#%p", obj)
	}
	st := logosUnionPkgs[fl.pkg.tpkg]
	if st == nil {
		st = &unionPkgState{byFile: make(map[*syntax.File]map[string]*unionUse), names: make(map[string]bool)}
		logosUnionPkgs[fl.pkg.tpkg] = st
	}
	if st.byFile == nil {
		st.byFile = make(map[*syntax.File]map[string]*unionUse)
	}
	slot := st.byFile[fl.file]
	if slot == nil {
		slot = make(map[string]*unionUse)
		st.byFile[fl.file] = slot
	}
	if u, ok := slot[key]; ok {
		return u
	}
	nnode, utext, ok := fl.spellTypeBoth(t, pos)
	if !ok {
		base.Fatalf("logos: union type %s at %s cannot be spelled in this file", key, pos.String())
	}
	texts := make([]string, len(members))
	nnodes := make([]syntax.Expr, len(members))
	for i, m := range members {
		nmt, mt, ok := fl.spellTypeBoth(m, pos)
		if !ok {
			base.Fatalf("logos: union member %s at %s cannot be spelled in this file", key, pos.String())
		}
		texts[i] = mt
		nnodes[i] = nmt
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
	names := make([]string, len(locals))
	for i, obj := range locals {
		names[i] = obj.Name()
	}
	u := &unionUse{suffix: suffix, union: ustr, members: members, texts: texts, utext: utext, nnodes: nnodes, nnode: nnode, locals: names}
	slot[key] = u
	logosUnionUses[fl.file] = append(logosUnionUses[fl.file], u)
	return u
}

func mustTypeExprAt(fl *fileLower, text string, pos syntax.Pos) syntax.Expr {
	e, ok := typeExprAt(text, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q", text)
	}
	return e
}

func mustTypeExprsAt(fl *fileLower, texts []string, pos syntax.Pos) []syntax.Expr {
	out := make([]syntax.Expr, len(texts))
	for i, text := range texts {
		out[i] = mustTypeExprAt(fl, text, pos)
	}
	return out
}

func (fl *fileLower) emitHelpers(u *unionUse) {
	pos := fl.file.Pos()
	fl.needRT = true
	anyField := func(name string) *syntax.Field {
		return at(&syntax.Field{Name: nameOf(pos, name), Type: nameOf(pos, "any")}, pos)
	}
	decl := func(name string, params, results []*syntax.Field, body []syntax.Stmt) *syntax.FuncDecl {
		return at(&syntax.FuncDecl{
			Name:       nameOf(pos, name),
			TParamList: u.tparams(pos),
			Type:       at(&syntax.FuncType{ParamList: params, ResultList: results}, pos),
			Body:       blockOf(pos, body),
		}, pos)
	}
	build := func(utype syntax.Expr, mtypes []syntax.Expr) []syntax.Decl {
		isCall := func(x syntax.Expr) *syntax.CallExpr {
			fn, _ := u.ref(pos, u.isName(), nil, "")
			return callOf(pos, fn, []syntax.Expr{x})
		}
		unionAssert := func(x syntax.Expr) *syntax.AssertExpr {
			return at(&syntax.AssertExpr{X: x, Type: utype}, pos)
		}

		isBody := []syntax.Stmt{
			at(&syntax.SwitchStmt{
				Tag: at(&syntax.TypeSwitchGuard{X: nameOf(pos, "v")}, pos),
				Body: []*syntax.CaseClause{at(&syntax.CaseClause{
					Cases: listOf(pos, mtypes),
					Body:  []syntax.Stmt{returnOf(pos, []syntax.Expr{nameOf(pos, "true")})},
					Colon: pos,
				}, pos)},
				Rbrace: pos,
			}, pos),
			returnOf(pos, []syntax.Expr{nameOf(pos, "false")}),
		}
		isFn := decl(u.isName(), []*syntax.Field{anyField("v")},
			[]*syntax.Field{at(&syntax.Field{Type: nameOf(pos, "bool")}, pos)}, isBody)

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
			[]*syntax.Field{at(&syntax.Field{Type: utype}, pos)},
			[]syntax.Stmt{check, returnOf(pos, []syntax.Expr{unionAssert(nameOf(pos, "x"))})})

		hit := returnOf(pos, []syntax.Expr{unionAssert(nameOf(pos, "x")), nameOf(pos, "true")})
		miss := []syntax.Stmt{
			typedVarDecl(pos, nameOf(pos, "z"), utype),
			returnOf(pos, []syntax.Expr{nameOf(pos, "z"), nameOf(pos, "false")}),
		}
		assert2Fn := decl(u.assert2Name(), []*syntax.Field{anyField("x")},
			[]*syntax.Field{
				at(&syntax.Field{Type: utype}, pos),
				at(&syntax.Field{Type: nameOf(pos, "bool")}, pos),
			},
			[]syntax.Stmt{ifOf(pos, nil, isCall(nameOf(pos, "x")), []syntax.Stmt{hit}), miss[0], miss[1]})
		decls := []syntax.Decl{isFn, assertFn, assert2Fn}
		if u.from {
			decls = append(decls, fl.buildFromDecls(u, utype, pos)...)
		}
		return decls
	}
	nativeDecls := build(u.nnode, u.nnodes)
	ejectDecls := build(mustTypeExprAt(fl, u.utext, pos), mustTypeExprsAt(fl, u.texts, pos))
	fl.file.DeclList = append(fl.file.DeclList, nativeDecls...)
	var text strings.Builder
	for _, d := range ejectDecls {
		text.WriteString("\n\n")
		var buf bytes.Buffer
		_, _ = syntax.Fprint(&buf, d, 0)
		text.WriteString(buf.String())
		text.WriteString("\n")
	}
	fl.add(fl.orig, fl.orig, text.String())
}

func (fl *fileLower) buildFromDecls(u *unionUse, utype syntax.Expr, pos syntax.Pos) []syntax.Decl {
	const tname = "_logos_T"
	if len(u.locals) > 0 {
		utype = nameOf(pos, "any")
	}
	params := func() []*syntax.Field {
		return []*syntax.Field{at(&syntax.Field{Name: nameOf(pos, "x"), Type: utype}, pos)}
	}
	assertX := func() *syntax.AssertExpr {
		return at(&syntax.AssertExpr{X: nameOf(pos, "x"), Type: nameOf(pos, tname)}, pos)
	}
	nonMember := func(miss syntax.Stmt) []syntax.Stmt {
		isFn, _ := u.ref(pos, u.isName(), nil, "")
		dyn := callOf(pos, nameOf(pos, "any"), []syntax.Expr{nameOf(pos, "z")})
		isNon := at(&syntax.Operation{
			Op: syntax.AndAnd,
			X:  at(&syntax.Operation{Op: syntax.Neq, X: nameOf(pos, "v"), Y: nilOf(pos)}, pos),
			Y:  at(&syntax.Operation{Op: syntax.Not, X: callOf(pos, isFn, []syntax.Expr{nameOf(pos, "v")})}, pos),
		}, pos)
		return []syntax.Stmt{
			typedVarDecl(pos, nameOf(pos, "z"), nameOf(pos, tname)),
			ifOf(pos, defineOf(pos, []syntax.Expr{nameOf(pos, "v")}, []syntax.Expr{dyn}), isNon, []syntax.Stmt{miss}),
		}
	}
	typeName := callOf(pos, at(&syntax.IndexExpr{X: rtSel(pos, "TypeName"), Index: nameOf(pos, tname)}, pos), nil)
	errLit := at(&syntax.CompositeLit{
		Type: rtSel(pos, "UnionAssertionError"),
		ElemList: []syntax.Expr{
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Value"), Value: nameOf(pos, "x")}, pos),
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Union"), Value: litOf(pos, fmt.Sprintf("%q", u.union), syntax.StringLit)}, pos),
			at(&syntax.KeyValueExpr{Key: nameOf(pos, "Asserted"), Value: typeName}, pos),
		},
		NKeys:  3,
		Rbrace: pos,
	}, pos)
	panicCall := exprStmtOf(pos, callOf(pos, nameOf(pos, "panic"), []syntax.Expr{
		at(&syntax.Operation{Op: syntax.And, X: errLit}, pos),
	}))
	from1 := at(&syntax.FuncDecl{
		Name:       nameOf(pos, u.fromName()),
		TParamList: u.tparams(pos, tname),
		Type: at(&syntax.FuncType{
			ParamList:  params(),
			ResultList: []*syntax.Field{at(&syntax.Field{Type: nameOf(pos, tname)}, pos)},
		}, pos),
		Body: blockOf(pos, append(nonMember(panicCall), returnOf(pos, []syntax.Expr{assertX()}))),
	}, pos)
	miss := returnOf(pos, []syntax.Expr{nameOf(pos, "z"), nameOf(pos, "false")})
	from2 := at(&syntax.FuncDecl{
		Name:       nameOf(pos, u.from2Name()),
		TParamList: u.tparams(pos, tname),
		Type: at(&syntax.FuncType{
			ParamList: params(),
			ResultList: []*syntax.Field{
				at(&syntax.Field{Type: nameOf(pos, tname)}, pos),
				at(&syntax.Field{Type: nameOf(pos, "bool")}, pos),
			},
		}, pos),
		Body: blockOf(pos, append(nonMember(miss),
			defineOf(pos, []syntax.Expr{nameOf(pos, "r"), nameOf(pos, "ok")}, []syntax.Expr{assertX()}),
			returnOf(pos, []syntax.Expr{nameOf(pos, "r"), nameOf(pos, "ok")}))),
	}, pos)
	return []syntax.Decl{from2, from1}
}

func (fl *fileLower) lowerUnionAsserts() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.AssignStmt:
			fl.lowerCommaOkAssign(n)
		case *syntax.VarDecl:
			fl.lowerCommaOkVar(n)
		}
		replaceUnionAssertExprs(n, fl.lowerSingleAssert)
		return true
	})
}

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

func (fl *fileLower) commaOkTarget(value syntax.Expr) (*syntax.AssertExpr, *unionUse) {
	a, ok := unparenAssert(value)
	if !ok {
		return nil, nil
	}
	u := fl.useOf(fl.pkg.info.TypeOf(a.Type), a.Pos())
	if u == nil {
		return nil, nil
	}
	return a, u
}

func (fl *fileLower) lowerCommaOkAssign(s *syntax.AssignStmt) {
	if s.Rhs == nil {
		return
	}
	if len(syntax.UnpackListExpr(s.Lhs)) != 2 {
		return
	}
	if a, u := fl.commaOkTarget(s.Rhs); a != nil {
		s.Rhs = fl.assertCall(a, u, u.assert2Name(), nil, "")
		return
	}
	if a, ok := unparenAssert(s.Rhs); ok {
		if ou := fl.fromHelper(a); ou != nil {
			s.Rhs = fl.fromCall(a, ou, ou.from2Name())
		}
	}
}

func (fl *fileLower) lowerCommaOkVar(d *syntax.VarDecl) {
	if len(d.NameList) != 2 || d.Values == nil {
		return
	}
	if a, u := fl.commaOkTarget(d.Values); a != nil {
		d.Values = fl.assertCall(a, u, u.assert2Name(), nil, "")
		return
	}
	if a, ok := unparenAssert(d.Values); ok {
		if ou := fl.fromHelper(a); ou != nil {
			d.Values = fl.fromCall(a, ou, ou.from2Name())
		}
	}
}

func (fl *fileLower) lowerSingleAssert(a *syntax.AssertExpr) syntax.Expr {
	if u := fl.useOf(fl.pkg.info.TypeOf(a.Type), a.Pos()); u != nil {
		return fl.assertCall(a, u, u.assertName(), nil, "")
	}
	if ou := fl.fromHelper(a); ou != nil {
		return fl.fromCall(a, ou, ou.fromName())
	}
	return nil
}

func (fl *fileLower) fromHelper(a *syntax.AssertExpr) *unionUse {
	t := fl.pkg.info.TypeOf(a.Type)
	if t == nil {
		return nil
	}
	if _, ok := types2.Unalias(t).(*types2.TypeParam); !ok {
		if _, ok := types2.Unalias(t).Underlying().(*types2.Interface); ok {
			return nil
		}
	}
	ot := fl.pkg.info.TypeOf(a.X)
	if ot == nil || types2.Identical(t, ot) {
		return nil
	}
	members, ok := types2.LogosUnionTerms(ot)
	if !ok {
		return nil
	}
	for _, m := range members {
		if types2.Identical(m, t) {
			return nil
		}
	}
	ou := fl.useOf(ot, a.Pos())
	if ou != nil {
		ou.from = true
	}
	return ou
}

func (fl *fileLower) fromCall(a *syntax.AssertExpr, ou *unionUse, name string) syntax.Expr {
	pos := a.Pos()
	native, eject, ok := fl.spellTypeBoth(fl.pkg.info.TypeOf(a.Type), pos)
	if !ok {
		base.Fatalf("logos: cannot spell assertion type at %s", pos.String())
	}
	return fl.assertCall(a, ou, name, native, eject)
}

func (fl *fileLower) assertCall(a *syntax.AssertExpr, u *unionUse, name string, targ syntax.Expr, targText string) syntax.Expr {
	pos := a.Pos()
	xStart := fl.off(syntax.StartPos(a.X))
	dot, end, ok := splitAssert(fl.src, xStart)
	if !ok {
		base.Fatalf("logos: cannot split assertion at %s", pos.String())
	}
	fn, head := u.ref(pos, name, targ, targText)
	fl.add(xStart, xStart, head+"(")
	fl.add(dot, end, ")")
	return callOf(pos, fn, []syntax.Expr{a.X})
}

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
			if i+1 < len(toks) && toks[i].kind == token.PERIOD && toks[i+1].kind == token.LPAREN {
				i--
				continue
			}
			return dot, end, true
		}
	}
	return 0, 0, false
}

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

type unionBind struct {
	name string
	use  *unionUse
}

func (b *unionBind) open() string {
	_, head := b.use.ref(syntax.Pos{}, b.use.assertName(), nil, "")
	return "{ " + b.name + " := " + head + "(" + b.name + "); _ = " + b.name + ";"
}

func (b *unionBind) stmts(pos syntax.Pos) []syntax.Stmt {
	fn, _ := b.use.ref(pos, b.use.assertName(), nil, "")
	call := callOf(pos, fn, []syntax.Expr{nameOf(pos, b.name)})
	return []syntax.Stmt{
		defineOf(pos, []syntax.Expr{nameOf(pos, b.name)}, []syntax.Expr{call}),
		assignOf(pos, []syntax.Expr{nameOf(pos, "_")}, []syntax.Expr{nameOf(pos, b.name)}),
	}
}

type caseSite struct {
	cases syntax.Expr
	colon syntax.Pos
}

func (fl *fileLower) lowerUnionCases() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		sw, ok := n.(*syntax.SwitchStmt)
		if !ok {
			return true
		}
		guard, ok := sw.Tag.(*syntax.TypeSwitchGuard)
		if !ok {
			return true
		}
		fl.expandSwitchStmt(sw, guard)
		return true
	})
}

func (fl *fileLower) expandSwitchStmt(sw *syntax.SwitchStmt, guard *syntax.TypeSwitchGuard) {
	sites := make([]caseSite, len(sw.Body))
	for i, cc := range sw.Body {
		sites[i] = caseSite{cases: cc.Cases, colon: cc.Colon}
	}
	results := fl.expandCaseSites(guard, sites)
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

func (fl *fileLower) wrapClause(cc *syntax.CaseClause, b *unionBind, end int) {
	stmts := append(b.stmts(cc.Colon), cc.Body...)
	cc.Body = []syntax.Stmt{blockOf(cc.Colon, stmts)}
	colonEnd := fl.off(cc.Colon) + len(":")
	fl.add(colonEnd, colonEnd, " "+b.open())
	fl.add(end, end, "}\n")
}

func (fl *fileLower) expandArmCases(e *syntax.SwitchExpr) []siteResult {
	guard, ok := e.Tag.(*syntax.TypeSwitchGuard)
	if !ok {
		return make([]siteResult, len(e.Arms))
	}
	sites := make([]caseSite, len(e.Arms))
	for i, arm := range e.Arms {
		sites[i] = caseSite{cases: arm.Cases, colon: arm.Arrow}
	}
	results := fl.expandCaseSites(guard, sites)
	for i, arm := range e.Arms {
		if results[i].cases != nil {
			arm.Cases = results[i].cases
		}
	}
	return results
}

type siteResult struct {
	cases  syntax.Expr
	bind   *unionBind
	drop   bool
	prefix string
}

func (fl *fileLower) dropSpan(s span) {
	_ = fl.dropInner(s)
	fl.add(s.start, s.end, "")
}

func (fl *fileLower) expandCaseSites(guard *syntax.TypeSwitchGuard, sites []caseSite) []siteResult {
	var seen []types2.Type
	results := make([]siteResult, len(sites))
	for i, site := range sites {
		if site.cases == nil {
			continue
		}
		results[i] = fl.expandCaseList(guard, site, caseElems(site.cases), &seen)
	}
	return results
}

func (fl *fileLower) expandCaseList(guard *syntax.TypeSwitchGuard, site caseSite, elems []syntax.Expr, seen *[]types2.Type) siteResult {
	changed := false
	for _, e := range elems {
		t := fl.pkg.info.TypeOf(e)
		if _, ok := types2.LogosUnionTerms(t); ok || (t != nil && seenIn(*seen, t)) {
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
	var texts []otext
	var first *unionUse
	insertOK := guard.Lhs != nil
	unions := 0
	for i, e := range elems {
		t := fl.pkg.info.TypeOf(e)
		var text otext
		if i < len(spans) {
			text = fl.srcUser(spans[i].start, spans[i].end)
		} else {
			text = gen(renderExpr(e))
		}
		if members, ok := types2.LogosUnionTerms(t); ok {
			u := fl.useOf(t, e.Pos())
			if u == nil {
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
				texts = append(texts, gen(u.texts[k]))
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
		return siteResult{drop: true}
	}
	fl.addT(spans[0].start, spans[len(spans)-1].end, joinO(texts, ", "))
	var bind *unionBind
	if insertOK && unions == 1 {
		bind = &unionBind{name: guard.Lhs.Value, use: first}
	}
	return siteResult{cases: listOf(site.cases.Pos(), out), bind: bind}
}

func seenIn(seen []types2.Type, t types2.Type) bool {
	for _, s := range seen {
		if types2.Identical(s, t) {
			return true
		}
	}
	return false
}

func caseElems(c syntax.Expr) []syntax.Expr {
	if l, ok := syntax.Unparen(c).(*syntax.ListExpr); ok {
		return l.ElemList
	}
	return []syntax.Expr{c}
}

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
