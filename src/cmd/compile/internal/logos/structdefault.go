// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/token"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func (fl *fileLower) lowerStructDefaults() {
	stripped := make(map[*syntax.FieldDefault]bool)
	typeNames := fl.defaultTypeNames()
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.StructType:
			fl.stripFieldDefaults(n, stripped, typeNames)
		case *syntax.CompositeLit:
			fl.fillStructDefaults(n)
		}
		return true
	})
}

// defaultTypeNames maps each package-level named struct type declaration in
// this file to its TypeDecl so defaults can bind to declaration-scope
// factories. Local or anonymous struct types get no entry: they keep the
// textual default path.
func (fl *fileLower) defaultTypeNames() map[*syntax.StructType]*syntax.TypeDecl {
	m := map[*syntax.StructType]*syntax.TypeDecl{}
	for _, d := range fl.file.DeclList {
		td, ok := d.(*syntax.TypeDecl)
		if !ok || td.Alias || td.Name == nil {
			continue
		}
		var st *syntax.StructType
		switch t := td.Type.(type) {
		case *syntax.StructType:
			st = t
		case *syntax.AssertedStruct:
			st = t.Struct
		}
		if st != nil {
			m[st] = td
		}
	}
	return m
}

// defaultFactoryName is the generated declaration-scope binding for one
// defaulted field. It is exported so ejected and cross-package consumers can
// reference it; declaration scope keeps private names and interface dynamic
// types intact while every call allocates fresh composite storage. The double
// underscore keeps (type, field) pairs like A_B.C and A.B_C distinct.
func defaultFactoryName(typeName, field string) string {
	return "LogosDefault_" + typeName + "__" + field
}

// defaultFactorySkips computes the package-level set of "Owner.Field" pairs
// that must not reference a generated factory: generated names colliding with
// each other or with an existing package-scope declaration. The set derives
// only from checker data, so every file lowering sees the same plan.
func (pl *packageLower) defaultFactorySkips() map[string]bool {
	if pl.factoryPlan != nil {
		return pl.factoryPlan
	}
	pl.factoryPlan = map[string]bool{}
	taken := map[string]string{}
	if pl.info != nil {
		for _, defs := range pl.info.LogosFieldDefaults {
			for _, d := range defs {
				if d.Owner == "" || d.Field == "_" {
					continue
				}
				key := d.Owner + "." + d.Field
				fname := defaultFactoryName(d.Owner, d.Field)
				if prev, dup := taken[fname]; dup && prev != key {
					pl.factoryPlan[key] = true
					pl.factoryPlan[prev] = true
					continue
				}
				taken[fname] = key
			}
		}
	}
	if pl.tpkg != nil && pl.tpkg.Scope() != nil {
		for fname, key := range taken {
			if pl.tpkg.Scope().Lookup(fname) != nil {
				pl.factoryPlan[key] = true
			}
		}
	}
	return pl.factoryPlan
}

// importQualifier returns the source qualifier ("pkg." or "") that names the
// given package in this file, so generated calls survive elided, generic, and
// aliased literals that carry no selector in lit.Type. synthQualifier reuses
// the file's import spellings and registers a synthesized import when needed.
func (fl *fileLower) importQualifier(pkg *types2.Package) string {
	if pkg == nil || pkg == fl.pkg.tpkg {
		return ""
	}
	q := fl.synthQualifier(pkg)
	if q == "" || q == "." {
		return ""
	}
	return q + "."
}

func (fl *fileLower) stripFieldDefaults(st *syntax.StructType, stripped map[*syntax.FieldDefault]bool, typeNames map[*syntax.StructType]*syntax.TypeDecl) {
	skips := fl.pkg.defaultFactorySkips()
	td := typeNames[st]
	var tparams string
	if td != nil && td.Name != nil && len(td.TParamList) > 0 {
		var params []string
		for _, p := range td.TParamList {
			if p.Name != nil {
				params = append(params, p.Name.Value+" "+renderExpr(p.Type))
			} else {
				params = append(params, renderExpr(p.Type))
			}
		}
		tparams = "[" + strings.Join(params, ", ") + "]"
	}
	for i, f := range st.FieldList {
		fd, ok := f.Type.(*syntax.FieldDefault)
		if !ok {
			continue
		}
		f.Type = fd.Type
		if !stripped[fd] {
			stripped[fd] = true
			typeEnd := fl.exprEndNested(fl.off(syntax.StartPos(fd.Type)))
			fl.add(typeEnd, fl.fieldDefaultEnd(st, i, fd, typeEnd), "")
		}
		if td == nil || td.Name == nil || f.Name == nil || f.Name.Value == "_" {
			continue
		}
		// Grouped fields (A, B int = v) share one FieldDefault: each name
		// still needs its own factory.
		if skips[td.Name.Value+"."+f.Name.Value] {
			continue
		}
		text := "func " + defaultFactoryName(td.Name.Value, f.Name.Value) + tparams +
			"() " + renderExpr(fd.Type) + " {\n\treturn " + renderExpr(fd.Default) + "\n}\n"
		fl.addGeneratedDecls(fd.Pos(), text)
	}
}

func (fl *fileLower) fieldDefaultEnd(st *syntax.StructType, i int, fd *syntax.FieldDefault, from int) int {
	if i < len(st.TagList) && st.TagList[i] != nil {
		return lastTokenEnd(fl.src, from, fl.off(st.TagList[i].Pos()))
	}
	return fl.exprEndOf(fd.Default)
}

type structDefault struct {
	field string
	text  string
}

func (fl *fileLower) fillStructDefaults(lit *syntax.CompositeLit) {
	if fl.hasSpread(lit) {
		// Spread lowering emits its initializer and its text together. A
		// generated default has no source span to reuse as a spread operand.
		return
	}
	missing := fl.missingDefaults(lit)
	if len(missing) == 0 {
		return
	}
	native, text := fl.structDefaultFields(lit.Pos(), missing)
	lit.ElemList = append(native, lit.ElemList...)
	lit.NKeys += len(native)
	brace := fl.literalBrace(lit)
	fl.add(brace+1, brace+1, text)
}

func (fl *fileLower) structDefaultFields(pos syntax.Pos, defaults []structDefault) ([]syntax.Expr, string) {
	var text strings.Builder
	var native []syntax.Expr
	for _, d := range defaults {
		value := fl.reparseDefault(d.text, pos)
		native = append(native, at(&syntax.KeyValueExpr{
			Key:   nameOf(pos, d.field),
			Value: value,
		}, pos))
		text.WriteString(d.field + ": " + d.text + ", ")
	}
	return native, text.String()
}

func (fl *fileLower) missingDefaults(lit *syntax.CompositeLit) []structDefault {
	names, texts := fl.structDefaultsFor(lit)
	if len(names) == 0 {
		return nil
	}
	present := make(map[string]bool)
	for _, spread := range fl.pkg.info.LogosSpreads[lit] {
		for _, name := range spread.Fields {
			present[name] = true
		}
	}
	for _, e := range lit.ElemList {
		if _, ok := e.(*syntax.SpreadExpr); ok {
			continue
		}
		kv, ok := syntax.Unparen(e).(*syntax.KeyValueExpr)
		if !ok {
			return nil
		}
		if key, ok := syntax.Unparen(kv.Key).(*syntax.Name); ok {
			present[key.Value] = true
		}
	}
	var out []structDefault
	for i, name := range names {
		if !present[name] {
			out = append(out, structDefault{field: name, text: texts[i]})
		}
	}
	return out
}

func (fl *fileLower) structDefaultsFor(lit *syntax.CompositeLit) ([]string, []string) {
	t := fl.pkg.info.TypeOf(lit)
	if t == nil {
		return nil, nil
	}
	u := types2.Unalias(t)
	st, _ := u.(*types2.Struct)
	if st == nil {
		if n, ok := u.(*types2.Named); ok {
			if s, ok := n.Underlying().(*types2.Struct); ok {
				st = s
			}
		}
	}
	if st == nil {
		return nil, nil
	}
	named, _ := u.(*types2.Named)
	var obj *types2.TypeName
	var targs string
	if named != nil {
		if o := named.Obj(); o != nil && o.Pkg() != nil && o.Parent() == o.Pkg().Scope() {
			obj = o
			if ta := named.TypeArgs(); ta != nil && ta.Len() > 0 {
				var parts []string
				for i := 0; i < ta.Len(); i++ {
					parts = append(parts, ta.At(i).String())
				}
				targs = "[" + strings.Join(parts, ", ") + "]"
			}
		}
	}
	qual := ""
	if obj != nil {
		qual = fl.importQualifier(obj.Pkg())
	}
	skips := fl.pkg.defaultFactorySkips()
	call := func(d types2.LogosFieldDefault) string {
		return qual + defaultFactoryName(d.Owner, d.Field) + targs + "()"
	}
	if defs := fl.pkg.info.LogosFieldDefaults[st]; len(defs) > 0 {
		names := make([]string, 0, len(defs))
		texts := make([]string, 0, len(defs))
		for _, d := range defs {
			names = append(names, d.Field)
			if d.Owner != "" && d.Field != "_" && !skips[d.Owner+"."+d.Field] {
				texts = append(texts, call(d))
			} else {
				texts = append(texts, renderExpr(d.Default))
			}
		}
		return names, texts
	}
	if obj == nil || obj.Pkg() == fl.pkg.tpkg {
		return nil, nil
	}
	imp, ok := facts.Lookup(obj.Pkg().Path())
	if !ok || imp == nil {
		return nil, nil
	}
	var names, texts []string
	for _, d := range imp.Defaults {
		if d.Type == obj.Name() {
			names = append(names, d.Field)
			texts = append(texts, qual+d.Text+targs+"()")
		}
	}
	return names, texts
}

func (fl *fileLower) literalBrace(lit *syntax.CompositeLit) int {
	if lit.Type == nil {
		return fl.off(lit.Pos())
	}
	brace := -1
	depth := 0
	typeBrace := false
	scanSegment(fl.src, fl.off(syntax.StartPos(lit.Type)), len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.STRUCT, token.INTERFACE:
			typeBrace = true
			return true
		case token.LBRACE:
			if !typeBrace && depth == 0 {
				brace = t.start
				return false
			}
			depth++
		case token.RBRACE:
			depth--
		}
		typeBrace = false
		return true
	})
	if brace < 0 {
		base.Fatalf("logos: composite literal at %s has no {", lit.Pos().String())
	}
	return brace
}

func (fl *fileLower) reparseDefault(text string, pos syntax.Pos) syntax.Expr {
	file, err := syntax.Parse(syntax.NewFileBase("logos-default"), strings.NewReader("package logosdefault\nvar _ = "+text), nil, nil, 0)
	if err != nil || file == nil || len(file.DeclList) != 1 {
		base.Fatalf("logos: cannot reparse default %q", text)
	}
	decl, ok := file.DeclList[0].(*syntax.VarDecl)
	if !ok || decl.Values == nil {
		base.Fatalf("logos: cannot reparse default %q", text)
	}
	value := decl.Values
	syntax.Inspect(value, func(n syntax.Node) bool {
		if n != nil {
			n.SetPos(pos)
		}
		return true
	})
	return value
}
