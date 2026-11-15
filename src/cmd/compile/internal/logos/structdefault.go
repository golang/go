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
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.StructType:
			fl.stripFieldDefaults(n, stripped)
		case *syntax.CompositeLit:
			fl.fillStructDefaults(n)
		}
		return true
	})
}

func (fl *fileLower) stripFieldDefaults(st *syntax.StructType, stripped map[*syntax.FieldDefault]bool) {
	for i, f := range st.FieldList {
		fd, ok := f.Type.(*syntax.FieldDefault)
		if !ok {
			continue
		}
		f.Type = fd.Type
		if stripped[fd] {
			continue
		}
		stripped[fd] = true
		typeEnd := fl.exprEndNested(fl.off(syntax.StartPos(fd.Type)))
		fl.add(typeEnd, fl.fieldDefaultEnd(st, i, fd, typeEnd), "")
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
	missing := fl.missingDefaults(lit)
	if len(missing) == 0 {
		return
	}
	var text strings.Builder
	var native []syntax.Expr
	for _, d := range missing {
		value := fl.reparseDefault(d.text, lit.Pos())
		native = append(native, at(&syntax.KeyValueExpr{
			Key:   nameOf(lit.Pos(), d.field),
			Value: value,
		}, lit.Pos()))
		text.WriteString(d.field + ": " + d.text + ", ")
	}
	lit.ElemList = append(native, lit.ElemList...)
	lit.NKeys += len(native)
	brace := fl.literalBrace(lit)
	fl.add(brace+1, brace+1, text.String())
}

func (fl *fileLower) missingDefaults(lit *syntax.CompositeLit) []structDefault {
	t := fl.pkg.info.TypeOf(lit)
	if t == nil {
		return nil
	}
	names, texts := fl.structDefaultsFor(t)
	if len(names) == 0 {
		return nil
	}
	present := make(map[string]bool)
	for _, e := range lit.ElemList {
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

func (fl *fileLower) structDefaultsFor(t types2.Type) ([]string, []string) {
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
	if defs := fl.pkg.info.LogosFieldDefaults[st]; len(defs) > 0 {
		names := make([]string, 0, len(defs))
		texts := make([]string, 0, len(defs))
		for _, d := range defs {
			names = append(names, d.Field)
			texts = append(texts, renderExpr(d.Default))
		}
		return names, texts
	}
	named, ok := u.(*types2.Named)
	if !ok {
		return nil, nil
	}
	obj := named.Obj()
	if obj == nil || obj.Pkg() == nil || obj.Pkg() == fl.pkg.tpkg {
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
			texts = append(texts, d.Text)
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
