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

// M6-S3 lowering of struct field defaults (SPEC T-104.5): the `= const`
// of a field declaration is stripped, and every composite literal of a
// defaulted struct gains its missing fields. Same-package defaults come
// from check info; cross-package defaults come from the facts sidecar.
// `var x T` and `new(T)` stay zero by construction: only literals grow.

// lowerStructDefaults strips field defaults and fills composite
// literals. It runs before the `&expr` pass, whose operand splices
// absorb these inserts.
func (fl *fileLower) lowerStructDefaults() {
	stripped := make(map[*syntax.FieldDefault]bool)
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if st, ok := n.(*syntax.StructType); ok {
			fl.stripFieldDefaults(st, stripped)
		}
		return true
	})
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if lit, ok := n.(*syntax.CompositeLit); ok {
			fl.fillStructDefaults(lit)
		}
		return true
	})
}

// stripFieldDefaults replaces each defaulted field type with its
// declared type and deletes the `= const` source. Shared declarations
// delete once per wrapper.
func (fl *fileLower) stripFieldDefaults(st *syntax.StructType, stripped map[*syntax.FieldDefault]bool) {
	for _, f := range st.FieldList {
		fd, ok := f.Type.(*syntax.FieldDefault)
		if !ok {
			continue
		}
		f.Type = fd.Type
		if stripped[fd] {
			continue
		}
		stripped[fd] = true
		fl.add(fl.off(syntax.EndPos(fd.Type)), fl.off(syntax.EndPos(fd.Default)), "")
	}
}

// structDefault is one missing field and its default source text.
type structDefault struct {
	field string
	text  string
}

// fillStructDefaults prepends the missing defaulted fields of one
// composite literal, in declaration order, to the native tree and the
// eject text.
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

// missingDefaults returns the defaulted fields absent from lit, in
// declaration order. Positional literals and literals of types without
// defaults return none.
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

// structDefaultsFor returns the defaulted field names of t with their
// default source texts: from check info for local types, from facts for
// imported loGo/s types. A Go importer never runs this pass, which is
// the documented hole.
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

// literalBrace returns the offset of the `{` opening lit: the literal
// position for elided types, else the first brace past the type end.
func (fl *fileLower) literalBrace(lit *syntax.CompositeLit) int {
	if lit.Type == nil {
		return fl.off(lit.Pos())
	}
	brace := -1
	scanSegment(fl.src, fl.off(syntax.EndPos(lit.Type)), len(fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind == token.LBRACE {
			brace = t.start
			return false
		}
		return true
	})
	if brace < 0 {
		base.Fatalf("logos: composite literal at %s has no {", lit.Pos().String())
	}
	return brace
}

// reparseDefault parses a default expression text into a fresh node
// positioned at pos, mirroring reparseType.
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
