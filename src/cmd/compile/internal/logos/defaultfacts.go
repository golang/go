// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

// M6-S3 struct field default facts (SPEC T-104.5, T-Sec6): one default
// record per defaulted field of each exported struct type, carrying the
// default expression source text for .lgo importers. Lower wires the
// call beside EmitOptionFacts; the records pass through MergePropFacts.

// EmitDefaultFacts adds the struct field defaults of pkg to f,
// allocating it when nil, and returns f. Only exported non-alias named
// types are recorded: importers cannot name anything else. Same-package
// lowering reads check info directly; facts serve other packages.
func EmitDefaultFacts(pkg *types2.Package, info *types2.Info, f *facts.Facts) *facts.Facts {
	if pkg == nil || pkg.Scope() == nil || info == nil || len(info.LogosFieldDefaults) == 0 {
		return f
	}
	known := make(map[facts.DefaultFact]bool)
	if f != nil {
		for _, d := range f.Defaults {
			known[d] = true
		}
	}
	add := func(d facts.DefaultFact) {
		if known[d] {
			return
		}
		known[d] = true
		if f == nil {
			f = &facts.Facts{}
		}
		f.Defaults = append(f.Defaults, d)
	}
	for _, name := range pkg.Scope().Names() {
		obj, ok := pkg.Scope().Lookup(name).(*types2.TypeName)
		if !ok || obj == nil || !obj.Exported() || obj.IsAlias() {
			continue
		}
		named, ok := obj.Type().(*types2.Named)
		if !ok {
			continue
		}
		st, ok := named.Underlying().(*types2.Struct)
		if !ok {
			continue
		}
		for _, d := range info.LogosFieldDefaults[st] {
			add(facts.DefaultFact{Type: name, Field: d.Field, Text: renderExpr(d.Default)})
		}
	}
	return f
}
