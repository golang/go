// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"bytes"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func writeDefaults(pkg *types2.Package, info *types2.Info, f *Facts) {
	if pkg == nil || pkg.Scope() == nil || info == nil || len(info.LogosFieldDefaults) == 0 {
		return
	}
	add := func(d DefaultFact) {
		f.Defaults = appendUnique(f.Defaults, d)
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
			var buf bytes.Buffer
			_, _ = syntax.Fprint(&buf, d.Default, syntax.LineForm)
			add(DefaultFact{Type: name, Field: d.Field, Text: buf.String()})
		}
	}
}
