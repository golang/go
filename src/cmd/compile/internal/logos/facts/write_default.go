// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"cmd/compile/internal/types2"
)

func writeDefaults(pkg *types2.Package, info *types2.Info, f *Facts) {
	if pkg == nil || pkg.Scope() == nil || info == nil || len(info.LogosFieldDefaults) == 0 {
		return
	}
	add := func(d DefaultFact) {
		f.Defaults = appendUnique(f.Defaults, d)
	}
	plan := FactoryNamePlan(info, pkg)
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
			// The fact names the generated declaration-scope factory, not
			// the raw default text: consumers emit pkg.Symbol() so private
			// names and interface dynamic types stay bound where declared.
			// The owner, not the scope name, names the declaring type, so a
			// defined type over the same struct resolves the real factory.
			// The plan name is the one the package actually emitted, so a
			// collision-renamed factory never resolves to a user symbol.
			// The owner rides in front of the symbol so consumers can check
			// whether their literal's type arguments fit the factory.
			if t := plan[d.Owner+"."+d.Field]; t != "" {
				add(DefaultFact{Type: name, Field: d.Field, Text: d.Owner + "/" + t})
			}
		}
	}
}
