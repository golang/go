// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/logos"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

func applyLogosReadonlyForGoImporter(path string, pkg *types2.Package, packages map[string]*types2.Package, f *facts.Facts) (*types2.Package, error) {
	if pkg == nil || f == nil {
		return pkg, nil
	}
	if err := logos.ApplyFuncFacts(pkg, f.Fns); err != nil {
		return nil, err
	}
	if err := logos.ApplyReadonlyFactsInPlace(pkg, f.Readonlys); err != nil {
		return nil, err
	}
	return pkg, nil
}

func applyLogosOverlayForLgoImporter(path string, pkg *types2.Package, packages map[string]*types2.Package, f *facts.Facts) (*types2.Package, error) {
	if pkg == nil || f == nil {
		return pkg, nil
	}
	if f.GoPackage || !logosHasStructuralFacts(f) {
		if err := logos.ApplyFuncFacts(pkg, f.Fns); err != nil {
			return nil, err
		}
		if err := logos.ApplyReadonlyFactsInPlace(pkg, f.Readonlys); err != nil {
			return nil, err
		}
		return pkg, nil
	}
	cloned, err := logos.CloneWithOverlaysAndReadonly(pkg, f.Opts, f.Consts, f.Fns, f.Readonlys, f.ConstMaps)
	if err != nil {
		return nil, err
	}
	if cloned != pkg && packages != nil {
		packages[path] = cloned
	}
	return cloned, nil
}
