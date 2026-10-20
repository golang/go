// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/logos"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

// applyLogosOverlayForLgoImporter rebuilds an imported loGo/s package
// with Option types. Callers invoke it only for .lgo importers; Go
// importers keep the lowered types without calling here. It returns the
// package to use and updates the import cache so later lookups in the
// same compilation see the same view.
func applyLogosOverlayForLgoImporter(path string, pkg *types2.Package, packages map[string]*types2.Package) *types2.Package {
	if pkg == nil {
		return pkg
	}
	f, ok := facts.Lookup(path)
	if !ok || f == nil || len(f.Opts) == 0 {
		return pkg
	}
	cloned := logos.CloneWithOverlay(pkg, f.Opts)
	if cloned != pkg && packages != nil {
		packages[path] = cloned
	}
	return cloned
}
