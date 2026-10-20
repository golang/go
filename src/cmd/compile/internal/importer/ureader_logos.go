// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

// LogosOverlayEnabled gates the option overlay for direct importer
// users. Noder applies its own overlay after ReadPackage and leaves
// this false, so this hook stays a no-op there; direct users set it
// before ReadPackage to request the .lgo view.
var LogosOverlayEnabled bool

// logosOverlayHook runs after ReadPackage. The .lgo overlay lives in
// noder, which owns the import cache and the facts lookup; this hook
// only consumes the enable flag so direct importer users keep a hook
// point without pulling new dependencies here.
func logosOverlayHook(pkg *types2.Package) *types2.Package {
	LogosOverlayEnabled = false
	logosRegisterPropFacts(pkg)
	logosRegisterDefaultFacts(pkg)
	return pkg
}

// logosRegisterDefaultFacts records the struct field defaults of an
// imported loGo/s package for .lgo importers (M6-S3, SPEC T-104.5).
// Each default record becomes one entry in the checker's imported
// defaults table, keyed by struct package path and name. Packages
// without facts leave the table unchanged.
func logosRegisterDefaultFacts(pkg *types2.Package) {
	if pkg == nil {
		return
	}
	f, ok := facts.Lookup(pkg.Path())
	if !ok || f == nil || len(f.Defaults) == 0 {
		return
	}
	byType := make(map[string][]types2.LogosImportedDefault)
	for _, d := range f.Defaults {
		if d.Type == "" || d.Field == "" || d.Text == "" {
			continue
		}
		byType[d.Type] = append(byType[d.Type], types2.LogosImportedDefault{Field: d.Field, Text: d.Text})
	}
	for typeName, defs := range byType {
		types2.LogosNoteImportedDefaults(pkg.Path(), typeName, defs)
	}
}

// logosRegisterPropFacts records the `: I` conformances of an imported
// loGo/s package for .lgo importers (M4-S2, SPEC T-203.2). The facts
// registry already holds the package's __.LOGOS member under its import
// path by the time ReadPackage returns; each conforms record becomes one
// entry in the checker's imported-conformance table, keyed by struct
// package path and name with the asserted interface as package path and
// name. String keys keep the entries valid across package clones (the
// noder option overlay rebuilds objects after this hook runs) and let a
// checker in another package match its own spelling of the interface.
// Packages without facts (plain Go, or types2 tests without archives)
// leave the table unchanged.
func logosRegisterPropFacts(pkg *types2.Package) {
	if pkg == nil {
		return
	}
	f, ok := facts.Lookup(pkg.Path())
	if !ok || f == nil || len(f.Conforms) == 0 {
		return
	}
	byType := make(map[string][]types2.LogosConformsIface)
	for _, c := range f.Conforms {
		if c.Type == "" || c.IfacePath == "" || c.Iface == "" {
			continue
		}
		byType[c.Type] = append(byType[c.Type], types2.LogosConformsIface{PkgPath: c.IfacePath, Name: c.Iface})
	}
	for typeName, ifaces := range byType {
		types2.LogosNoteImportedConforms(pkg.Path(), typeName, ifaces)
	}
}
