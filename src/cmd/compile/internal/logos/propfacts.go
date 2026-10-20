// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

// This file carries the M4-S2 property facts (SPEC Sec 2.3, T-203.2,
// Sec 6.1, T-Sec6). For each exported property interface it emits one
// prop record naming the interface and the accessor prefix; the property
// set itself travels in the lowered Logos_GetF/Logos_SetF methods of the
// export data, which .lgo importers read back through the same accessor
// shapes the checker uses. For each exported struct with a `: I`
// declaration it emits one conforms record per asserted interface, named
// as the interface's package path and type name, so an importer in
// another package can match its own spelling of the same interface and
// type-check the conformance without seeing the declaration.
//
// MergePropFacts is called once per file from lowerProperties (the M4-S2
// pass runs last in lowerFile); both emitters below derive from the
// package scope and check info, so repeat calls produce identical records
// and the merge reinstalls the same facts every time.

// propAccessorPrefix is the accessor prefix every generated method
// carries (types2 logosGetterName/logosSetterName).
const propAccessorPrefix = "Logos_"

// EmitPropFacts returns the prop and conforms records of pkg, or nil
// when the package declares no property interface and no conformance.
// Only exported names are recorded: importers cannot name anything else.
func EmitPropFacts(pkg *types2.Package, info *types2.Info) *facts.Facts {
	if pkg == nil || pkg.Scope() == nil || info == nil {
		return nil
	}
	var out facts.Facts
	for _, name := range pkg.Scope().Names() {
		obj, ok := pkg.Scope().Lookup(name).(*types2.TypeName)
		if !ok || obj == nil || !obj.Exported() || obj.IsAlias() {
			continue
		}
		if _, ok := types2.LogosPropsOf(obj.Type()); ok {
			out.Props = append(out.Props, facts.PropFact{Interface: name, Prefix: propAccessorPrefix})
		}
	}
	for _, decl := range info.LogosPropDecls {
		if decl == nil || decl.Def == nil || !decl.Def.Exported() {
			continue
		}
		for _, iface := range decl.Ifaces {
			path, iname, ok := propIfaceKey(iface)
			if !ok {
				continue
			}
			ifaceObj := ifaceScopeLookup(pkg, path, iname)
			if ifaceObj == nil || !ifaceObj.Exported() {
				continue
			}
			out.Conforms = append(out.Conforms, facts.ConformsFact{
				Type:      decl.Def.Name(),
				IfacePath: path,
				Iface:     iname,
			})
		}
	}
	if len(out.Props) == 0 && len(out.Conforms) == 0 {
		return nil
	}
	return &out
}

// propIfaceKey names iface as its origin package path and type name,
// following aliases to the named interface type. It reports false for
// types with no such name.
func propIfaceKey(iface types2.Type) (string, string, bool) {
	named, ok := types2.Unalias(iface).(*types2.Named)
	if !ok {
		return "", "", false
	}
	obj := named.Origin().Obj()
	if obj == nil || obj.Pkg() == nil || obj.Name() == "" {
		return "", "", false
	}
	return obj.Pkg().Path(), obj.Name(), true
}

// ifaceScopeLookup reports the type name iname of the package at path as
// seen from pkg: the package's own scope for a self reference, or the
// scope of the directly imported package otherwise. It returns nil when
// the name does not resolve, in which case the conformance is unusable
// to importers and the caller skips it.
func ifaceScopeLookup(pkg *types2.Package, path, iname string) *types2.TypeName {
	var scope *types2.Scope
	if path == pkg.Path() {
		scope = pkg.Scope()
	} else {
		for _, imp := range pkg.Imports() {
			if imp.Path() == path {
				scope = imp.Scope()
				break
			}
		}
	}
	if scope == nil {
		return nil
	}
	obj, ok := scope.Lookup(iname).(*types2.TypeName)
	if !ok {
		return nil
	}
	return obj
}

// MergePropFacts reinstalls the __.LOGOS facts for the package under
// lowering: fresh option records plus this package's prop records. Every
// input derives from the package scope and check info, so the result is
// identical on every call and installing it once per file converges.
// Rebuilding (rather than appending to the current facts) keeps records
// of a previously lowered package in the same process from leaking into
// this one. Const, fn, and comptime records installed by other producers
// pass through untouched.
func MergePropFacts(pkg *types2.Package, info *types2.Info) {
	merged := EmitOptionFacts(pkg, info)
	if merged == nil {
		merged = &facts.Facts{}
	}
	if cur := facts.Current(); cur != nil {
		merged.Consts = cur.Consts
		merged.Fns = cur.Fns
		merged.Comptimes = cur.Comptimes
		merged.Defaults = cur.Defaults // M6-S3: struct field defaults pass through.
	}
	if prop := EmitPropFacts(pkg, info); prop != nil {
		merged.Props = prop.Props
		merged.Conforms = prop.Conforms
	}
	facts.SetCurrent(merged)
}
