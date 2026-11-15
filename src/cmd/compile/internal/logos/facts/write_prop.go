// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import "cmd/compile/internal/types2"

const propAccessorPrefix = "Logos_"

func writeProps(pkg *types2.Package, info *types2.Info, f *Facts) {
	if pkg == nil || pkg.Scope() == nil || info == nil {
		return
	}
	for _, name := range pkg.Scope().Names() {
		obj, ok := pkg.Scope().Lookup(name).(*types2.TypeName)
		if !ok || obj == nil || !obj.Exported() || obj.IsAlias() {
			continue
		}
		if _, ok := types2.LogosPropsOf(obj.Type()); ok {
			f.Props = append(f.Props, PropFact{Interface: name, Prefix: propAccessorPrefix})
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
			f.Conforms = appendUnique(f.Conforms, ConformsFact{
				Type:      decl.Def.Name(),
				IfacePath: path,
				Iface:     iname,
			})
		}
	}
}

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
