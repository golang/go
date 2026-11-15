// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func Write(pkg *types2.Package, info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte) *Facts {
	f := &Facts{}
	if pkg == nil || pkg.Scope() == nil {
		return f
	}
	writeOpts(pkg, f)
	writeConsts(pkg, info, files, f)
	writeDefaults(pkg, info, f)
	writeConstFuncs(pkg, info, files, src, f)
	writeProps(pkg, info, f)
	writeJSON(pkg, info, f)
	return f
}

func WriteChecked(pkg *types2.Package, info *types2.Info, files []*syntax.File, f *Facts) *Facts {
	if pkg == nil || pkg.Scope() == nil {
		return f
	}
	out := f
	if out == nil {
		out = &Facts{}
	}
	writeAsserts(pkg, out)
	writeLocks(pkg, info, files, out)
	writeNilErrors(pkg, info, files, out)
	if f == nil && out.empty() {
		return nil
	}
	return out
}

func findOrCreateFnFact(f *Facts, object string) *FnFact {
	for i := range f.Fns {
		if f.Fns[i].Object == object {
			return &f.Fns[i]
		}
	}
	f.Fns = append(f.Fns, FnFact{Object: object})
	return &f.Fns[len(f.Fns)-1]
}

func unaliasAs[T types2.Type](t types2.Type) (T, bool) {
	var zero T
	if v, ok := types2.Unalias(t).(T); ok {
		return v, true
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok && n.Underlying() != nil {
		if v, ok := n.Underlying().(T); ok {
			return v, true
		}
	}
	return zero, false
}

func eachExportedSig(pkg *types2.Package, yield func(object string, sig *types2.Signature)) {
	scope := pkg.Scope()
	if scope == nil {
		return
	}
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if obj == nil || !obj.Exported() {
			continue
		}
		switch obj := obj.(type) {
		case *types2.Func:
			sig, ok := obj.Type().(*types2.Signature)
			if !ok || sig.Recv() != nil {
				continue
			}
			yield(name, sig)
		case *types2.TypeName:
			if obj.IsAlias() {
				continue
			}
			named, ok := obj.Type().(*types2.Named)
			if !ok {
				continue
			}
			for i := range named.NumMethods() {
				m := named.Method(i)
				if m == nil || !m.Exported() {
					continue
				}
				sig, ok := m.Type().(*types2.Signature)
				if !ok {
					continue
				}
				if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
					continue
				}
				yield(MethodObject(name, sig.Recv(), m.Name()), sig)
			}
		}
	}
}

func eachExportedNamed(pkg *types2.Package, yield func(name string, named *types2.Named)) {
	scope := pkg.Scope()
	if scope == nil {
		return
	}
	for _, name := range scope.Names() {
		obj, ok := scope.Lookup(name).(*types2.TypeName)
		if !ok || obj == nil || !obj.Exported() || obj.IsAlias() {
			continue
		}
		named, ok := obj.Type().(*types2.Named)
		if !ok {
			continue
		}
		yield(name, named)
	}
}

func appendUnique[E comparable](s []E, e E) []E {
	for _, have := range s {
		if have == e {
			return s
		}
	}
	return append(s, e)
}

func joinPath(path []string) string {
	return strings.Join(path, ".")
}
