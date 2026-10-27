// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"maps"
	"slices"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/goobj"
	"cmd/internal/obj"
)

// preassignSymIdxs assigns a SymIdx to the symbols that will appear in export
// data. This allows the export data to be emitted before the backend runs and NumberSyms is called.
func preassignSymIdxs(symABIs *ssagen.SymABIs) {
	// The visit is used to populate usedTypes, which is then used to filter additional
	// symbols from types.LocalPkg.Syms that are not top level, but that need to be
	// exported. For example, this includes a used type (meaning that it won't be removed by deadlocals)
	// that's defined locally inside a top level function.
	usedTypes := make(map[*types.Type]bool)
	var visit func(*types.Type)
	visit = func(t *types.Type) {
		if t == nil || usedTypes[t] {
			return
		}
		usedTypes[t] = true
		switch t.Kind() {
		case types.TPTR, types.TSLICE, types.TARRAY, types.TCHAN:
			visit(t.Elem())
		case types.TMAP:
			visit(t.Key())
			visit(t.Elem())
		case types.TSTRUCT:
			for _, f := range t.Fields() {
				visit(f.Type)
			}
		case types.TFUNC:
			for _, f := range t.RecvParamsResults() {
				visit(f.Type)
			}
		case types.TINTER:
			for _, m := range t.AllMethods() {
				visit(m.Type)
			}
		}
		if t.Sym() != nil && !t.IsInterface() {
			for _, m := range t.Methods() {
				visit(m.Type)
			}
		}
	}

	for _, n := range typecheck.Target.Externs {
		if n.Op() == ir.ONAME && nameShouldBeIndexed(n) {
			base.Ctxt.NumberSymEarly(n.Linksym())
		}
		if n.Op() == ir.OTYPE {
			visit(n.Type())
		}
	}

	for _, fn := range typecheck.Target.Funcs {
		if funcShouldBeIndexed(fn, symABIs) {
			base.Ctxt.NumberSymEarly(fn.Nname.Linksym())
		}
		for _, dcl := range fn.Dcl {
			if dcl.Op() == ir.ONAME && dcl.Used() {
				visit(dcl.Type())
			}
		}
		for _, cv := range fn.ClosureVars {
			if cv.Used() {
				visit(cv.Type())
			}
		}
	}

	reflectdata.ForEachRuntimeType(visit)

	// Number any other types from types.LocalPkg.Syms that are reachable from the
	// top level declarations or from types we know we will generate runtime data for.
	for _, name := range slices.Sorted(maps.Keys(types.LocalPkg.Syms)) {
		if n, ok := types.LocalPkg.Syms[name].Def.(*ir.Name); ok && usedTypes[n.Type()] {
			if n.Op() == ir.OTYPE && !n.Alias() && typeShouldBeIndexed(n.Type()) {
				base.Ctxt.NumberSymEarly(reflectdata.TypeLinksym(n.Type()))
				base.Ctxt.NumberSymEarly(reflectdata.TypeLinksym(n.Type().PtrTo()))
			}
		}
	}
}

// lsymShouldBeIndexed checks the common conditions where names, types, and funcs should be preassigned an index for export data:
//   - We should not have already preassigned an index for the symbol (nothing to do)
//   - It should not be content addressable (we don't necessarily know their content yet)
//   - It should belong to the current package.
func lsymShouldBeIndexed(lsym *obj.LSym) bool {
	return lsym.PkgIdx == goobj.PkgIdxInvalid && !lsym.Indexed() && !lsym.ContentAddressable() && !base.Ctxt.IsNonPkgSym(lsym)
}

// nameShouldBeIndexed checks the conditions where names should be preassigned an index for export data:
//   - They are global variable symbols (PEXTERN)
//   - They belong to the current package (otherwise they should be numbered when that package is compiled)
//   - They don't have a linkname (see IsNonPkgSym).
//
// TODO(matloob): Can we drop the linkname condition? Is it already checked by IsNonPkgSym?
func nameShouldBeIndexed(name *ir.Name) bool {
	return lsymShouldBeIndexed(name.Linksym()) &&
		name.Class == ir.PEXTERN && name.Sym().Pkg == types.LocalPkg && name.Sym().Linkname == ""
}

// typeShouldBeIndexed checks the conditions where types should be preassigned an index for export data:
//   - They belong to the current package
//   - They are not DUPOK.
func typeShouldBeIndexed(typ *types.Type) bool {
	return typ.Sym() != nil && typ.Sym().Pkg == types.LocalPkg && !reflectdata.TypeCanBeDupok(typ) &&
		lsymShouldBeIndexed(types.TypeSym(typ).Linksym())
}

// TODO(matloob): better document when functions should be preassigned an index for export data.
func funcShouldBeIndexed(fn *ir.Func, symABIs *ssagen.SymABIs) bool {
	name := fn.Nname
	if fn.ABI != obj.ABIInternal || fn.Dupok() || fn.IsClosure() || ir.IsBlank(name) || name.Sym().Linkname != "" || !lsymShouldBeIndexed(name.Linksym()) {
		return false
	}
	return len(fn.Body) != 0 || fn.WasmImport != nil || needsIntrinsicBody(fn, symABIs)
}
