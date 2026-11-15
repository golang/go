// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"fmt"
	"strings"
)

const (
	sortOrderLeft  = "_logos_a"
	sortOrderRight = "_logos_b"
	sortOrderIndex = "_logos_i"
	sortOrderDiff  = "_logos_c"
)

type sortClass int

const (
	sortNone sortClass = iota
	sortOrdered
	sortBool
	sortComplex
	sortStruct
	sortArray
	sortInterface
	sortDynamic
)

func (fl *fileLower) rangedConstMap(x syntax.Expr) (*types2.Map, bool) {
	t := fl.pkg.info.TypeOf(x)
	if t == nil {
		return nil, false
	}
	if view := constAsView(t); view != nil {
		mt, ok := compositeUnderlying(view.Elem).(*types2.Map)
		return mt, ok
	}
	ref, ok := fl.compositeRefOf(x)
	if !ok || ref.method != nil {
		return nil, false
	}
	mt, ok := compositeUnderlying(t).(*types2.Map)
	return mt, ok
}

func (fl *fileLower) sortedRangeEntry(key types2.Type, pos syntax.Pos) (entry, compare string) {
	site := &sortedRangeSite{pos: pos}
	switch fl.sortClassOf(key, site) {
	case sortInterface, sortDynamic:
		return "SortedAllDynamic", ""
	case sortNone:
		return "SortedAll", "nil"
	}
	return "SortedAll", fl.sortOrderFunc(key, site)
}

func sortedRangeCompare(text string, pos syntax.Pos) syntax.Expr {
	e, ok := parseConstMapEntryExpr(text, pos)
	if !ok {
		base.Fatalf("logos: cannot parse sorted range order %q at %s", text, pos.String())
	}
	return e
}

func (fl *fileLower) sortClassOf(t types2.Type, site *sortedRangeSite) sortClass {
	if fl.asOption(t) != nil {
		return sortDynamic
	}
	switch u := compositeUnderlying(t).(type) {
	case *types2.Basic:
		return basicSortClass(u)
	case *types2.Struct:
		return fl.structSortClass(t, u, site)
	case *types2.Array:
		return fl.arraySortClass(t, u, site)
	case *types2.Pointer, *types2.Chan:
		return sortNone
	case *types2.Interface:
		return sortInterface
	}
	return sortDynamic
}

func basicSortClass(b *types2.Basic) sortClass {
	switch info := b.Info(); {
	case info&types2.IsOrdered != 0:
		return sortOrdered
	case info&types2.IsBoolean != 0:
		return sortBool
	case info&types2.IsComplex != 0:
		return sortComplex
	}
	return sortNone
}

func (fl *fileLower) memberSortClass(t types2.Type, site *sortedRangeSite) sortClass {
	if class := fl.sortClassOf(t, site); class != sortInterface {
		return class
	}
	return sortNone
}

func (fl *fileLower) structSortClass(t types2.Type, st *types2.Struct, site *sortedRangeSite) sortClass {
	result := sortNone
	for i := range st.NumFields() {
		f := st.Field(i)
		if f.Name() == "_" {
			continue
		}
		switch class := fl.memberSortClass(f.Type(), site); {
		case class == sortDynamic:
			return sortDynamic
		case class == sortNone:
			continue
		case !fl.fieldNameable(f):
			return sortDynamic
		}
		result = sortStruct
	}
	return fl.spellableSortClass(t, result, site)
}

func (fl *fileLower) fieldNameable(f *types2.Var) bool {
	return f.Exported() || f.Pkg() == fl.pkg.tpkg
}

func (fl *fileLower) arraySortClass(t types2.Type, at *types2.Array, site *sortedRangeSite) sortClass {
	switch class := fl.memberSortClass(at.Elem(), site); class {
	case sortNone, sortDynamic:
		return class
	}
	return fl.spellableSortClass(t, sortArray, site)
}

func (fl *fileLower) spellableSortClass(t types2.Type, class sortClass, site *sortedRangeSite) sortClass {
	if class != sortNone {
		if _, ok := fl.spellSortedType(t, site); !ok {
			return sortDynamic
		}
	}
	return class
}

func (fl *fileLower) sortOrderFunc(t types2.Type, site *sortedRangeSite) string {
	switch fl.sortClassOf(t, site) {
	case sortOrdered:
		return rtAlias + ".CompareOrdered"
	case sortBool:
		return rtAlias + ".CompareBool"
	case sortComplex:
		return rtAlias + ".CompareComplex"
	case sortStruct:
		return fl.structSortFunc(t, site)
	case sortArray:
		return fl.arraySortFunc(t, site)
	}
	return ""
}

func (fl *fileLower) structSortFunc(t types2.Type, site *sortedRangeSite) string {
	st := compositeUnderlying(t).(*types2.Struct)
	var b strings.Builder
	fmt.Fprintf(&b, "func(%s, %s %s) int { ", sortOrderLeft, sortOrderRight, fl.mustSpellSorted(t, site))
	for i := range st.NumFields() {
		f := st.Field(i)
		if f.Name() == "_" {
			continue
		}
		if fn := fl.sortOrderFunc(f.Type(), site); fn != "" {
			b.WriteString(sortOrderStep(fn, "."+f.Name()))
		}
	}
	b.WriteString("return 0 }")
	return b.String()
}

func (fl *fileLower) arraySortFunc(t types2.Type, site *sortedRangeSite) string {
	at := compositeUnderlying(t).(*types2.Array)
	step := sortOrderStep(fl.sortOrderFunc(at.Elem(), site), "["+sortOrderIndex+"]")
	return fmt.Sprintf("func(%s, %s %s) int { for %[4]s := 0; %[4]s < len(%[1]s); %[4]s++ { %[5]s}; return 0 }",
		sortOrderLeft, sortOrderRight, fl.mustSpellSorted(t, site), sortOrderIndex, step)
}

func sortOrderStep(fn, selector string) string {
	return fmt.Sprintf("if %[1]s := %[2]s(%[3]s%[5]s, %[4]s%[5]s); %[1]s != 0 { return %[1]s }; ",
		sortOrderDiff, fn, sortOrderLeft, sortOrderRight, selector)
}

type sortedRangeSite struct {
	pos      syntax.Pos
	scope    *types2.Scope
	scopeSet bool
}

func (fl *fileLower) spellSortedType(t types2.Type, site *sortedRangeSite) (string, bool) {
	s, ok := fl.spellLowered(t)
	if !ok {
		return "", false
	}
	if !fl.spellScopeSafe(t, site) {
		return "", false
	}
	return s, true
}

func (fl *fileLower) mustSpellSorted(t types2.Type, site *sortedRangeSite) string {
	s, ok := fl.spellSortedType(t, site)
	if !ok {
		base.Fatalf("logos: cannot spell const map type %s at %s", t, site.pos.String())
	}
	return s
}

func (fl *fileLower) spellScopeSafe(t types2.Type, site *sortedRangeSite) bool {
	if !site.scopeSet {
		site.scope, site.scopeSet = fl.sortedScopeAt(site.pos), true
	}
	if site.scope == nil || fl.pkg.info == nil {
		return false
	}
	c := &sortedSpellCheck{
		fl:          fl,
		site:        site,
		scope:       site.scope,
		seenLowered: make(map[types2.Type]bool),
		seenPrinted: make(map[types2.Type]bool),
	}
	return c.lowered(t) && c.basicOK("int")
}

type sortedSpellCheck struct {
	fl          *fileLower
	site        *sortedRangeSite
	scope       *types2.Scope
	seenLowered map[types2.Type]bool
	seenPrinted map[types2.Type]bool
}

func (c *sortedSpellCheck) lowered(t types2.Type) bool {
	if t == nil || c.seenLowered[t] {
		return true
	}
	c.seenLowered[t] = true
	if c.fl.asOption(t) != nil {
		return false
	}
	switch u := types2.Unalias(t).(type) {
	case *types2.ConstView:
		return c.lowered(u.Elem)
	case *types2.ReadonlyDescendants:
		return c.lowered(u.Elem)
	case *types2.Named, *types2.Alias, *types2.Basic, *types2.TypeParam:
		return c.printed(t)
	case *types2.Pointer:
		return c.lowered(u.Elem())
	case *types2.Slice:
		return c.lowered(u.Elem())
	case *types2.Array:
		return c.lowered(u.Elem())
	case *types2.Map:
		return c.lowered(u.Key()) && c.lowered(u.Elem())
	case *types2.Chan:
		return c.lowered(u.Elem())
	case *types2.Signature:
		return c.loweredTuple(u.Params()) && c.loweredTuple(u.Results())
	case *types2.Struct:
		for i := range u.NumFields() {
			if !c.lowered(u.Field(i).Type()) {
				return false
			}
		}
		return true
	case *types2.Interface:
		for i := range u.NumMethods() {
			if !c.lowered(u.Method(i).Type()) {
				return false
			}
		}
		for i := range u.NumEmbeddeds() {
			if !c.lowered(u.EmbeddedType(i)) {
				return false
			}
		}
		return true
	case *types2.LogosTuple:
		if st, ok := u.Underlying().(*types2.Struct); ok {
			for i := range st.NumFields() {
				if !c.lowered(st.Field(i).Type()) {
					return false
				}
			}
			return true
		}
	}
	return false
}

func (c *sortedSpellCheck) loweredTuple(t *types2.Tuple) bool {
	if t == nil {
		return true
	}
	for i := range t.Len() {
		if !c.lowered(t.At(i).Type()) {
			return false
		}
	}
	return true
}

func (c *sortedSpellCheck) printed(t types2.Type) bool {
	if t == nil || c.seenPrinted[t] {
		return true
	}
	c.seenPrinted[t] = true
	switch t := t.(type) {
	case *types2.Alias:
		return c.nameOK(t.Obj()) && c.printedAlias(t)
	case *types2.Named:
		return c.nameOK(t.Obj()) && c.printedNamed(t)
	case *types2.TypeParam:
		return c.nameOK(t.Obj())
	case *types2.Basic:
		if t.Kind() == types2.UnsafePointer {
			return c.qualifierOK("unsafe")
		}
		return c.basicOK(t.Name())
	case *types2.Pointer:
		return c.printed(t.Elem())
	case *types2.Slice:
		return c.printed(t.Elem())
	case *types2.Array:
		return c.printed(t.Elem())
	case *types2.Map:
		return c.printed(t.Key()) && c.printed(t.Elem())
	case *types2.Chan:
		return c.printed(t.Elem())
	case *types2.Signature:
		return c.printedParams(t.TypeParams()) && c.printedTuple(t.Params()) && c.printedTuple(t.Results())
	case *types2.Tuple:
		return c.printedTuple(t)
	case *types2.Struct:
		for i := range t.NumFields() {
			if !c.printed(t.Field(i).Type()) {
				return false
			}
		}
		return true
	case *types2.Interface:
		for i := range t.NumMethods() {
			if !c.printed(t.Method(i).Type()) {
				return false
			}
		}
		for i := range t.NumEmbeddeds() {
			if !c.printed(t.EmbeddedType(i)) {
				return false
			}
		}
		return true
	case *types2.ConstView:
		return c.printed(t.Elem)
	case *types2.ReadonlyDescendants:
		return c.printed(t.Elem)
	case *types2.Option:
		return c.printed(t.Elem)
	case *types2.LogosTuple:
		for i := range t.Len() {
			if !c.printed(t.Elem(i)) {
				return false
			}
		}
		return true
	}
	return false
}

func (c *sortedSpellCheck) printedNamed(t *types2.Named) bool {
	if args := t.TypeArgs(); args != nil {
		return c.printedArgs(args)
	}
	return c.printedParams(t.TypeParams())
}

func (c *sortedSpellCheck) printedAlias(t *types2.Alias) bool {
	if args := t.TypeArgs(); args != nil && args.Len() != 0 {
		return c.printedArgs(args)
	}
	return c.printedParams(t.TypeParams())
}

func (c *sortedSpellCheck) printedArgs(t *types2.TypeList) bool {
	if t == nil {
		return true
	}
	for i := range t.Len() {
		if !c.printed(t.At(i)) {
			return false
		}
	}
	return true
}

func (c *sortedSpellCheck) printedParams(t *types2.TypeParamList) bool {
	if t == nil {
		return true
	}
	for i := range t.Len() {
		tp := t.At(i)
		if tp == nil {
			return false
		}
		if !c.printed(tp) {
			return false
		}
		if b := tp.Constraint(); b != nil && !c.printed(b) {
			return false
		}
	}
	return true
}

func (c *sortedSpellCheck) printedTuple(t *types2.Tuple) bool {
	if t == nil {
		return true
	}
	for i := range t.Len() {
		if !c.printed(t.At(i).Type()) {
			return false
		}
	}
	return true
}

func (c *sortedSpellCheck) nameOK(obj *types2.TypeName) bool {
	if obj == nil {
		return false
	}
	if p := obj.Pkg(); p != nil && p != c.fl.pkg.tpkg {
		return c.qualifierOK(p.Path())
	}
	return c.scope.LookupParent(obj.Name(), c.site.pos) == obj
}

func (c *sortedSpellCheck) basicOK(name string) bool {
	tn, ok := c.scope.LookupParent(name, c.site.pos).(*types2.TypeName)
	return ok && tn.Pkg() == nil
}

func (c *sortedSpellCheck) qualifierOK(path string) bool {
	if c.fl.genImps == nil {
		return false
	}
	q := c.fl.genImps.qual[path]
	if q == "" {
		return false
	}
	got := c.scope.LookupParent(q, c.site.pos)
	if got == nil {
		return true
	}
	pn, ok := got.(*types2.PkgName)
	if !ok || pn.Imported() == nil {
		return false
	}
	return pn.Imported().Path() == path
}

func (fl *fileLower) sortedScopeAt(pos syntax.Pos) *types2.Scope {
	if fl.pkg.info == nil {
		return nil
	}
	fileScope := fl.pkg.info.Scopes[fl.file]
	if fileScope == nil {
		return nil
	}
	return fileScope.Innermost(pos)
}
