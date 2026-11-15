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
	switch fl.sortClassOf(key) {
	case sortInterface, sortDynamic:
		return "SortedAllDynamic", ""
	case sortNone:
		return "SortedAll", "nil"
	}
	return "SortedAll", fl.sortOrderFunc(key, pos)
}

func sortedRangeCompare(text string, pos syntax.Pos) syntax.Expr {
	e, ok := parseConstMapEntryExpr(text, pos)
	if !ok {
		base.Fatalf("logos: cannot parse sorted range order %q at %s", text, pos.String())
	}
	return e
}

func (fl *fileLower) sortClassOf(t types2.Type) sortClass {
	if fl.asOption(t) != nil {
		return sortDynamic
	}
	switch u := compositeUnderlying(t).(type) {
	case *types2.Basic:
		return basicSortClass(u)
	case *types2.Struct:
		return fl.structSortClass(t, u)
	case *types2.Array:
		return fl.arraySortClass(t, u)
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

func (fl *fileLower) memberSortClass(t types2.Type) sortClass {
	if class := fl.sortClassOf(t); class != sortInterface {
		return class
	}
	return sortNone
}

func (fl *fileLower) structSortClass(t types2.Type, st *types2.Struct) sortClass {
	result := sortNone
	for i := range st.NumFields() {
		f := st.Field(i)
		if f.Name() == "_" {
			continue
		}
		switch class := fl.memberSortClass(f.Type()); {
		case class == sortDynamic:
			return sortDynamic
		case class == sortNone:
			continue
		case !fl.fieldNameable(f):
			return sortDynamic
		}
		result = sortStruct
	}
	return fl.spellableSortClass(t, result)
}

func (fl *fileLower) fieldNameable(f *types2.Var) bool {
	return f.Exported() || f.Pkg() == fl.pkg.tpkg
}

func (fl *fileLower) arraySortClass(t types2.Type, at *types2.Array) sortClass {
	switch class := fl.memberSortClass(at.Elem()); class {
	case sortNone, sortDynamic:
		return class
	}
	return fl.spellableSortClass(t, sortArray)
}

func (fl *fileLower) spellableSortClass(t types2.Type, class sortClass) sortClass {
	if class != sortNone && !spellableType(t, fl.pkg.tpkg) {
		return sortDynamic
	}
	return class
}

func (fl *fileLower) sortOrderFunc(t types2.Type, pos syntax.Pos) string {
	switch fl.sortClassOf(t) {
	case sortOrdered:
		return rtAlias + ".CompareOrdered"
	case sortBool:
		return rtAlias + ".CompareBool"
	case sortComplex:
		return rtAlias + ".CompareComplex"
	case sortStruct:
		return fl.structSortFunc(t, pos)
	case sortArray:
		return fl.arraySortFunc(t, pos)
	}
	return ""
}

func (fl *fileLower) structSortFunc(t types2.Type, pos syntax.Pos) string {
	st := compositeUnderlying(t).(*types2.Struct)
	var b strings.Builder
	fmt.Fprintf(&b, "func(%s, %s %s) int { ", sortOrderLeft, sortOrderRight, fl.mustSpell(t, pos))
	for i := range st.NumFields() {
		f := st.Field(i)
		if f.Name() == "_" {
			continue
		}
		if fn := fl.sortOrderFunc(f.Type(), pos); fn != "" {
			b.WriteString(sortOrderStep(fn, "."+f.Name()))
		}
	}
	b.WriteString("return 0 }")
	return b.String()
}

func (fl *fileLower) arraySortFunc(t types2.Type, pos syntax.Pos) string {
	at := compositeUnderlying(t).(*types2.Array)
	step := sortOrderStep(fl.sortOrderFunc(at.Elem(), pos), "["+sortOrderIndex+"]")
	return fmt.Sprintf("func(%s, %s %s) int { for %[4]s := 0; %[4]s < len(%[1]s); %[4]s++ { %[5]s}; return 0 }",
		sortOrderLeft, sortOrderRight, fl.mustSpell(t, pos), sortOrderIndex, step)
}

func sortOrderStep(fn, selector string) string {
	return fmt.Sprintf("if %[1]s := %[2]s(%[3]s%[5]s, %[4]s%[5]s); %[1]s != 0 { return %[1]s }; ",
		sortOrderDiff, fn, sortOrderLeft, sortOrderRight, selector)
}
