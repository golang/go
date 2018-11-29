// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

// Attribute is a set of common symbol attributes.
type Attribute int32

const (
	// AttrDuplicateOK marks a symbol that can be present in multiple object
	// files.
	AttrDuplicateOK Attribute = 1 << iota
	// AttrExternal marks function symbols loaded from host object files.
	AttrExternal
	// AttrNoSplit marks functions that cannot split the stack; the linker
	// cares because it checks that there are no call chains of nosplit
	// functions that require more than StackLimit bytes (see
	// lib.go:dostkcheck)
	AttrNoSplit
	// AttrReachable marks symbols that are transitively referenced from the
	// entry points. Unreachable symbols are not written to the output.
	AttrReachable
	// AttrCgoExportDynamic and AttrCgoExportStatic mark symbols referenced
	// by directives written by cgo (in response to //export directives in
	// the source).
	AttrCgoExportDynamic
	AttrCgoExportStatic
	// AttrSpecial marks symbols that do not have their address (i.e. Value)
	// computed by the usual mechanism of data.go:dodata() &
	// data.go:address().
	AttrSpecial
	// AttrStackCheck is used by dostkcheck to only check each NoSplit
	// function's stack usage once.
	AttrStackCheck
	// AttrNotInSymbolTable marks symbols that are not written to the symbol table.
	AttrNotInSymbolTable
	// AttrOnList marks symbols that are on some list (such as the list of
	// all text symbols, or one of the lists of data symbols) and is
	// consulted to avoid bugs where a symbol is put on a list twice.
	AttrOnList
	// AttrLocal marks symbols that are only visible within the module
	// (executable or shared library) being linked. Only relevant when
	// dynamically linking Go code.
	AttrLocal
	// AttrReflectMethod marks certain methods from the reflect package that
	// can be used to call arbitrary methods. If no symbol with this bit set
	// is marked as reachable, more dead code elimination can be done.
	AttrReflectMethod
	// AttrMakeTypelink Amarks types that should be added to the typelink
	// table. See typelinks.go:typelinks().
	AttrMakeTypelink
	// AttrShared marks symbols compiled with the -shared option.
	AttrShared
	// AttrVisibilityHidden symbols are ELF symbols with
	// visibility set to STV_HIDDEN. They become local symbols in
	// the final executable. Only relevant when internally linking
	// on an ELF platform.
	AttrVisibilityHidden
	// AttrSubSymbol mostly means that the symbol appears on the Sub list of some
	// other symbol.  Unfortunately, it's not 100% reliable; at least, it's not set
	// correctly for the .TOC. symbol in Link.dodata.  Usually the Outer field of the
	// symbol points to the symbol whose list it is on, but that it is not set for the
	// symbols added to .windynamic in initdynimport in pe.go.
	//
	// TODO(mwhudson): fix the inconsistencies noticed above.
	//
	// Sub lists are used when loading host objects (sections from the host object
	// become regular linker symbols and symbols go on the Sub list of their section)
	// and for constructing the global offset table when internally linking a dynamic
	// executable.
	//
	// TODO(mwhudson): perhaps a better name for this is AttrNonGoSymbol.
	AttrSubSymbol
	// AttrContainer is set on text symbols that are present as the .Outer for some
	// other symbol.
	AttrContainer
	// 17 attributes defined so far.
)

func (a Attribute) DuplicateOK() bool      { return a&AttrDuplicateOK != 0 }
func (a Attribute) External() bool         { return a&AttrExternal != 0 }
func (a Attribute) NoSplit() bool          { return a&AttrNoSplit != 0 }
func (a Attribute) Reachable() bool        { return a&AttrReachable != 0 }
func (a Attribute) CgoExportDynamic() bool { return a&AttrCgoExportDynamic != 0 }
func (a Attribute) CgoExportStatic() bool  { return a&AttrCgoExportStatic != 0 }
func (a Attribute) Special() bool          { return a&AttrSpecial != 0 }
func (a Attribute) StackCheck() bool       { return a&AttrStackCheck != 0 }
func (a Attribute) NotInSymbolTable() bool { return a&AttrNotInSymbolTable != 0 }
func (a Attribute) OnList() bool           { return a&AttrOnList != 0 }
func (a Attribute) Local() bool            { return a&AttrLocal != 0 }
func (a Attribute) ReflectMethod() bool    { return a&AttrReflectMethod != 0 }
func (a Attribute) MakeTypelink() bool     { return a&AttrMakeTypelink != 0 }
func (a Attribute) Shared() bool           { return a&AttrShared != 0 }
func (a Attribute) VisibilityHidden() bool { return a&AttrVisibilityHidden != 0 }
func (a Attribute) SubSymbol() bool        { return a&AttrSubSymbol != 0 }
func (a Attribute) Container() bool        { return a&AttrContainer != 0 }

func (a Attribute) CgoExport() bool {
	return a.CgoExportDynamic() || a.CgoExportStatic()
}

func (a *Attribute) Set(flag Attribute, value bool) {
	if value {
		*a |= flag
	} else {
		*a &^= flag
	}
}
