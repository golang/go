// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package analysis

// This file computes the markup for information from go/types:
// IMPORTS, identifier RESOLUTION, METHOD SETS, size/alignment, and
// the IMPLEMENTS relation.
//
// IMPORTS links connect import specs to the documentation for the
// imported package.
//
// RESOLUTION links referring identifiers to their defining
// identifier, and adds tooltips for kind and type.
//
// METHOD SETS, size/alignment, and the IMPLEMENTS relation are
// displayed in the lower pane when a type's defining identifier is
// clicked.

import (
	"fmt"
	"go/types"
	"reflect"
	"strconv"
	"strings"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/types/typeutil"
)

// TODO(adonovan): audit to make sure it's safe on ill-typed packages.

// TODO(adonovan): use same Sizes as loader.Config.
var sizes = types.StdSizes{WordSize: 8, MaxAlign: 8}

func (a *analysis) doTypeInfo(info *loader.PackageInfo, implements map[*types.Named]implementsFacts) {
	// We must not assume the corresponding SSA packages were
	// created (i.e. were transitively error-free).

	// IMPORTS
	for _, f := range info.Files {
		// Package decl.
		fi, offset := a.fileAndOffset(f.Name.Pos())
		fi.addLink(aLink{
			start: offset,
			end:   offset + len(f.Name.Name),
			title: "Package docs for " + info.Pkg.Path(),
			// TODO(adonovan): fix: we're putting the untrusted Path()
			// into a trusted field.  What's the appropriate sanitizer?
			href: "/pkg/" + info.Pkg.Path(),
		})

		// Import specs.
		for _, imp := range f.Imports {
			// Remove quotes.
			L := int(imp.End()-imp.Path.Pos()) - len(`""`)
			path, _ := strconv.Unquote(imp.Path.Value)
			fi, offset := a.fileAndOffset(imp.Path.Pos())
			fi.addLink(aLink{
				start: offset + 1,
				end:   offset + 1 + L,
				title: "Package docs for " + path,
				// TODO(adonovan): fix: we're putting the untrusted path
				// into a trusted field.  What's the appropriate sanitizer?
				href: "/pkg/" + path,
			})
		}
	}

	// RESOLUTION
	qualifier := types.RelativeTo(info.Pkg)
	for id, obj := range info.Uses {
		// Position of the object definition.
		pos := obj.Pos()
		Len := len(obj.Name())

		// Correct the position for non-renaming import specs.
		//  import "sync/atomic"
		//          ^^^^^^^^^^^
		if obj, ok := obj.(*types.PkgName); ok && id.Name == obj.Imported().Name() {
			// Assume this is a non-renaming import.
			// NB: not true for degenerate renamings: `import foo "foo"`.
			pos++
			Len = len(obj.Imported().Path())
		}

		if obj.Pkg() == nil {
			continue // don't mark up built-ins.
		}

		fi, offset := a.fileAndOffset(id.NamePos)
		fi.addLink(aLink{
			start: offset,
			end:   offset + len(id.Name),
			title: types.ObjectString(obj, qualifier),
			href:  a.posURL(pos, Len),
		})
	}

	// IMPLEMENTS & METHOD SETS
	for _, obj := range info.Defs {
		if obj, ok := obj.(*types.TypeName); ok {
			a.namedType(obj, implements)
		}
	}
}

func (a *analysis) namedType(obj *types.TypeName, implements map[*types.Named]implementsFacts) {
	qualifier := types.RelativeTo(obj.Pkg())
	T := obj.Type().(*types.Named)
	v := &TypeInfoJSON{
		Name:    obj.Name(),
		Size:    sizes.Sizeof(T),
		Align:   sizes.Alignof(T),
		Methods: []anchorJSON{}, // (JS wants non-nil)
	}

	// addFact adds the fact "is implemented by T" (by) or
	// "implements T" (!by) to group.
	addFact := func(group *implGroupJSON, T types.Type, by bool) {
		Tobj := deref(T).(*types.Named).Obj()
		var byKind string
		if by {
			// Show underlying kind of implementing type,
			// e.g. "slice", "array", "struct".
			s := reflect.TypeOf(T.Underlying()).String()
			byKind = strings.ToLower(strings.TrimPrefix(s, "*types."))
		}
		group.Facts = append(group.Facts, implFactJSON{
			ByKind: byKind,
			Other: anchorJSON{
				Href: a.posURL(Tobj.Pos(), len(Tobj.Name())),
				Text: types.TypeString(T, qualifier),
			},
		})
	}

	// IMPLEMENTS
	if r, ok := implements[T]; ok {
		if isInterface(T) {
			// "T is implemented by <conc>" ...
			// "T is implemented by <iface>"...
			// "T implements        <iface>"...
			group := implGroupJSON{
				Descr: types.TypeString(T, qualifier),
			}
			// Show concrete types first; use two passes.
			for _, sub := range r.to {
				if !isInterface(sub) {
					addFact(&group, sub, true)
				}
			}
			for _, sub := range r.to {
				if isInterface(sub) {
					addFact(&group, sub, true)
				}
			}
			for _, super := range r.from {
				addFact(&group, super, false)
			}
			v.ImplGroups = append(v.ImplGroups, group)
		} else {
			// T is concrete.
			if r.from != nil {
				// "T implements <iface>"...
				group := implGroupJSON{
					Descr: types.TypeString(T, qualifier),
				}
				for _, super := range r.from {
					addFact(&group, super, false)
				}
				v.ImplGroups = append(v.ImplGroups, group)
			}
			if r.fromPtr != nil {
				// "*C implements <iface>"...
				group := implGroupJSON{
					Descr: "*" + types.TypeString(T, qualifier),
				}
				for _, psuper := range r.fromPtr {
					addFact(&group, psuper, false)
				}
				v.ImplGroups = append(v.ImplGroups, group)
			}
		}
	}

	// METHOD SETS
	for _, sel := range typeutil.IntuitiveMethodSet(T, &a.prog.MethodSets) {
		meth := sel.Obj().(*types.Func)
		pos := meth.Pos() // may be 0 for error.Error
		v.Methods = append(v.Methods, anchorJSON{
			Href: a.posURL(pos, len(meth.Name())),
			Text: types.SelectionString(sel, qualifier),
		})
	}

	// Since there can be many specs per decl, we
	// can't attach the link to the keyword 'type'
	// (as we do with 'func'); we use the Ident.
	fi, offset := a.fileAndOffset(obj.Pos())
	fi.addLink(aLink{
		start:   offset,
		end:     offset + len(obj.Name()),
		title:   fmt.Sprintf("type info for %s", obj.Name()),
		onclick: fmt.Sprintf("onClickTypeInfo(%d)", fi.addData(v)),
	})

	// Add info for exported package-level types to the package info.
	if obj.Exported() && isPackageLevel(obj) {
		// TODO(adonovan): Path is not unique!
		// It is possible to declare a non-test package called x_test.
		a.result.pkgInfo(obj.Pkg().Path()).addType(v)
	}
}

// -- utilities --------------------------------------------------------

func isInterface(T types.Type) bool { return types.IsInterface(T) }

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}

// isPackageLevel reports whether obj is a package-level object.
func isPackageLevel(obj types.Object) bool {
	return obj.Pkg().Scope().Lookup(obj.Name()) == obj
}
