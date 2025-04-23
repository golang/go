// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeindex provides an [Index] of type information for a
// package, allowing efficient lookup of, say, whether a given symbol
// is referenced and, if so, where from; or of the [cursor.Cursor] for
// the declaration of a particular [types.Object] symbol.
package typeindex

import (
	"encoding/binary"
	"go/ast"
	"go/types"
	"iter"

	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/astutil/cursor"
	"golang.org/x/tools/internal/astutil/edge"
	"golang.org/x/tools/internal/typesinternal"
)

// New constructs an Index for the package of type-annotated syntax
//
// TODO(adonovan): accept a FileSet too?
// We regret not requiring one in inspector.New.
func New(inspect *inspector.Inspector, pkg *types.Package, info *types.Info) *Index {
	ix := &Index{
		inspect:  inspect,
		info:     info,
		packages: make(map[string]*types.Package),
		def:      make(map[types.Object]cursor.Cursor),
		uses:     make(map[types.Object]*uses),
	}

	addPackage := func(pkg2 *types.Package) {
		if pkg2 != nil && pkg2 != pkg {
			ix.packages[pkg2.Path()] = pkg2
		}
	}

	for cur := range cursor.Root(inspect).Preorder((*ast.ImportSpec)(nil), (*ast.Ident)(nil)) {
		switch n := cur.Node().(type) {
		case *ast.ImportSpec:
			// Index direct imports, including blank ones.
			if pkgname := info.PkgNameOf(n); pkgname != nil {
				addPackage(pkgname.Imported())
			}

		case *ast.Ident:
			// Index all defining and using identifiers.
			if obj := info.Defs[n]; obj != nil {
				ix.def[obj] = cur
			}

			if obj := info.Uses[n]; obj != nil {
				// Index indirect dependencies (via fields and methods).
				if !typesinternal.IsPackageLevel(obj) {
					addPackage(obj.Pkg())
				}

				us, ok := ix.uses[obj]
				if !ok {
					us = &uses{}
					us.code = us.initial[:0]
					ix.uses[obj] = us
				}
				delta := cur.Index() - us.last
				if delta < 0 {
					panic("non-monotonic")
				}
				us.code = binary.AppendUvarint(us.code, uint64(delta))
				us.last = cur.Index()
			}
		}
	}
	return ix
}

// An Index holds an index mapping [types.Object] symbols to their syntax.
// In effect, it is the inverse of [types.Info].
type Index struct {
	inspect  *inspector.Inspector
	info     *types.Info
	packages map[string]*types.Package      // packages of all symbols referenced from this package
	def      map[types.Object]cursor.Cursor // Cursor of *ast.Ident that defines the Object
	uses     map[types.Object]*uses         // Cursors of *ast.Idents that use the Object
}

// A uses holds the list of Cursors of Idents that use a given symbol.
//
// The Uses map of [types.Info] is substantial, so it pays to compress
// its inverse mapping here, both in space and in CPU due to reduced
// allocation. A Cursor is 2 words; a Cursor.Index is 4 bytes; but
// since Cursors are naturally delivered in ascending order, we can
// use varint-encoded deltas at a cost of only ~1.7-2.2 bytes per use.
//
// Many variables have only one or two uses, so their encoded uses may
// fit in the 4 bytes of initial, saving further CPU and space
// essentially for free since the struct's size class is 4 words.
type uses struct {
	code    []byte  // varint-encoded deltas of successive Cursor.Index values
	last    int32   // most recent Cursor.Index value; used during encoding
	initial [4]byte // use slack in size class as initial space for code
}

// Uses returns the sequence of Cursors of [*ast.Ident]s in this package
// that refer to obj. If obj is nil, the sequence is empty.
func (ix *Index) Uses(obj types.Object) iter.Seq[cursor.Cursor] {
	return func(yield func(cursor.Cursor) bool) {
		if uses := ix.uses[obj]; uses != nil {
			var last int32
			for code := uses.code; len(code) > 0; {
				delta, n := binary.Uvarint(code)
				last += int32(delta)
				if !yield(cursor.At(ix.inspect, last)) {
					return
				}
				code = code[n:]
			}
		}
	}
}

// Used reports whether any of the specified objects are used, in
// other words, obj != nil && Uses(obj) is non-empty for some obj in objs.
//
// (This treatment of nil allows Used to be called directly on the
// result of [Index.Object] so that analyzers can conveniently skip
// packages that don't use a symbol of interest.)
func (ix *Index) Used(objs ...types.Object) bool {
	for _, obj := range objs {
		if obj != nil && ix.uses[obj] != nil {
			return true
		}
	}
	return false
}

// Def returns the Cursor of the [*ast.Ident] in this package
// that declares the specified object, if any.
func (ix *Index) Def(obj types.Object) (cursor.Cursor, bool) {
	cur, ok := ix.def[obj]
	return cur, ok
}

// Package returns the package of the specified path,
// or nil if it is not referenced from this package.
func (ix *Index) Package(path string) *types.Package {
	return ix.packages[path]
}

// Object returns the package-level symbol name within the package of
// the specified path, or nil if the package or symbol does not exist
// or is not visible from this package.
func (ix *Index) Object(path, name string) types.Object {
	if pkg := ix.Package(path); pkg != nil {
		return pkg.Scope().Lookup(name)
	}
	return nil
}

// Selection returns the named method or field belonging to the
// package-level type returned by Object(path, typename).
func (ix *Index) Selection(path, typename, name string) types.Object {
	if obj := ix.Object(path, typename); obj != nil {
		if tname, ok := obj.(*types.TypeName); ok {
			obj, _, _ := types.LookupFieldOrMethod(tname.Type(), true, obj.Pkg(), name)
			return obj
		}
	}
	return nil
}

// Calls returns the sequence of cursors for *ast.CallExpr nodes that
// call the specified callee, as defined by [typeutil.Callee].
// If callee is nil, the sequence is empty.
func (ix *Index) Calls(callee types.Object) iter.Seq[cursor.Cursor] {
	return func(yield func(cursor.Cursor) bool) {
		for cur := range ix.Uses(callee) {
			ek, _ := cur.ParentEdge()

			// The call may be of the form f() or x.f(),
			// optionally with parens; ascend from f to call.
			//
			// It is tempting but wrong to use the first
			// CallExpr ancestor: we have to make sure the
			// ident is in the CallExpr.Fun position, otherwise
			// f(f, f) would have two spurious matches.
			// Avoiding Enclosing is also significantly faster.

			// inverse unparen: f -> (f)
			for ek == edge.ParenExpr_X {
				cur = cur.Parent()
				ek, _ = cur.ParentEdge()
			}

			// ascend selector: f -> x.f
			if ek == edge.SelectorExpr_Sel {
				cur = cur.Parent()
				ek, _ = cur.ParentEdge()
			}

			// inverse unparen again
			for ek == edge.ParenExpr_X {
				cur = cur.Parent()
				ek, _ = cur.ParentEdge()
			}

			// ascend from f or x.f to call
			if ek == edge.CallExpr_Fun {
				curCall := cur.Parent()
				call := curCall.Node().(*ast.CallExpr)
				if typeutil.Callee(ix.info, call) == callee {
					if !yield(curCall) {
						return
					}
				}
			}
		}
	}
}
