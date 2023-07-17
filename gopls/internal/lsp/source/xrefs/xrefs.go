// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package xrefs defines the serializable index of cross-package
// references that is computed during type checking.
//
// See ../references.go for the 'references' query.
package xrefs

import (
	"go/ast"
	"go/types"
	"sort"

	"golang.org/x/tools/go/types/objectpath"
	"golang.org/x/tools/gopls/internal/lsp/frob"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/typeparams"
)

// Index constructs a serializable index of outbound cross-references
// for the specified type-checked package.
func Index(files []*source.ParsedGoFile, pkg *types.Package, info *types.Info) []byte {
	// pkgObjects maps each referenced package Q to a mapping:
	// from each referenced symbol in Q to the ordered list
	// of references to that symbol from this package.
	// A nil types.Object indicates a reference
	// to the package as a whole: an import.
	pkgObjects := make(map[*types.Package]map[types.Object]*gobObject)

	// getObjects returns the object-to-references mapping for a package.
	getObjects := func(pkg *types.Package) map[types.Object]*gobObject {
		objects, ok := pkgObjects[pkg]
		if !ok {
			objects = make(map[types.Object]*gobObject)
			pkgObjects[pkg] = objects
		}
		return objects
	}

	objectpathFor := new(objectpath.Encoder).For

	for fileIndex, pgf := range files {

		nodeRange := func(n ast.Node) protocol.Range {
			rng, err := pgf.PosRange(n.Pos(), n.End())
			if err != nil {
				panic(err) // can't fail
			}
			return rng
		}

		ast.Inspect(pgf.File, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.Ident:
				// Report a reference for each identifier that
				// uses a symbol exported from another package.
				// (The built-in error.Error method has no package.)
				if n.IsExported() {
					if obj, ok := info.Uses[n]; ok &&
						obj.Pkg() != nil &&
						obj.Pkg() != pkg {

						// For instantiations of generic methods,
						// use the generic object (see issue #60622).
						if fn, ok := obj.(*types.Func); ok {
							obj = typeparams.OriginMethod(fn)
						}

						objects := getObjects(obj.Pkg())
						gobObj, ok := objects[obj]
						if !ok {
							path, err := objectpathFor(obj)
							if err != nil {
								// Capitalized but not exported
								// (e.g. local const/var/type).
								return true
							}
							gobObj = &gobObject{Path: path}
							objects[obj] = gobObj
						}

						gobObj.Refs = append(gobObj.Refs, gobRef{
							FileIndex: fileIndex,
							Range:     nodeRange(n),
						})
					}
				}

			case *ast.ImportSpec:
				// Report a reference from each import path
				// string to the imported package.
				pkgname, ok := source.ImportedPkgName(info, n)
				if !ok {
					return true // missing import
				}
				objects := getObjects(pkgname.Imported())
				gobObj, ok := objects[nil]
				if !ok {
					gobObj = &gobObject{Path: ""}
					objects[nil] = gobObj
				}
				gobObj.Refs = append(gobObj.Refs, gobRef{
					FileIndex: fileIndex,
					Range:     nodeRange(n.Path),
				})
			}
			return true
		})
	}

	// Flatten the maps into slices, and sort for determinism.
	var packages []*gobPackage
	for p := range pkgObjects {
		objects := pkgObjects[p]
		gp := &gobPackage{
			PkgPath: source.PackagePath(p.Path()),
			Objects: make([]*gobObject, 0, len(objects)),
		}
		for _, gobObj := range objects {
			gp.Objects = append(gp.Objects, gobObj)
		}
		sort.Slice(gp.Objects, func(i, j int) bool {
			return gp.Objects[i].Path < gp.Objects[j].Path
		})
		packages = append(packages, gp)
	}
	sort.Slice(packages, func(i, j int) bool {
		return packages[i].PkgPath < packages[j].PkgPath
	})

	return packageCodec.Encode(packages)
}

// Lookup searches a serialized index produced by an indexPackage
// operation on m, and returns the locations of all references from m
// to any object in the target set. Each object is denoted by a pair
// of (package path, object path).
func Lookup(m *source.Metadata, data []byte, targets map[source.PackagePath]map[objectpath.Path]struct{}) (locs []protocol.Location) {
	var packages []*gobPackage
	packageCodec.Decode(data, &packages)
	for _, gp := range packages {
		if objectSet, ok := targets[gp.PkgPath]; ok {
			for _, gobObj := range gp.Objects {
				if _, ok := objectSet[gobObj.Path]; ok {
					for _, ref := range gobObj.Refs {
						uri := m.CompiledGoFiles[ref.FileIndex]
						locs = append(locs, protocol.Location{
							URI:   protocol.URIFromSpanURI(uri),
							Range: ref.Range,
						})
					}
				}
			}
		}
	}

	return locs
}

// -- serialized representation --

// The cross-reference index records the location of all references
// from one package to symbols defined in other packages
// (dependencies). It does not record within-package references.
// The index for package P consists of a list of gopPackage records,
// each enumerating references to symbols defined a single dependency, Q.

// TODO(adonovan): opt: choose a more compact encoding.
// The gobRef.Range field is the obvious place to begin.

// (The name says gob but in fact we use frob.)
// var packageCodec = frob.For[[]*gobPackage]()
var packageCodec = frob.CodecFor117(new([]*gobPackage))

// A gobPackage records the set of outgoing references from the index
// package to symbols defined in a dependency package.
type gobPackage struct {
	PkgPath source.PackagePath // defining package (Q)
	Objects []*gobObject       // set of Q objects referenced by P
}

// A gobObject records all references to a particular symbol.
type gobObject struct {
	Path objectpath.Path // symbol name within package; "" => import of package itself
	Refs []gobRef        // locations of references within P, in lexical order
}

type gobRef struct {
	FileIndex int            // index of enclosing file within P's CompiledGoFiles
	Range     protocol.Range // source range of reference
}
