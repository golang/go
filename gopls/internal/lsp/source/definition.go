// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
)

// Definition handles the textDocument/definition request for Go files.
func Definition(ctx context.Context, snapshot Snapshot, fh FileHandle, position protocol.Position) ([]protocol.Location, error) {
	ctx, done := event.Start(ctx, "source.Definition")
	defer done()

	pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, fh.URI())
	if err != nil {
		return nil, err
	}
	pos, err := pgf.PositionPos(position)
	if err != nil {
		return nil, err
	}

	// Handle the case where the cursor is in an import.
	importLocations, err := importDefinition(ctx, snapshot, pkg, pgf, pos)
	if err != nil {
		return nil, err
	}
	if len(importLocations) > 0 {
		return importLocations, nil
	}

	// Handle the case where the cursor is in the package name.
	// We use "<= End" to accept a query immediately after the package name.
	if pgf.File != nil && pgf.File.Name.Pos() <= pos && pos <= pgf.File.Name.End() {
		// If there's no package documentation, just use current file.
		declFile := pgf
		for _, pgf := range pkg.CompiledGoFiles() {
			if pgf.File.Name != nil && pgf.File.Doc != nil {
				declFile = pgf
				break
			}
		}
		loc, err := declFile.NodeLocation(declFile.File.Name)
		if err != nil {
			return nil, err
		}
		return []protocol.Location{loc}, nil
	}

	// Handle the case where the cursor is in a linkname directive.
	locations, err := LinknameDefinition(ctx, snapshot, fh, position)
	if !errors.Is(err, ErrNoLinkname) {
		return locations, err
	}

	// Handle the case where the cursor is in an embed directive.
	locations, err = EmbedDefinition(pgf.Mapper, position)
	if !errors.Is(err, ErrNoEmbed) {
		return locations, err
	}

	// The general case: the cursor is on an identifier.
	_, obj, _ := referencedObject(pkg, pgf, pos)
	if obj == nil {
		return nil, nil
	}

	// Handle objects with no position: builtin, unsafe.
	if !obj.Pos().IsValid() {
		var pgf *ParsedGoFile
		if obj.Parent() == types.Universe {
			// pseudo-package "builtin"
			builtinPGF, err := snapshot.BuiltinFile(ctx)
			if err != nil {
				return nil, err
			}
			pgf = builtinPGF

		} else if obj.Pkg() == types.Unsafe {
			// package "unsafe"
			unsafe := snapshot.Metadata("unsafe")
			if unsafe == nil {
				return nil, fmt.Errorf("no metadata for package 'unsafe'")
			}
			uri := unsafe.GoFiles[0]
			fh, err := snapshot.ReadFile(ctx, uri)
			if err != nil {
				return nil, err
			}
			pgf, err = snapshot.ParseGo(ctx, fh, ParseFull&^SkipObjectResolution)
			if err != nil {
				return nil, err
			}

		} else {
			return nil, bug.Errorf("internal error: no position for %v", obj.Name())
		}
		// Inv: pgf ∈ {builtin,unsafe}.go

		// Use legacy (go/ast) object resolution.
		astObj := pgf.File.Scope.Lookup(obj.Name())
		if astObj == nil {
			// Every built-in should have documentation syntax.
			return nil, bug.Errorf("internal error: no object for %s", obj.Name())
		}
		decl, ok := astObj.Decl.(ast.Node)
		if !ok {
			return nil, bug.Errorf("internal error: no declaration for %s", obj.Name())
		}
		loc, err := pgf.PosLocation(decl.Pos(), decl.Pos()+token.Pos(len(obj.Name())))
		if err != nil {
			return nil, err
		}
		return []protocol.Location{loc}, nil
	}

	// Finally, map the object position.
	loc, err := mapPosition(ctx, pkg.FileSet(), snapshot, obj.Pos(), adjustedObjEnd(obj))
	if err != nil {
		return nil, err
	}
	return []protocol.Location{loc}, nil
}

// referencedObject returns the identifier and object referenced at the
// specified position, which must be within the file pgf, for the purposes of
// definition/hover/call hierarchy operations. It returns a nil object if no
// object was found at the given position.
//
// If the returned identifier is a type-switch implicit (i.e. the x in x :=
// e.(type)), the third result will be the type of the expression being
// switched on (the type of e in the example). This facilitates workarounds for
// limitations of the go/types API, which does not report an object for the
// identifier x.
//
// For embedded fields, referencedObject returns the type name object rather
// than the var (field) object.
//
// TODO(rfindley): this function exists to preserve the pre-existing behavior
// of source.Identifier. Eliminate this helper in favor of sharing
// functionality with objectsAt, after choosing suitable primitives.
func referencedObject(pkg Package, pgf *ParsedGoFile, pos token.Pos) (*ast.Ident, types.Object, types.Type) {
	path := pathEnclosingObjNode(pgf.File, pos)
	if len(path) == 0 {
		return nil, nil, nil
	}
	var obj types.Object
	info := pkg.GetTypesInfo()
	switch n := path[0].(type) {
	case *ast.Ident:
		obj = info.ObjectOf(n)
		// If n is the var's declaring ident in a type switch
		// [i.e. the x in x := foo.(type)], it will not have an object. In this
		// case, set obj to the first implicit object (if any), and return the type
		// of the expression being switched on.
		//
		// The type switch may have no case clauses and thus no
		// implicit objects; this is a type error ("unused x"),
		if obj == nil {
			if implicits, typ := typeSwitchImplicits(info, path); len(implicits) > 0 {
				return n, implicits[0], typ
			}
		}

		// If the original position was an embedded field, we want to jump
		// to the field's type definition, not the field's definition.
		if v, ok := obj.(*types.Var); ok && v.Embedded() {
			// types.Info.Uses contains the embedded field's *types.TypeName.
			if typeName := info.Uses[n]; typeName != nil {
				obj = typeName
			}
		}
		return n, obj, nil
	}
	return nil, nil, nil
}

// importDefinition returns locations defining a package referenced by the
// import spec containing pos.
//
// If pos is not inside an import spec, it returns nil, nil.
func importDefinition(ctx context.Context, s Snapshot, pkg Package, pgf *ParsedGoFile, pos token.Pos) ([]protocol.Location, error) {
	var imp *ast.ImportSpec
	for _, spec := range pgf.File.Imports {
		// We use "<= End" to accept a query immediately after an ImportSpec.
		if spec.Path.Pos() <= pos && pos <= spec.Path.End() {
			imp = spec
		}
	}
	if imp == nil {
		return nil, nil
	}

	importPath := UnquoteImportPath(imp)
	impID := pkg.Metadata().DepsByImpPath[importPath]
	if impID == "" {
		return nil, fmt.Errorf("failed to resolve import %q", importPath)
	}
	impMetadata := s.Metadata(impID)
	if impMetadata == nil {
		return nil, fmt.Errorf("missing information for package %q", impID)
	}

	var locs []protocol.Location
	for _, f := range impMetadata.CompiledGoFiles {
		fh, err := s.ReadFile(ctx, f)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			continue
		}
		pgf, err := s.ParseGo(ctx, fh, ParseHeader)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			continue
		}
		loc, err := pgf.NodeLocation(pgf.File)
		if err != nil {
			return nil, err
		}
		locs = append(locs, loc)
	}

	if len(locs) == 0 {
		return nil, fmt.Errorf("package %q has no readable files", impID) // incl. unsafe
	}

	return locs, nil
}

// TODO(rfindley): avoid the duplicate column mapping here, by associating a
// column mapper with each file handle.
func mapPosition(ctx context.Context, fset *token.FileSet, s FileSource, start, end token.Pos) (protocol.Location, error) {
	file := fset.File(start)
	uri := span.URIFromPath(file.Name())
	fh, err := s.ReadFile(ctx, uri)
	if err != nil {
		return protocol.Location{}, err
	}
	content, err := fh.Content()
	if err != nil {
		return protocol.Location{}, err
	}
	m := protocol.NewMapper(fh.URI(), content)
	return m.PosLocation(file, start, end)
}
