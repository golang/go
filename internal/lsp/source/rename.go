// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"regexp"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/refactor/satisfy"
)

type renamer struct {
	fset               *token.FileSet
	pkg                Package // the package containing the declaration of the ident
	refs               []*ReferenceInfo
	objsToUpdate       map[types.Object]bool
	hadConflicts       bool
	errors             string
	from, to           string
	satisfyConstraints map[satisfy.Constraint]bool
	packages           map[*types.Package]Package // may include additional packages that are a rdep of pkg
	msets              typeutil.MethodSetCache
	changeMethods      bool
}

// Rename returns a map of TextEdits for each file modified when renaming a given identifier within a package.
func Rename(ctx context.Context, view View, f GoFile, pos token.Pos, newName string) (map[span.URI][]TextEdit, error) {
	pkg := f.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, fmt.Errorf("package for %s is ill typed", f.URI())
	}

	// Get the identifier to rename.
	ident, err := Identifier(ctx, view, f, pos)
	if err != nil {
		return nil, err
	}
	if ident.Name == newName {
		return nil, fmt.Errorf("old and new names are the same: %s", newName)
	}
	if !isValidIdentifier(ident.Name) {
		return nil, fmt.Errorf("invalid identifier to rename: %q", ident.Name)
	}

	// Do not rename identifiers declared in another package.
	if pkg.GetTypes() != ident.decl.obj.Pkg() {
		return nil, fmt.Errorf("failed to rename because %q is declared in package %q", ident.Name, ident.decl.obj.Pkg().Name())
	}

	// TODO(suzmue): Support renaming of imported packages.
	if _, ok := ident.decl.obj.(*types.PkgName); ok {
		return nil, fmt.Errorf("renaming imported package %s not supported", ident.Name)
	}

	refs, err := ident.References(ctx)
	if err != nil {
		return nil, err
	}

	r := renamer{
		fset:         f.FileSet(),
		pkg:          pkg,
		refs:         refs,
		objsToUpdate: make(map[types.Object]bool),
		from:         ident.Name,
		to:           newName,
		packages:     make(map[*types.Package]Package),
	}
	r.packages[pkg.GetTypes()] = pkg

	// Check that the renaming of the identifier is ok.
	for _, from := range refs {
		r.check(from.obj)
	}
	if r.hadConflicts {
		return nil, fmt.Errorf(r.errors)
	}

	return r.update(ctx, view)
}

// Rename all references to the identifier.
func (r *renamer) update(ctx context.Context, view View) (map[span.URI][]TextEdit, error) {
	result := make(map[span.URI][]TextEdit)

	docRegexp := regexp.MustCompile(`\b` + r.from + `\b`)
	for _, ref := range r.refs {
		refSpan, err := ref.Range.Span()
		if err != nil {
			return nil, err
		}

		edit := TextEdit{
			Span:    refSpan,
			NewText: r.to,
		}
		result[refSpan.URI()] = append(result[refSpan.URI()], edit)

		if ref.isDeclaration {
			// Perform the rename in doc comments too (declared in the original package)
			if doc := r.docComment(r.pkg, ref.ident); doc != nil {
				for _, comment := range doc.List {
					for _, locs := range docRegexp.FindAllStringIndex(comment.Text, -1) {
						rng := span.NewRange(r.fset, comment.Pos()+token.Pos(locs[0]), comment.Pos()+token.Pos(locs[1]))
						spn, err := rng.Span()
						if err != nil {
							return nil, err
						}
						result[refSpan.URI()] = append(result[refSpan.URI()], TextEdit{
							Span:    spn,
							NewText: r.to,
						})
					}
					comment.Text = docRegexp.ReplaceAllString(comment.Text, r.to)
				}
			}
		}

	}

	return result, nil
}

// docComment returns the doc for an identifier.
func (r *renamer) docComment(pkg Package, id *ast.Ident) *ast.CommentGroup {
	_, nodes, _ := pathEnclosingInterval(r.fset, pkg, id.Pos(), id.End())
	for _, node := range nodes {
		switch decl := node.(type) {
		case *ast.FuncDecl:
			return decl.Doc
		case *ast.Field:
			return decl.Doc
		case *ast.GenDecl:
			return decl.Doc
		// For {Type,Value}Spec, if the doc on the spec is absent,
		// search for the enclosing GenDecl
		case *ast.TypeSpec:
			if decl.Doc != nil {
				return decl.Doc
			}
		case *ast.ValueSpec:
			if decl.Doc != nil {
				return decl.Doc
			}
		case *ast.Ident:
		default:
			return nil
		}
	}
	return nil
}
