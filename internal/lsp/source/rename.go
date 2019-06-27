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
	ctx                context.Context
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
func (i *IdentifierInfo) Rename(ctx context.Context, newName string) (map[span.URI][]TextEdit, error) {
	if i.Name == newName {
		return nil, fmt.Errorf("old and new names are the same: %s", newName)
	}
	if !isValidIdentifier(i.Name) {
		return nil, fmt.Errorf("invalid identifier to rename: %q", i.Name)
	}

	// Do not rename identifiers declared in another package.
	if i.pkg == nil || i.pkg.IsIllTyped() {
		return nil, fmt.Errorf("package for %s is ill typed", i.File.URI())
	}
	if i.pkg.GetTypes() != i.decl.obj.Pkg() {
		return nil, fmt.Errorf("failed to rename because %q is declared in package %q", i.Name, i.decl.obj.Pkg().Name())
	}

	// TODO(suzmue): Support renaming of imported packages.
	if _, ok := i.decl.obj.(*types.PkgName); ok {
		return nil, fmt.Errorf("renaming imported package %s not supported", i.Name)
	}

	refs, err := i.References(ctx)
	if err != nil {
		return nil, err
	}

	r := renamer{
		fset:         i.File.FileSet(),
		pkg:          i.pkg,
		refs:         refs,
		objsToUpdate: make(map[types.Object]bool),
		from:         i.Name,
		to:           newName,
		packages:     make(map[*types.Package]Package),
	}
	r.packages[i.pkg.GetTypes()] = i.pkg

	// Check that the renaming of the identifier is ok.
	for _, from := range refs {
		r.check(from.obj)
	}
	if r.hadConflicts {
		return nil, fmt.Errorf(r.errors)
	}

	return r.update(ctx)
}

// Rename all references to the identifier.
func (r *renamer) update(ctx context.Context) (map[span.URI][]TextEdit, error) {
	result := make(map[span.URI][]TextEdit)

	docRegexp, err := regexp.Compile(`\b` + r.from + `\b`)
	if err != nil {
		return nil, err
	}
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

		if !ref.isDeclaration { // not a declaration
			continue
		}

		doc := r.docComment(r.pkg, ref.ident)
		if doc == nil { // no doc comment
			continue
		}

		// Perform the rename in doc comments declared in the original package
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
		}
	}

	return result, nil
}

// docComment returns the doc for an identifier.
func (r *renamer) docComment(pkg Package, id *ast.Ident) *ast.CommentGroup {
	_, nodes, _ := pathEnclosingInterval(r.ctx, r.fset, pkg, id.Pos(), id.End())
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
