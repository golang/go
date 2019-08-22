// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"regexp"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/trace"
	"golang.org/x/tools/refactor/satisfy"
	errors "golang.org/x/xerrors"
)

type renamer struct {
	ctx                context.Context
	fset               *token.FileSet
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
func (i *IdentifierInfo) Rename(ctx context.Context, newName string) (map[span.URI][]diff.TextEdit, error) {
	ctx, done := trace.StartSpan(ctx, "source.Rename")
	defer done()

	// If the object declaration is nil, assume it is an import spec.
	if i.decl.obj == nil {
		// Find the corresponding package name for this import spec
		// and rename that instead.
		ident, err := i.getPkgName(ctx)
		if err != nil {
			return nil, err
		}
		return ident.Rename(ctx, newName)
	}
	if i.Name == newName {
		return nil, errors.Errorf("old and new names are the same: %s", newName)
	}
	if !isValidIdentifier(i.Name) {
		return nil, errors.Errorf("invalid identifier to rename: %q", i.Name)
	}
	// Do not rename builtin identifiers.
	if i.decl.obj.Parent() == types.Universe {
		return nil, errors.Errorf("cannot rename builtin %q", i.Name)
	}
	if i.pkg == nil || i.pkg.IsIllTyped() {
		return nil, errors.Errorf("package for %s is ill typed", i.File.URI())
	}
	// Do not rename identifiers declared in another package.
	if i.pkg.GetTypes() != i.decl.obj.Pkg() {
		return nil, errors.Errorf("failed to rename because %q is declared in package %q", i.Name, i.decl.obj.Pkg().Name())
	}

	refs, err := i.References(ctx)
	if err != nil {
		return nil, err
	}

	r := renamer{
		ctx:          ctx,
		fset:         i.File.FileSet(),
		refs:         refs,
		objsToUpdate: make(map[types.Object]bool),
		from:         i.Name,
		to:           newName,
		packages:     make(map[*types.Package]Package),
	}
	for _, from := range refs {
		r.packages[from.pkg.GetTypes()] = from.pkg
	}

	// Check that the renaming of the identifier is ok.
	for _, ref := range refs {
		r.check(ref.obj)
		if r.hadConflicts { // one error is enough.
			break
		}
	}
	if r.hadConflicts {
		return nil, errors.Errorf(r.errors)
	}

	changes, err := r.update()
	if err != nil {
		return nil, err
	}

	// Sort edits for each file.
	for _, edits := range changes {
		diff.SortTextEdits(edits)
	}
	return changes, nil
}

// getPkgName gets the pkg name associated with an identifer representing
// the import path in an import spec.
func (i *IdentifierInfo) getPkgName(ctx context.Context) (*IdentifierInfo, error) {
	file := i.File.FileSet().File(i.Range.Start)
	pkgLine := file.Line(i.Range.Start)

	for _, obj := range i.pkg.GetTypesInfo().Defs {
		pkgName, ok := obj.(*types.PkgName)
		if ok && file.Line(pkgName.Pos()) == pkgLine {
			return getPkgNameIdentifier(ctx, i, pkgName)
		}
	}
	for _, obj := range i.pkg.GetTypesInfo().Implicits {
		pkgName, ok := obj.(*types.PkgName)
		if ok && file.Line(pkgName.Pos()) == pkgLine {
			return getPkgNameIdentifier(ctx, i, pkgName)
		}
	}
	return nil, errors.Errorf("no package name for %q", i.Name)
}

// getPkgNameIdentifier returns an IdentifierInfo representing pkgName.
// pkgName must be in the same package and file as ident.
func getPkgNameIdentifier(ctx context.Context, ident *IdentifierInfo, pkgName *types.PkgName) (*IdentifierInfo, error) {
	decl := declaration{
		obj:         pkgName,
		wasImplicit: true,
	}
	var err error
	if decl.rng, err = objToRange(ctx, ident.File.FileSet(), decl.obj); err != nil {
		return nil, err
	}
	if decl.node, err = objToNode(ctx, ident.File.View(), ident.pkg.GetTypes(), decl.obj, decl.rng); err != nil {
		return nil, err
	}
	return &IdentifierInfo{
		Name:             pkgName.Name(),
		Range:            decl.rng,
		File:             ident.File,
		decl:             decl,
		pkg:              ident.pkg,
		wasEmbeddedField: false,
		qf:               ident.qf,
	}, nil
}

// Rename all references to the identifier.
func (r *renamer) update() (map[span.URI][]diff.TextEdit, error) {
	result := make(map[span.URI][]diff.TextEdit)
	seen := make(map[span.Span]bool)

	docRegexp, err := regexp.Compile(`\b` + r.from + `\b`)
	if err != nil {
		return nil, err
	}
	for _, ref := range r.refs {
		refSpan, err := ref.Range.Span()
		if err != nil {
			return nil, err
		}
		if seen[refSpan] {
			continue
		}
		seen[refSpan] = true

		// Renaming a types.PkgName may result in the addition or removal of an identifier,
		// so we deal with this separately.
		if pkgName, ok := ref.obj.(*types.PkgName); ok && ref.isDeclaration {
			edit, err := r.updatePkgName(pkgName)
			if err != nil {
				return nil, err
			}
			result[refSpan.URI()] = append(result[refSpan.URI()], *edit)
			continue
		}

		// Replace the identifier with r.to.
		edit := diff.TextEdit{
			Span:    refSpan,
			NewText: r.to,
		}

		result[refSpan.URI()] = append(result[refSpan.URI()], edit)

		if !ref.isDeclaration || ref.ident == nil { // uses do not have doc comments to update.
			continue
		}

		doc := r.docComment(ref.pkg, ref.ident)
		if doc == nil {
			continue
		}

		// Perform the rename in doc comments declared in the original package.
		for _, comment := range doc.List {
			for _, locs := range docRegexp.FindAllStringIndex(comment.Text, -1) {
				rng := span.NewRange(r.fset, comment.Pos()+token.Pos(locs[0]), comment.Pos()+token.Pos(locs[1]))
				spn, err := rng.Span()
				if err != nil {
					return nil, err
				}
				result[spn.URI()] = append(result[spn.URI()], diff.TextEdit{
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

// updatePkgName returns the updates to rename a pkgName in the import spec
func (r *renamer) updatePkgName(pkgName *types.PkgName) (*diff.TextEdit, error) {
	// Modify ImportSpec syntax to add or remove the Name as needed.
	pkg := r.packages[pkgName.Pkg()]
	_, path, _ := pathEnclosingInterval(r.ctx, r.fset, pkg, pkgName.Pos(), pkgName.Pos())
	if len(path) < 2 {
		return nil, errors.Errorf("no path enclosing interval for %s", pkgName.Name())
	}
	spec, ok := path[1].(*ast.ImportSpec)
	if !ok {
		return nil, errors.Errorf("failed to update PkgName for %s", pkgName.Name())
	}

	var astIdent *ast.Ident // will be nil if ident is removed
	if pkgName.Imported().Name() != r.to {
		// ImportSpec.Name needed
		astIdent = &ast.Ident{NamePos: spec.Path.Pos(), Name: r.to}
	}

	// Make a copy of the ident that just has the name and path.
	updated := &ast.ImportSpec{
		Name:   astIdent,
		Path:   spec.Path,
		EndPos: spec.EndPos,
	}

	rng := span.NewRange(r.fset, spec.Pos(), spec.End())
	spn, err := rng.Span()
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	format.Node(&buf, r.fset, updated)
	newText := buf.String()

	return &diff.TextEdit{
		Span:    spn,
		NewText: newText,
	}, nil
}
