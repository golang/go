// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"regexp"
	"strings"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/refactor/satisfy"
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
	packages           map[*types.Package]Package // may include additional packages that are a dep of pkg
	msets              typeutil.MethodSetCache
	changeMethods      bool
}

type PrepareItem struct {
	Range protocol.Range
	Text  string
}

// PrepareRename searches for a valid renaming at position pp.
//
// The returned usererr is intended to be displayed to the user to explain why
// the prepare fails. Probably we could eliminate the redundancy in returning
// two errors, but for now this is done defensively.
func PrepareRename(ctx context.Context, snapshot Snapshot, f FileHandle, pp protocol.Position) (_ *PrepareItem, usererr, err error) {
	ctx, done := event.Start(ctx, "source.PrepareRename")
	defer done()

	qos, err := qualifiedObjsAtProtocolPos(ctx, snapshot, f.URI(), pp)
	if err != nil {
		return nil, nil, err
	}
	node, obj, pkg := qos[0].node, qos[0].obj, qos[0].sourcePkg
	if err := checkRenamable(obj); err != nil {
		return nil, err, err
	}
	mr, err := posToMappedRange(snapshot, pkg, node.Pos(), node.End())
	if err != nil {
		return nil, nil, err
	}
	rng, err := mr.Range()
	if err != nil {
		return nil, nil, err
	}
	if _, isImport := node.(*ast.ImportSpec); isImport {
		// We're not really renaming the import path.
		rng.End = rng.Start
	}
	return &PrepareItem{
		Range: rng,
		Text:  obj.Name(),
	}, nil, nil
}

// checkRenamable verifies if an obj may be renamed.
func checkRenamable(obj types.Object) error {
	if v, ok := obj.(*types.Var); ok && v.Embedded() {
		return errors.New("can't rename embedded fields: rename the type directly or name the field")
	}
	if obj.Name() == "_" {
		return errors.New("can't rename \"_\"")
	}
	return nil
}

// Rename returns a map of TextEdits for each file modified when renaming a
// given identifier within a package.
func Rename(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position, newName string) (map[span.URI][]protocol.TextEdit, error) {
	ctx, done := event.Start(ctx, "source.Rename")
	defer done()

	qos, err := qualifiedObjsAtProtocolPos(ctx, s, f.URI(), pp)
	if err != nil {
		return nil, err
	}

	obj, pkg := qos[0].obj, qos[0].pkg

	if err := checkRenamable(obj); err != nil {
		return nil, err
	}
	if obj.Name() == newName {
		return nil, fmt.Errorf("old and new names are the same: %s", newName)
	}
	if !isValidIdentifier(newName) {
		return nil, fmt.Errorf("invalid identifier to rename: %q", newName)
	}
	if pkg == nil || pkg.IsIllTyped() {
		return nil, fmt.Errorf("package for %s is ill typed", f.URI())
	}
	refs, err := references(ctx, s, qos, true, false, true)
	if err != nil {
		return nil, err
	}
	r := renamer{
		ctx:          ctx,
		fset:         s.FileSet(),
		refs:         refs,
		objsToUpdate: make(map[types.Object]bool),
		from:         obj.Name(),
		to:           newName,
		packages:     make(map[*types.Package]Package),
	}

	// A renaming initiated at an interface method indicates the
	// intention to rename abstract and concrete methods as needed
	// to preserve assignability.
	for _, ref := range refs {
		if obj, ok := ref.obj.(*types.Func); ok {
			recv := obj.Type().(*types.Signature).Recv()
			if recv != nil && IsInterface(recv.Type().Underlying()) {
				r.changeMethods = true
				break
			}
		}
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
		return nil, fmt.Errorf(r.errors)
	}

	changes, err := r.update()
	if err != nil {
		return nil, err
	}
	result := make(map[span.URI][]protocol.TextEdit)
	for uri, edits := range changes {
		// These edits should really be associated with FileHandles for maximal correctness.
		// For now, this is good enough.
		fh, err := s.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		data, err := fh.Read()
		if err != nil {
			return nil, err
		}
		converter := span.NewContentConverter(uri.Filename(), data)
		m := &protocol.ColumnMapper{
			URI:       uri,
			Converter: converter,
			Content:   data,
		}
		// Sort the edits first.
		diff.SortTextEdits(edits)
		protocolEdits, err := ToProtocolEdits(m, edits)
		if err != nil {
			return nil, err
		}
		result[uri] = protocolEdits
	}
	return result, nil
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
		refSpan, err := ref.spanRange.Span()
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
		// go/parser strips out \r\n returns from the comment text, so go
		// line-by-line through the comment text to get the correct positions.
		for _, comment := range doc.List {
			if isDirective(comment.Text) {
				continue
			}
			lines := strings.Split(comment.Text, "\n")
			tok := r.fset.File(comment.Pos())
			commentLine := tok.Position(comment.Pos()).Line
			for i, line := range lines {
				lineStart := comment.Pos()
				if i > 0 {
					lineStart = tok.LineStart(commentLine + i)
				}
				for _, locs := range docRegexp.FindAllIndex([]byte(line), -1) {
					rng := span.NewRange(r.fset, lineStart+token.Pos(locs[0]), lineStart+token.Pos(locs[1]))
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
		case *ast.AssignStmt:
			// *ast.AssignStmt doesn't have an associated comment group.
			// So, we try to find a comment just before the identifier.

			// Try to find a comment group only for short variable declarations (:=).
			if decl.Tok != token.DEFINE {
				return nil
			}

			var file *ast.File
			for _, f := range pkg.GetSyntax() {
				if f.Pos() <= id.Pos() && id.Pos() <= f.End() {
					file = f
					break
				}
			}
			if file == nil {
				return nil
			}

			identLine := r.fset.Position(id.Pos()).Line
			for _, comment := range file.Comments {
				if comment.Pos() > id.Pos() {
					// Comment is after the identifier.
					continue
				}

				lastCommentLine := r.fset.Position(comment.End()).Line
				if lastCommentLine+1 == identLine {
					return comment
				}
			}
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
	_, path, _ := pathEnclosingInterval(r.fset, pkg, pkgName.Pos(), pkgName.Pos())
	if len(path) < 2 {
		return nil, fmt.Errorf("no path enclosing interval for %s", pkgName.Name())
	}
	spec, ok := path[1].(*ast.ImportSpec)
	if !ok {
		return nil, fmt.Errorf("failed to update PkgName for %s", pkgName.Name())
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
