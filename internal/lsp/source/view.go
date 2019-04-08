// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

// View abstracts the underlying architecture of the package using the source
// package. The view provides access to files and their contents, so the source
// package does not directly access the file system.
type View interface {
	Logger() xlog.Logger
	FileSet() *token.FileSet
	GetFile(ctx context.Context, uri span.URI) (File, error)
	SetContent(ctx context.Context, uri span.URI, content []byte) error
}

// File represents a Go source file that has been type-checked. It is the input
// to most of the exported functions in this package, as it wraps up the
// building blocks for most queries. Users of the source package can abstract
// the loading of packages into their own caching systems.
type File interface {
	URI() span.URI
	GetAST(ctx context.Context) *ast.File
	GetFileSet(ctx context.Context) *token.FileSet
	GetPackage(ctx context.Context) Package
	GetToken(ctx context.Context) *token.File
	GetContent(ctx context.Context) []byte
}

// Package represents a Go package that has been type-checked. It maintains
// only the relevant fields of a *go/packages.Package.
type Package interface {
	GetFilenames() []string
	GetSyntax() []*ast.File
	GetErrors() []packages.Error
	GetTypes() *types.Package
	GetTypesInfo() *types.Info
	GetTypesSizes() types.Sizes
	IsIllTyped() bool
	GetActionGraph(ctx context.Context, a *analysis.Analyzer) (*Action, error)
}

// TextEdit represents a change to a section of a document.
// The text within the specified span should be replaced by the supplied new text.
type TextEdit struct {
	Span    span.Span
	NewText string
}

// DiffToEdits converts from a sequence of diff operations to a sequence of
// source.TextEdit
func DiffToEdits(uri span.URI, ops []*diff.Op) []TextEdit {
	edits := make([]TextEdit, 0, len(ops))
	for _, op := range ops {
		s := span.New(uri, span.NewPoint(op.I1+1, 1, 0), span.NewPoint(op.I2+1, 1, 0))
		switch op.Kind {
		case diff.Delete:
			// Delete: unformatted[i1:i2] is deleted.
			edits = append(edits, TextEdit{Span: s})
		case diff.Insert:
			// Insert: formatted[j1:j2] is inserted at unformatted[i1:i1].
			if content := strings.Join(op.Content, ""); content != "" {
				edits = append(edits, TextEdit{Span: s, NewText: content})
			}
		}
	}
	return edits
}

func EditsToDiff(edits []TextEdit) []*diff.Op {
	iToJ := 0
	ops := make([]*diff.Op, len(edits))
	for i, edit := range edits {
		i1 := edit.Span.Start().Line() - 1
		i2 := edit.Span.End().Line() - 1
		kind := diff.Insert
		if edit.NewText == "" {
			kind = diff.Delete
		}
		ops[i] = &diff.Op{
			Kind:    kind,
			Content: diff.SplitLines(edit.NewText),
			I1:      i1,
			I2:      i2,
			J1:      i1 + iToJ,
		}
		if kind == diff.Insert {
			iToJ += len(ops[i].Content)
		} else {
			iToJ -= i2 - i1
		}
	}
	return ops
}
