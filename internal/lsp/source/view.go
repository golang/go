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

// Cache abstracts the core logic of dealing with the environment from the
// higher level logic that processes the information to produce results.
// The cache provides access to files and their contents, so the source
// package does not directly access the file system.
// A single cache is intended to be process wide, and is the primary point of
// sharing between all consumers.
// A cache may have many active sessions at any given time.
type Cache interface {
	// NewSession creates a new Session manager and returns it.
	NewSession(log xlog.Logger) Session
}

// Session represents a single connection from a client.
// This is the level at which things like open files are maintained on behalf
// of the client.
// A session may have many active views at any given time.
type Session interface {
	// NewView creates a new View and returns it.
	NewView(name string, folder span.URI, config *packages.Config) View

	// Cache returns the cache that created this session.
	Cache() Cache

	// Returns the logger in use for this session.
	Logger() xlog.Logger

	View(name string) View
	ViewOf(uri span.URI) View
	Views() []View

	Shutdown(ctx context.Context)
}

// View represents a single workspace.
// This is the level at which we maintain configuration like working directory
// and build tags.
type View interface {
	// Session returns the session that created this view.
	Session() Session
	Name() string
	Folder() span.URI
	FileSet() *token.FileSet
	BuiltinPackage() *ast.Package
	GetFile(ctx context.Context, uri span.URI) (File, error)
	SetContent(ctx context.Context, uri span.URI, content []byte) error
	BackgroundContext() context.Context
	Config() packages.Config
	SetEnv([]string)
	Shutdown(ctx context.Context)
	Ignore(span.URI) bool
}

// File represents a source file of any type.
type File interface {
	URI() span.URI
	View() View
	GetContent(ctx context.Context) []byte
	GetFileSet(ctx context.Context) *token.FileSet
	GetToken(ctx context.Context) *token.File
}

// GoFile represents a Go source file that has been type-checked.
type GoFile interface {
	File
	GetAST(ctx context.Context) *ast.File
	GetPackage(ctx context.Context) Package

	// GetActiveReverseDeps returns the active files belonging to the reverse
	// dependencies of this file's package.
	GetActiveReverseDeps(ctx context.Context) []GoFile
}

// Package represents a Go package that has been type-checked. It maintains
// only the relevant fields of a *go/packages.Package.
type Package interface {
	PkgPath() string
	GetFilenames() []string
	GetSyntax() []*ast.File
	GetErrors() []packages.Error
	GetTypes() *types.Package
	GetTypesInfo() *types.Info
	GetTypesSizes() types.Sizes
	IsIllTyped() bool
	GetActionGraph(ctx context.Context, a *analysis.Analyzer) (*Action, error)
	GetImport(pkgPath string) Package
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
