// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/packages"
)

// View abstracts the underlying architecture of the package using the source
// package. The view provides access to files and their contents, so the source
// package does not directly access the file system.
type View interface {
	GetFile(ctx context.Context, uri URI) (File, error)
	SetContent(ctx context.Context, uri URI, content []byte) error
	GetAnalysisCache() *AnalysisCache
	FileSet() *token.FileSet
}

// File represents a Go source file that has been type-checked. It is the input
// to most of the exported functions in this package, as it wraps up the
// building blocks for most queries. Users of the source package can abstract
// the loading of packages into their own caching systems.
type File interface {
	GetAST(ctx context.Context) *ast.File
	GetFileSet(ctx context.Context) *token.FileSet
	GetPackage(ctx context.Context) *packages.Package
	GetToken(ctx context.Context) *token.File
	GetContent(ctx context.Context) []byte
}

// Range represents a start and end position.
// Because Range is based purely on two token.Pos entries, it is not self
// contained. You need access to a token.FileSet to regain the file
// information.
type Range struct {
	Start token.Pos
	End   token.Pos
}

// TextEdit represents a change to a section of a document.
// The text within the specified range should be replaced by the supplied new text.
type TextEdit struct {
	Range   Range
	NewText string
}
