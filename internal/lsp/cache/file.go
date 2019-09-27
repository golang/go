// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"go/token"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// viewFile extends source.File with helper methods for the view package.
type viewFile interface {
	source.File

	filename() string
	addURI(uri span.URI) int
}

// fileBase holds the common functionality for all files.
// It is intended to be embedded in the file implementations
type fileBase struct {
	uris  []span.URI
	fname string
	kind  source.FileKind

	view *view
}

func basename(filename string) string {
	return strings.ToLower(filepath.Base(filename))
}

func (f *fileBase) URI() span.URI {
	return f.uris[0]
}

func (f *fileBase) Kind() source.FileKind {
	return f.kind
}

func (f *fileBase) filename() string {
	return f.fname
}

// View returns the view associated with the file.
func (f *fileBase) View() source.View {
	return f.view
}

func (f *fileBase) FileSet() *token.FileSet {
	return f.view.Session().Cache().FileSet()
}
