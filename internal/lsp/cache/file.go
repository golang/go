// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
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

	view  *view
	fc    *source.FileContent
	token *token.File
}

func basename(filename string) string {
	return strings.ToLower(filepath.Base(filename))
}

func (f *fileBase) URI() span.URI {
	return f.uris[0]
}

func (f *fileBase) filename() string {
	return f.fname
}

// View returns the view associated with the file.
func (f *fileBase) View() source.View {
	return f.view
}

// Content returns the contents of the file, reading it from file system if needed.
func (f *fileBase) Content(ctx context.Context) *source.FileContent {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()

	f.read(ctx)
	return f.fc
}

func (f *fileBase) FileSet() *token.FileSet {
	return f.view.Session().Cache().FileSet()
}

// read is the internal part of GetContent. It assumes that the caller is
// holding the mutex of the file's view.
func (f *fileBase) read(ctx context.Context) {
	if err := ctx.Err(); err != nil {
		f.fc = &source.FileContent{Error: err}
		return
	}
	if f.fc != nil {
		if len(f.view.contentChanges) == 0 {
			return
		}

		f.view.mcache.mu.Lock()
		err := f.view.applyContentChanges(ctx)
		f.view.mcache.mu.Unlock()

		if err != nil {
			f.fc = &source.FileContent{Error: err}
			return
		}
	}
	// We don't know the content yet, so read it.
	f.fc = f.view.Session().ReadFile(f.URI())
}

// isPopulated returns true if all of the computed fields of the file are set.
func (f *goFile) isPopulated() bool {
	return f.ast != nil && f.token != nil && f.pkg != nil && f.meta != nil && f.imports != nil
}
