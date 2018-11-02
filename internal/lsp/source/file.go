// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"go/ast"
	"go/token"
	"io/ioutil"

	"golang.org/x/tools/internal/lsp/protocol"
)

// File holds all the information we know about a file.
type File struct {
	URI     protocol.DocumentURI
	view    *View
	active  bool
	content []byte
	ast     *ast.File
	token   *token.File
}

// SetContent sets the overlay contents for a file.
// Setting it to nil will revert it to the on disk contents, and remove it
// from the active set.
func (f *File) SetContent(content []byte) {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	f.content = content
	// the ast and token fields are invalid
	f.ast = nil
	f.token = nil
	// and we might need to update the overlay
	switch {
	case f.active && content == nil:
		// we were active, and want to forget the content
		f.active = false
		if filename, err := FromURI(f.URI); err == nil {
			delete(f.view.Config.Overlay, filename)
		}
		f.content = nil
	case content != nil:
		// an active overlay, update the map
		f.active = true
		if filename, err := FromURI(f.URI); err == nil {
			f.view.Config.Overlay[filename] = f.content
		}
	}
}

// Read returns the contents of the file, reading it from file system if needed.
func (f *File) Read() ([]byte, error) {
	f.view.mu.Lock()
	defer f.view.mu.Unlock()
	return f.read()
}

// read is the internal part of Read that presumes the lock is already held
func (f *File) read() ([]byte, error) {
	if f.content != nil {
		return f.content, nil
	}
	// we don't know the content yet, so read it
	filename, err := FromURI(f.URI)
	if err != nil {
		return nil, err
	}
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	f.content = content
	return f.content, nil
}
