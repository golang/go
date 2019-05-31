// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"io/ioutil"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// nativeFileSystem implements FileSystem reading from the normal os file system.
type nativeFileSystem struct{}

// nativeFileHandle implements FileHandle for nativeFileSystem
type nativeFileHandle struct {
	fs       *nativeFileSystem
	identity source.FileIdentity
}

func (fs *nativeFileSystem) GetFile(uri span.URI) source.FileHandle {
	return &nativeFileHandle{
		fs: fs,
		identity: source.FileIdentity{
			URI: uri,
			// TODO: decide what the version string is for a native file system
			// could be the mtime?
			Version: "",
		},
	}
}

func (h *nativeFileHandle) FileSystem() source.FileSystem {
	return h.fs
}

func (h *nativeFileHandle) Identity() source.FileIdentity {
	return h.identity
}

func (h *nativeFileHandle) Read(ctx context.Context) *source.FileContent {
	r := &source.FileContent{}
	filename, err := h.identity.URI.Filename()
	if err != nil {
		r.Error = err
		return r
	}
	r.Data, r.Error = ioutil.ReadFile(filename)
	if r.Error != nil {
		r.Hash = hashContents(r.Data)
	}
	return r
}
