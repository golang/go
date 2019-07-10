// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"io/ioutil"
	"os"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
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
	version := "DOES NOT EXIST"
	if fi, err := os.Stat(uri.Filename()); err == nil {
		version = fi.ModTime().String()
	}
	return &nativeFileHandle{
		fs: fs,
		identity: source.FileIdentity{
			URI:     uri,
			Version: version,
		},
	}
}

func (h *nativeFileHandle) FileSystem() source.FileSystem {
	return h.fs
}

func (h *nativeFileHandle) Identity() source.FileIdentity {
	return h.identity
}

func (h *nativeFileHandle) Kind() source.FileKind {
	// TODO: How should we determine the file kind?
	return source.Go
}

func (h *nativeFileHandle) Read(ctx context.Context) ([]byte, string, error) {
	ctx, done := trace.StartSpan(ctx, "cache.nativeFileHandle.Read", telemetry.File.Of(h.identity.URI.Filename()))
	defer done()
	//TODO: this should fail if the version is not the same as the handle
	data, err := ioutil.ReadFile(h.identity.URI.Filename())
	if err != nil {
		return nil, "", err
	}
	return data, hashContents(data), nil
}
