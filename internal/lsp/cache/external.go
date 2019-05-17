// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"io/ioutil"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// nativeFileSystem implements FileSystem reading from the normal os file system.
type nativeFileSystem struct{}

func (nativeFileSystem) ReadFile(uri span.URI) *source.FileContent {
	r := &source.FileContent{URI: uri}
	filename, err := uri.Filename()
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
