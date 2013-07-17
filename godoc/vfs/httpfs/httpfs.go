// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package httpfs implements http.FileSystem using a godoc vfs.FileSystem.
package httpfs

import (
	"fmt"
	"io"
	"net/http"
	"os"

	"code.google.com/p/go.tools/godoc/vfs"
)

func New(fs vfs.FileSystem) http.FileSystem {
	return &httpFS{fs}
}

type httpFS struct {
	fs vfs.FileSystem
}

func (h *httpFS) Open(name string) (http.File, error) {
	fi, err := h.fs.Stat(name)
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return &httpDir{h.fs, name, nil}, nil
	}
	f, err := h.fs.Open(name)
	if err != nil {
		return nil, err
	}
	return &httpFile{h.fs, f, name}, nil
}

// httpDir implements http.File for a directory in a FileSystem.
type httpDir struct {
	fs      vfs.FileSystem
	name    string
	pending []os.FileInfo
}

func (h *httpDir) Close() error               { return nil }
func (h *httpDir) Stat() (os.FileInfo, error) { return h.fs.Stat(h.name) }
func (h *httpDir) Read([]byte) (int, error) {
	return 0, fmt.Errorf("cannot Read from directory %s", h.name)
}

func (h *httpDir) Seek(offset int64, whence int) (int64, error) {
	if offset == 0 && whence == 0 {
		h.pending = nil
		return 0, nil
	}
	return 0, fmt.Errorf("unsupported Seek in directory %s", h.name)
}

func (h *httpDir) Readdir(count int) ([]os.FileInfo, error) {
	if h.pending == nil {
		d, err := h.fs.ReadDir(h.name)
		if err != nil {
			return nil, err
		}
		if d == nil {
			d = []os.FileInfo{} // not nil
		}
		h.pending = d
	}

	if len(h.pending) == 0 && count > 0 {
		return nil, io.EOF
	}
	if count <= 0 || count > len(h.pending) {
		count = len(h.pending)
	}
	d := h.pending[:count]
	h.pending = h.pending[count:]
	return d, nil
}

// httpFile implements http.File for a file (not directory) in a FileSystem.
type httpFile struct {
	fs vfs.FileSystem
	vfs.ReadSeekCloser
	name string
}

func (h *httpFile) Stat() (os.FileInfo, error) { return h.fs.Stat(h.name) }
func (h *httpFile) Readdir(int) ([]os.FileInfo, error) {
	return nil, fmt.Errorf("cannot Readdir from file %s", h.name)
}
