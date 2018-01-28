// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gatefs provides an implementation of the FileSystem
// interface that wraps another FileSystem and limits its concurrency.
package gatefs // import "golang.org/x/tools/godoc/vfs/gatefs"

import (
	"fmt"
	"os"

	"golang.org/x/tools/godoc/vfs"
)

// New returns a new FileSystem that delegates to fs.
// If gateCh is non-nil and buffered, it's used as a gate
// to limit concurrency on calls to fs.
func New(fs vfs.FileSystem, gateCh chan bool) vfs.FileSystem {
	if cap(gateCh) == 0 {
		return fs
	}
	return gatefs{fs, gate(gateCh)}
}

type gate chan bool

func (g gate) enter() { g <- true }
func (g gate) leave() { <-g }

type gatefs struct {
	fs vfs.FileSystem
	gate
}

func (fs gatefs) String() string {
	return fmt.Sprintf("gated(%s, %d)", fs.fs.String(), cap(fs.gate))
}

func (fs gatefs) RootType(path string) vfs.RootType {
	return fs.fs.RootType(path)
}

func (fs gatefs) Open(p string) (vfs.ReadSeekCloser, error) {
	fs.enter()
	defer fs.leave()
	rsc, err := fs.fs.Open(p)
	if err != nil {
		return nil, err
	}
	return gatef{rsc, fs.gate}, nil
}

func (fs gatefs) Lstat(p string) (os.FileInfo, error) {
	fs.enter()
	defer fs.leave()
	return fs.fs.Lstat(p)
}

func (fs gatefs) Stat(p string) (os.FileInfo, error) {
	fs.enter()
	defer fs.leave()
	return fs.fs.Stat(p)
}

func (fs gatefs) ReadDir(p string) ([]os.FileInfo, error) {
	fs.enter()
	defer fs.leave()
	return fs.fs.ReadDir(p)
}

type gatef struct {
	rsc vfs.ReadSeekCloser
	gate
}

func (f gatef) Read(p []byte) (n int, err error) {
	f.enter()
	defer f.leave()
	return f.rsc.Read(p)
}

func (f gatef) Seek(offset int64, whence int) (ret int64, err error) {
	f.enter()
	defer f.leave()
	return f.rsc.Seek(offset, whence)
}

func (f gatef) Close() error {
	f.enter()
	defer f.leave()
	return f.rsc.Close()
}
