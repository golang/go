// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vfs defines virtual filesystem types.
package vfs

import (
	"io"
)

// Opener is a minimal virtual filesystem that can only open
// regular files.
type Opener interface {
	Open(name string) (ReadSeekCloser, error)
}

// A ReadSeekCloser can Read, Seek, and Close.
type ReadSeekCloser interface {
	io.Reader
	io.Seeker
	io.Closer
}
