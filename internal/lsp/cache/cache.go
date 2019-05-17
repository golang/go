// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"crypto/sha1"
	"fmt"
	"go/token"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

func New() source.Cache {
	return &cache{
		fset: token.NewFileSet(),
	}
}

type cache struct {
	nativeFileSystem

	fset *token.FileSet
}

func (c *cache) NewSession(log xlog.Logger) source.Session {
	return &session{
		cache:         c,
		log:           log,
		overlays:      make(map[span.URI]*source.FileContent),
		filesWatchMap: NewWatchMap(),
	}
}

func (c *cache) FileSet() *token.FileSet {
	return c.fset
}

func hashContents(contents []byte) string {
	// TODO: consider whether sha1 is the best choice here
	// This hash is used for internal identity detection only
	return fmt.Sprintf("%x", sha1.Sum(contents))
}
