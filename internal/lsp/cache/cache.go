// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"crypto/sha1"
	"fmt"
	"go/token"
	"strconv"
	"sync/atomic"

	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

func New() source.Cache {
	index := atomic.AddInt64(&cacheIndex, 1)
	c := &cache{
		id:   strconv.FormatInt(index, 10),
		fset: token.NewFileSet(),
	}
	debug.AddCache(debugCache{c})
	return c
}

type cache struct {
	nativeFileSystem

	id   string
	fset *token.FileSet
}

func (c *cache) NewSession(log xlog.Logger) source.Session {
	index := atomic.AddInt64(&sessionIndex, 1)
	s := &session{
		cache:         c,
		id:            strconv.FormatInt(index, 10),
		log:           log,
		overlays:      make(map[span.URI]*source.FileContent),
		filesWatchMap: NewWatchMap(),
	}
	debug.AddSession(debugSession{s})
	return s
}

func (c *cache) FileSet() *token.FileSet {
	return c.fset
}

func hashContents(contents []byte) string {
	// TODO: consider whether sha1 is the best choice here
	// This hash is used for internal identity detection only
	return fmt.Sprintf("%x", sha1.Sum(contents))
}

var cacheIndex, sessionIndex, viewIndex int64

type debugCache struct{ *cache }

func (c *cache) ID() string                  { return c.id }
func (c debugCache) FileSet() *token.FileSet { return c.fset }
