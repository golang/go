// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"crypto/sha1"
	"fmt"
	"go/token"
	"strconv"
	"sync/atomic"

	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
)

func New() source.Cache {
	index := atomic.AddInt64(&cacheIndex, 1)
	c := &cache{
		fs:   &nativeFileSystem{},
		id:   strconv.FormatInt(index, 10),
		fset: token.NewFileSet(),
	}
	debug.AddCache(debugCache{c})
	return c
}

type cache struct {
	fs   source.FileSystem
	id   string
	fset *token.FileSet

	store memoize.Store
}

type fileKey struct {
	identity source.FileIdentity
}

type fileHandle struct {
	cache      *cache
	underlying source.FileHandle
	handle     *memoize.Handle
}

type fileData struct {
	memoize.NoCopy
	bytes []byte
	hash  string
	err   error
}

func (c *cache) GetFile(uri span.URI) source.FileHandle {
	underlying := c.fs.GetFile(uri)
	key := fileKey{
		identity: underlying.Identity(),
	}
	h := c.store.Bind(key, func(ctx context.Context) interface{} {
		data := &fileData{}
		data.bytes, data.hash, data.err = underlying.Read(ctx)
		return data
	})
	return &fileHandle{
		cache:      c,
		underlying: underlying,
		handle:     h,
	}
}

func (c *cache) NewSession(log xlog.Logger) source.Session {
	index := atomic.AddInt64(&sessionIndex, 1)
	s := &session{
		cache:         c,
		id:            strconv.FormatInt(index, 10),
		log:           log,
		overlays:      make(map[span.URI]*overlay),
		filesWatchMap: NewWatchMap(),
	}
	debug.AddSession(debugSession{s})
	return s
}

func (c *cache) FileSet() *token.FileSet {
	return c.fset
}

func (h *fileHandle) FileSystem() source.FileSystem {
	return h.cache
}

func (h *fileHandle) Identity() source.FileIdentity {
	return h.underlying.Identity()
}

func (h *fileHandle) Kind() source.FileKind {
	return h.underlying.Kind()
}

func (h *fileHandle) Read(ctx context.Context) ([]byte, string, error) {
	v := h.handle.Get(ctx)
	if v == nil {
		return nil, "", ctx.Err()
	}
	data := v.(*fileData)
	return data.bytes, data.hash, data.err
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
