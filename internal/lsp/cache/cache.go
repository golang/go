// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"crypto/sha1"
	"fmt"
	"go/token"
	"io/ioutil"
	"os"
	"reflect"
	"strconv"
	"sync/atomic"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

func New(ctx context.Context, options func(*source.Options)) *Cache {
	index := atomic.AddInt64(&cacheIndex, 1)
	c := &Cache{
		id:      strconv.FormatInt(index, 10),
		fset:    token.NewFileSet(),
		options: options,
	}
	return c
}

type Cache struct {
	id      string
	fset    *token.FileSet
	options func(*source.Options)

	store memoize.Store
}

type fileKey struct {
	uri     span.URI
	modTime time.Time
}

type fileHandle struct {
	uri span.URI
	memoize.NoCopy
	bytes []byte
	hash  string
	err   error
}

func (c *Cache) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	var modTime time.Time
	if fi, err := os.Stat(uri.Filename()); err == nil {
		modTime = fi.ModTime()
	}

	key := fileKey{
		uri:     uri,
		modTime: modTime,
	}
	h := c.store.Bind(key, func(ctx context.Context) interface{} {
		return readFile(ctx, uri, modTime)
	})
	v := h.Get(ctx)
	if v == nil {
		return nil, ctx.Err()
	}
	return v.(*fileHandle), nil
}

// ioLimit limits the number of parallel file reads per process.
var ioLimit = make(chan struct{}, 128)

func readFile(ctx context.Context, uri span.URI, origTime time.Time) *fileHandle {
	ctx, done := event.Start(ctx, "cache.getFile", tag.File.Of(uri.Filename()))
	_ = ctx
	defer done()

	ioLimit <- struct{}{}
	defer func() { <-ioLimit }()

	var modTime time.Time
	if fi, err := os.Stat(uri.Filename()); err == nil {
		modTime = fi.ModTime()
	}

	if modTime != origTime {
		return &fileHandle{err: errors.Errorf("%s: file has been modified", uri.Filename())}
	}
	data, err := ioutil.ReadFile(uri.Filename())
	if err != nil {
		return &fileHandle{err: err}
	}
	return &fileHandle{
		uri:   uri,
		bytes: data,
		hash:  hashContents(data),
	}
}

func (c *Cache) NewSession(ctx context.Context) *Session {
	index := atomic.AddInt64(&sessionIndex, 1)
	s := &Session{
		cache:    c,
		id:       strconv.FormatInt(index, 10),
		options:  source.DefaultOptions(),
		overlays: make(map[span.URI]*overlay),
	}
	event.Log(ctx, "New session", KeyCreateSession.Of(s))
	return s
}

func (c *Cache) FileSet() *token.FileSet {
	return c.fset
}

func (h *fileHandle) URI() span.URI {
	return h.uri
}

func (h *fileHandle) Kind() source.FileKind {
	return source.DetectLanguage("", h.uri.Filename())
}

func (h *fileHandle) Version() float64 {
	return 0
}

func (h *fileHandle) Identity() source.FileIdentity {
	return source.FileIdentity{
		URI:        h.uri,
		Identifier: h.hash,
		Kind:       h.Kind(),
	}
}

func (h *fileHandle) Read() ([]byte, error) {
	return h.bytes, h.err
}

func hashContents(contents []byte) string {
	// TODO: consider whether sha1 is the best choice here
	// This hash is used for internal identity detection only
	return fmt.Sprintf("%x", sha1.Sum(contents))
}

var cacheIndex, sessionIndex, viewIndex int64

func (c *Cache) ID() string                     { return c.id }
func (c *Cache) MemStats() map[reflect.Type]int { return c.store.Stats() }
