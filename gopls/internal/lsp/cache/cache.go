// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"reflect"
	"strconv"
	"sync/atomic"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/robustio"
)

// New Creates a new cache for gopls operation results, using the given file
// set, shared store, and session options.
//
// Both the fset and store may be nil, but if store is non-nil so must be fset
// (and they must always be used together), otherwise it may be possible to get
// cached data referencing token.Pos values not mapped by the FileSet.
func New(store *memoize.Store) *Cache {
	index := atomic.AddInt64(&cacheIndex, 1)

	if store == nil {
		store = &memoize.Store{}
	}

	c := &Cache{
		id:         strconv.FormatInt(index, 10),
		store:      store,
		memoizedFS: &memoizedFS{filesByID: map[robustio.FileID][]*DiskFile{}},
	}
	return c
}

// A Cache holds caching stores that are bundled together for consistency.
//
// TODO(rfindley): once fset and store need not be bundled together, the Cache
// type can be eliminated.
type Cache struct {
	id string

	store *memoize.Store

	*memoizedFS // implements source.FileSource
}

// NewSession creates a new gopls session with the given cache and options overrides.
//
// The provided optionsOverrides may be nil.
//
// TODO(rfindley): move this to session.go.
func NewSession(ctx context.Context, c *Cache) *Session {
	index := atomic.AddInt64(&sessionIndex, 1)
	s := &Session{
		id:          strconv.FormatInt(index, 10),
		cache:       c,
		gocmdRunner: &gocommand.Runner{},
		overlayFS:   newOverlayFS(c),
		parseCache:  newParseCache(1 * time.Minute), // keep recently parsed files for a minute, to optimize typing CPU
	}
	event.Log(ctx, "New session", KeyCreateSession.Of(s))
	return s
}

var cacheIndex, sessionIndex, viewIndex int64

func (c *Cache) ID() string                     { return c.id }
func (c *Cache) MemStats() map[reflect.Type]int { return c.store.Stats() }

// FileStats returns information about the set of files stored in the cache.
// It is intended for debugging only.
func (c *Cache) FileStats() (files, largest, errs int) {
	return c.fileStats()
}
