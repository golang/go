// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package govulncheck

import (
	"sync"
	"time"

	vulnc "golang.org/x/vuln/client"
	"golang.org/x/vuln/osv"
)

// inMemoryCache is an implementation of the [client.Cache] interface
// that "decorates" another instance of that interface to provide
// an additional layer of (memory-based) caching.
type inMemoryCache struct {
	mu         sync.Mutex
	underlying vulnc.Cache
	db         map[string]*db
}

var _ vulnc.Cache = &inMemoryCache{}

type db struct {
	retrieved time.Time
	index     vulnc.DBIndex
	entry     map[string][]*osv.Entry
}

// NewInMemoryCache returns a new memory-based cache that decorates
// the provided cache (file-based, perhaps).
func NewInMemoryCache(underlying vulnc.Cache) *inMemoryCache {
	return &inMemoryCache{
		underlying: underlying,
		db:         make(map[string]*db),
	}
}

func (c *inMemoryCache) lookupDBLocked(dbName string) *db {
	cached := c.db[dbName]
	if cached == nil {
		cached = &db{entry: make(map[string][]*osv.Entry)}
		c.db[dbName] = cached
	}
	return cached
}

// ReadIndex returns the index for dbName from the cache, or returns zero values
// if it is not present.
func (c *inMemoryCache) ReadIndex(dbName string) (vulnc.DBIndex, time.Time, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	cached := c.lookupDBLocked(dbName)

	if cached.retrieved.IsZero() {
		// First time ReadIndex is called.
		index, retrieved, err := c.underlying.ReadIndex(dbName)
		if err != nil {
			return index, retrieved, err
		}
		cached.index, cached.retrieved = index, retrieved
	}
	return cached.index, cached.retrieved, nil
}

// WriteIndex puts the index and retrieved time into the cache.
func (c *inMemoryCache) WriteIndex(dbName string, index vulnc.DBIndex, retrieved time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	cached := c.lookupDBLocked(dbName)
	cached.index, cached.retrieved = index, retrieved
	// TODO(hyangah): shouldn't we invalidate all cached entries?
	return c.underlying.WriteIndex(dbName, index, retrieved)
}

// ReadEntries returns the vulndb entries for path from the cache.
func (c *inMemoryCache) ReadEntries(dbName, path string) ([]*osv.Entry, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	cached := c.lookupDBLocked(dbName)
	entries, ok := cached.entry[path]
	if !ok {
		// cache miss
		entries, err := c.underlying.ReadEntries(dbName, path)
		if err != nil {
			return entries, err
		}
		cached.entry[path] = entries
	}
	return entries, nil
}

// WriteEntries puts the entries for path into the cache.
func (c *inMemoryCache) WriteEntries(dbName, path string, entries []*osv.Entry) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	cached := c.lookupDBLocked(dbName)
	cached.entry[path] = entries
	return c.underlying.WriteEntries(dbName, path, entries)
}
