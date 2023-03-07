// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"container/heap"
	"context"
	"go/token"
	"runtime"
	"sort"
	"sync"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
)

// This file contains an implementation of a bounded-size parse cache, that
// offsets the base token.Pos value of each cached file so that they may be
// later described by a single dedicated FileSet.
//
// This is achieved by tracking a monotonic offset in the token.Pos space, that
// is incremented before parsing allow room for the resulting parsed file.

// Keep 200 recently parsed files, based on the following rationale:
//   - One of the most important benefits of caching is avoiding re-parsing
//     everything in a package when working on a single file. No packages in
//     Kubernetes have > 200 files (only one has > 100).
//   - Experience has shown that ~1000 parsed files can use noticeable space.
//     200 feels like a sweet spot between limiting cache size and optimizing
//     cache hits for low-latency operations.
const parseCacheMaxFiles = 200

// parsePadding is additional padding allocated between entries in the parse
// cache to allow for increases in length (such as appending missing braces)
// caused by fixAST.
//
// This is used to mitigate a chicken and egg problem: we must know the base
// offset of the file we're about to parse, before we start parsing, and yet
// src fixups may affect the actual size of the parsed content (and therefore
// the offsets of subsequent files).
//
// When we encounter a file that no longer fits in its allocated space in the
// fileset, we have no choice but to re-parse it. Leaving a generous padding
// reduces the likelihood of this "slow path".
//
// This value is mutable for testing, so that we can exercise the slow path.
var parsePadding = 1000 // mutable for testing

// A parseCache holds a bounded number of recently accessed parsed Go files. As
// new files are stored, older files may be evicted from the cache.
//
// The parseCache.parseFiles method exposes a batch API for parsing (and
// caching) multiple files. This is necessary for type-checking, where files
// must be parsed in a common fileset.
type parseCache struct {
	mu         sync.Mutex
	m          map[parseKey]*parseCacheEntry
	lru        queue     // min-atime priority queue of *parseCacheEntry
	clock      uint64    // clock time, incremented when the cache is updated
	nextOffset token.Pos // token.Pos offset for the next parsed file
}

// parseKey uniquely identifies a parsed Go file.
type parseKey struct {
	file source.FileIdentity
	mode source.ParseMode
}

type parseCacheEntry struct {
	key      parseKey
	promise  *memoize.Promise // memoize.Promise[*source.ParsedGoFile]
	atime    uint64           // clock time of last access
	lruIndex int
}

// startParse prepares a parsing pass, using the following steps:
//   - search for cache hits
//   - create new promises for cache misses
//   - store as many new promises in the cache as space will allow
//
// The resulting slice has an entry for every given file handle, though some
// entries may be nil if there was an error reading the file (in which case the
// resulting error will be non-nil).
func (c *parseCache) startParse(mode source.ParseMode, fhs ...source.FileHandle) ([]*memoize.Promise, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Any parsing pass increments the clock, as we'll update access times.
	// (technically, if fhs is empty this isn't necessary, but that's a degenerate case).
	//
	// All entries parsed from a single call get the same access time.
	c.clock++

	// Read file data and collect cacheable files.
	var (
		data           = make([][]byte, len(fhs)) // file content for each readable file
		promises       = make([]*memoize.Promise, len(fhs))
		firstReadError error // first error from fh.Read, or nil
	)
	for i, fh := range fhs {
		content, err := fh.Content()
		if err != nil {
			if firstReadError == nil {
				firstReadError = err
			}
			continue
		}
		data[i] = content

		key := parseKey{
			file: fh.FileIdentity(),
			mode: mode,
		}

		// Check for a cache hit.
		if e, ok := c.m[key]; ok {
			e.atime = c.clock
			heap.Fix(&c.lru, e.lruIndex)
			promises[i] = e.promise
			continue
		}

		// ...otherwise, create a new promise to parse with a non-overlapping offset
		fset := token.NewFileSet()
		if c.nextOffset > 0 {
			// Add a dummy file so that this parsed file does not overlap with others.
			fset.AddFile("", 1, int(c.nextOffset))
		}
		c.nextOffset += token.Pos(len(content) + parsePadding + 1) // leave room for src fixes
		fh := fh
		promise := memoize.NewPromise(string(fh.URI()), func(ctx context.Context, _ interface{}) interface{} {
			return parseGoSrc(ctx, fset, fh.URI(), content, mode)
		})
		promises[i] = promise

		var e *parseCacheEntry
		if len(c.lru) < parseCacheMaxFiles {
			// add new entry
			e = new(parseCacheEntry)
			if c.m == nil {
				c.m = make(map[parseKey]*parseCacheEntry)
			}
		} else {
			// evict oldest entry
			e = heap.Pop(&c.lru).(*parseCacheEntry)
			delete(c.m, e.key)
		}
		e.key = key
		e.promise = promise
		e.atime = c.clock
		c.m[e.key] = e
		heap.Push(&c.lru, e)
	}

	if len(c.m) != len(c.lru) {
		panic("map and LRU are inconsistent")
	}

	return promises, firstReadError
}

// parseFiles returns a ParsedGoFile for the given file handles in the
// requested parse mode.
//
// If parseFiles returns an error, it still returns a slice,
// but with a nil entry for each file that could not be parsed.
//
// The second result is a FileSet describing all resulting parsed files.
//
// For parsed files that already exists in the cache, access time will be
// updated. For others, parseFiles will parse and store as many results in the
// cache as space allows.
func (c *parseCache) parseFiles(ctx context.Context, mode source.ParseMode, fhs ...source.FileHandle) ([]*source.ParsedGoFile, *token.FileSet, error) {
	promises, firstReadError := c.startParse(mode, fhs...)

	// Await all parsing.
	var g errgroup.Group
	g.SetLimit(runtime.GOMAXPROCS(-1)) // parsing is CPU-bound.
	pgfs := make([]*source.ParsedGoFile, len(fhs))
	for i, promise := range promises {
		if promise == nil {
			continue
		}
		i := i
		promise := promise
		g.Go(func() error {
			result, err := promise.Get(ctx, nil)
			if err != nil {
				return err
			}
			pgfs[i] = result.(*source.ParsedGoFile)
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, nil, err
	}

	// Construct a token.FileSet mapping all parsed files, and update their
	// Tok to the corresponding file in the new fileset.
	//
	// In the unlikely event that a parsed file no longer fits in its allocated
	// space in the FileSet range, it will need to be re-parsed.

	var tokenFiles []*token.File
	fileIndex := make(map[*token.File]int) // to look up original indexes after sorting
	for i, pgf := range pgfs {
		if pgf == nil {
			continue
		}
		fileIndex[pgf.Tok] = i
		tokenFiles = append(tokenFiles, pgf.Tok)
	}

	sort.Slice(tokenFiles, func(i, j int) bool {
		return tokenFiles[i].Base() < tokenFiles[j].Base()
	})

	var needReparse []int // files requiring reparsing
	out := tokenFiles[:0]
	for i, f := range tokenFiles {
		if i < len(tokenFiles)-1 && f.Base()+f.Size() >= tokenFiles[i+1].Base() {
			if f != tokenFiles[i+1] { // no need to re-parse duplicates
				needReparse = append(needReparse, fileIndex[f])
			}
		} else {
			out = append(out, f)
		}
	}
	fset := source.FileSetFor(out...)

	// Re-parse any remaining files using the stitched fileSet.
	for _, i := range needReparse {
		// Start from scratch, rather than using ParsedGoFile.Src, so that source
		// fixing operates exactly the same (note that fixing stops after a limited
		// number of tries).
		fh := fhs[i]
		content, err := fh.Content()
		if err != nil {
			if firstReadError == nil {
				firstReadError = err
			}
			continue
		}
		pgfs[i] = parseGoSrc(ctx, fset, fh.URI(), content, mode)
	}

	// Ensure each PGF refers to a token.File from the new FileSet.
	for i, pgf := range pgfs {
		if pgf == nil {
			continue
		}
		newTok := fset.File(token.Pos(pgf.Tok.Base()))
		if newTok == nil {
			panic("internal error: missing tok for " + pgf.URI)
		}
		if newTok.Base() != pgf.Tok.Base() || newTok.Size() != pgf.Tok.Size() {
			panic("internal error: mismatching token.File in synthetic FileSet")
		}
		pgf2 := *pgf
		pgf2.Tok = newTok
		pgfs[i] = &pgf2
	}

	return pgfs, fset, firstReadError
}

// -- priority queue boilerplate --

// queue is a min-atime prority queue of cache entries.
type queue []*parseCacheEntry

func (q queue) Len() int { return len(q) }

func (q queue) Less(i, j int) bool { return q[i].atime < q[j].atime }

func (q queue) Swap(i, j int) {
	q[i], q[j] = q[j], q[i]
	q[i].lruIndex = i
	q[j].lruIndex = j
}

func (q *queue) Push(x interface{}) {
	e := x.(*parseCacheEntry)
	e.lruIndex = len(*q)
	*q = append(*q, e)
}

func (q *queue) Pop() interface{} {
	last := len(*q) - 1
	e := (*q)[last]
	(*q)[last] = nil // aid GC
	*q = (*q)[:last]
	return e
}
