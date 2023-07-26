// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"container/heap"
	"context"
	"fmt"
	"go/parser"
	"go/token"
	"math/bits"
	"runtime"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/tokeninternal"
)

// This file contains an implementation of an LRU parse cache, that offsets the
// base token.Pos value of each cached file so that they may be later described
// by a single dedicated FileSet.
//
// This is achieved by tracking a monotonic offset in the token.Pos space, that
// is incremented before parsing allow room for the resulting parsed file.

// reservedForParsing defines the room in the token.Pos space reserved for
// cached parsed files.
//
// Files parsed through the parseCache are guaranteed not to have overlapping
// spans: the parseCache tracks a monotonic base for newly parsed files.
//
// By offsetting the initial base of a FileSet, we can allow other operations
// accepting the FileSet (such as the gcimporter) to add new files using the
// normal FileSet APIs without overlapping with cached parsed files.
//
// Note that 1<<60 represents an exabyte of parsed data, more than any gopls
// process can ever parse.
//
// On 32-bit systems we don't cache parse results (see parseFiles).
const reservedForParsing = 1 << (bits.UintSize - 4)

// fileSetWithBase returns a new token.FileSet with Base() equal to the
// requested base.
//
// If base < 1, fileSetWithBase panics.
// (1 is the smallest permitted FileSet base).
func fileSetWithBase(base int) *token.FileSet {
	fset := token.NewFileSet()
	if base > 1 {
		// Add a dummy file to set the base of fset. We won't ever use the
		// resulting FileSet, so it doesn't matter how we achieve this.
		//
		// FileSets leave a 1-byte padding between files, so we set the base by
		// adding a zero-length file at base-1.
		fset.AddFile("", base-1, 0)
	}
	if fset.Base() != base {
		panic("unexpected FileSet.Base")
	}
	return fset
}

const (
	// Always keep 100 recent files, independent of their wall-clock age, to
	// optimize the case where the user resumes editing after a delay.
	parseCacheMinFiles = 100
)

// parsePadding is additional padding allocated to allow for increases in
// length (such as appending missing braces) caused by fixAST.
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

// A parseCache holds recently accessed parsed Go files. After new files are
// stored, older files may be evicted from the cache via garbage collection.
//
// The parseCache.parseFiles method exposes a batch API for parsing (and
// caching) multiple files. This is necessary for type-checking, where files
// must be parsed in a common fileset.
type parseCache struct {
	expireAfter time.Duration // interval at which to collect expired cache entries
	done        chan struct{} // closed when GC is stopped

	mu       sync.Mutex
	m        map[parseKey]*parseCacheEntry
	lru      queue  // min-atime priority queue of *parseCacheEntry
	clock    uint64 // clock time, incremented when the cache is updated
	nextBase int    // base offset for the next parsed file
}

// newParseCache creates a new parse cache and starts a goroutine to garbage
// collect entries whose age is at least expireAfter.
//
// Callers must call parseCache.stop when the parse cache is no longer in use.
func newParseCache(expireAfter time.Duration) *parseCache {
	c := &parseCache{
		expireAfter: expireAfter,
		m:           make(map[parseKey]*parseCacheEntry),
		done:        make(chan struct{}),
	}
	go c.gc()
	return c
}

// stop causes the GC goroutine to exit.
func (c *parseCache) stop() {
	close(c.done)
}

// parseKey uniquely identifies a parsed Go file.
type parseKey struct {
	uri             span.URI
	mode            parser.Mode
	purgeFuncBodies bool
}

type parseCacheEntry struct {
	key      parseKey
	hash     source.Hash
	promise  *memoize.Promise // memoize.Promise[*source.ParsedGoFile]
	atime    uint64           // clock time of last access, for use in LRU sorting
	walltime time.Time        // actual time of last access, for use in time-based eviction; too coarse for LRU on some systems
	lruIndex int              // owned by the queue implementation
}

// startParse prepares a parsing pass, creating new promises in the cache for
// any cache misses.
//
// The resulting slice has an entry for every given file handle, though some
// entries may be nil if there was an error reading the file (in which case the
// resulting error will be non-nil).
func (c *parseCache) startParse(mode parser.Mode, purgeFuncBodies bool, fhs ...source.FileHandle) ([]*memoize.Promise, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Any parsing pass increments the clock, as we'll update access times.
	// (technically, if fhs is empty this isn't necessary, but that's a degenerate case).
	//
	// All entries parsed from a single call get the same access time.
	c.clock++
	walltime := time.Now()

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
			uri:             fh.URI(),
			mode:            mode,
			purgeFuncBodies: purgeFuncBodies,
		}

		if e, ok := c.m[key]; ok {
			if e.hash == fh.FileIdentity().Hash { // cache hit
				e.atime = c.clock
				e.walltime = walltime
				heap.Fix(&c.lru, e.lruIndex)
				promises[i] = e.promise
				continue
			} else {
				// A cache hit, for a different version. Delete it.
				delete(c.m, e.key)
				heap.Remove(&c.lru, e.lruIndex)
			}
		}

		uri := fh.URI()
		promise := memoize.NewPromise("parseCache.parse", func(ctx context.Context, _ interface{}) interface{} {
			// Allocate 2*len(content)+parsePadding to allow for re-parsing once
			// inside of parseGoSrc without exceeding the allocated space.
			base, nextBase := c.allocateSpace(2*len(content) + parsePadding)

			pgf, fixes1 := ParseGoSrc(ctx, fileSetWithBase(base), uri, content, mode, purgeFuncBodies)
			file := pgf.Tok
			if file.Base()+file.Size()+1 > nextBase {
				// The parsed file exceeds its allocated space, likely due to multiple
				// passes of src fixing. In this case, we have no choice but to re-do
				// the operation with the correct size.
				//
				// Even though the final successful parse requires only file.Size()
				// bytes of Pos space, we need to accommodate all the missteps to get
				// there, as parseGoSrc will repeat them.
				actual := file.Base() + file.Size() - base // actual size consumed, after re-parsing
				base2, nextBase2 := c.allocateSpace(actual)
				pgf2, fixes2 := ParseGoSrc(ctx, fileSetWithBase(base2), uri, content, mode, purgeFuncBodies)

				// In golang/go#59097 we observed that this panic condition was hit.
				// One bug was found and fixed, but record more information here in
				// case there is still a bug here.
				if end := pgf2.Tok.Base() + pgf2.Tok.Size(); end != nextBase2-1 {
					var errBuf bytes.Buffer
					fmt.Fprintf(&errBuf, "internal error: non-deterministic parsing result:\n")
					fmt.Fprintf(&errBuf, "\t%q (%d-%d) does not span %d-%d\n", uri, pgf2.Tok.Base(), base2, end, nextBase2-1)
					fmt.Fprintf(&errBuf, "\tfirst %q (%d-%d)\n", pgf.URI, pgf.Tok.Base(), pgf.Tok.Base()+pgf.Tok.Size())
					fmt.Fprintf(&errBuf, "\tfirst space: (%d-%d), second space: (%d-%d)\n", base, nextBase, base2, nextBase2)
					fmt.Fprintf(&errBuf, "\tfirst mode: %v, second mode: %v", pgf.Mode, pgf2.Mode)
					fmt.Fprintf(&errBuf, "\tfirst err: %v, second err: %v", pgf.ParseErr, pgf2.ParseErr)
					fmt.Fprintf(&errBuf, "\tfirst fixes: %v, second fixes: %v", fixes1, fixes2)
					panic(errBuf.String())
				}
				pgf = pgf2
			}
			return pgf
		})
		promises[i] = promise

		// add new entry; entries are gc'ed asynchronously
		e := &parseCacheEntry{
			key:      key,
			hash:     fh.FileIdentity().Hash,
			promise:  promise,
			atime:    c.clock,
			walltime: walltime,
		}
		c.m[e.key] = e
		heap.Push(&c.lru, e)
	}

	if len(c.m) != len(c.lru) {
		panic("map and LRU are inconsistent")
	}

	return promises, firstReadError
}

func (c *parseCache) gc() {
	const period = 10 * time.Second // gc period
	timer := time.NewTicker(period)
	defer timer.Stop()

	for {
		select {
		case <-c.done:
			return
		case <-timer.C:
		}

		c.gcOnce()
	}
}

func (c *parseCache) gcOnce() {
	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()

	for len(c.m) > parseCacheMinFiles {
		e := heap.Pop(&c.lru).(*parseCacheEntry)
		if now.Sub(e.walltime) >= c.expireAfter {
			delete(c.m, e.key)
		} else {
			heap.Push(&c.lru, e)
			break
		}
	}
}

// allocateSpace reserves the next n bytes of token.Pos space in the
// cache.
//
// It returns the resulting file base, next base, and an offset FileSet to use
// for parsing.
func (c *parseCache) allocateSpace(size int) (int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.nextBase == 0 {
		// FileSet base values must be at least 1.
		c.nextBase = 1
	}
	base := c.nextBase
	c.nextBase += size + 1
	return base, c.nextBase
}

// parseFiles returns a ParsedGoFile for each file handle in fhs, in the
// requested parse mode.
//
// For parsed files that already exists in the cache, access time will be
// updated. For others, parseFiles will parse and store as many results in the
// cache as space allows.
//
// The token.File for each resulting parsed file will be added to the provided
// FileSet, using the tokeninternal.AddExistingFiles API. Consequently, the
// given fset should only be used in other APIs if its base is >=
// reservedForParsing.
//
// If parseFiles returns an error, it still returns a slice,
// but with a nil entry for each file that could not be parsed.
func (c *parseCache) parseFiles(ctx context.Context, fset *token.FileSet, mode parser.Mode, purgeFuncBodies bool, fhs ...source.FileHandle) ([]*source.ParsedGoFile, error) {
	pgfs := make([]*source.ParsedGoFile, len(fhs))

	// Temporary fall-back for 32-bit systems, where reservedForParsing is too
	// small to be viable. We don't actually support 32-bit systems, so this
	// workaround is only for tests and can be removed when we stop running
	// 32-bit TryBots for gopls.
	if bits.UintSize == 32 {
		for i, fh := range fhs {
			var err error
			pgfs[i], err = parseGoImpl(ctx, fset, fh, mode, purgeFuncBodies)
			if err != nil {
				return pgfs, err
			}
		}
		return pgfs, nil
	}

	promises, firstErr := c.startParse(mode, purgeFuncBodies, fhs...)

	// Await all parsing.
	var g errgroup.Group
	g.SetLimit(runtime.GOMAXPROCS(-1)) // parsing is CPU-bound.
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

	if err := g.Wait(); err != nil && firstErr == nil {
		firstErr = err
	}

	// Augment the FileSet to map all parsed files.
	var tokenFiles []*token.File
	for _, pgf := range pgfs {
		if pgf == nil {
			continue
		}
		tokenFiles = append(tokenFiles, pgf.Tok)
	}
	tokeninternal.AddExistingFiles(fset, tokenFiles)

	const debugIssue59080 = true
	if debugIssue59080 {
		for _, f := range tokenFiles {
			pos := token.Pos(f.Base())
			f2 := fset.File(pos)
			if f2 != f {
				panic(fmt.Sprintf("internal error: File(%d (start)) = %v, not %v", pos, f2, f))
			}
			pos = token.Pos(f.Base() + f.Size())
			f2 = fset.File(pos)
			if f2 != f {
				panic(fmt.Sprintf("internal error: File(%d (end)) = %v, not %v", pos, f2, f))
			}
		}
	}

	return pgfs, firstErr
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
