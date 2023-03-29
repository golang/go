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

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/tokeninternal"
)

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

// A parseCache holds a bounded number of recently accessed parsed Go files. As
// new files are stored, older files may be evicted from the cache.
//
// The parseCache.parseFiles method exposes a batch API for parsing (and
// caching) multiple files. This is necessary for type-checking, where files
// must be parsed in a common fileset.
type parseCache struct {
	mu       sync.Mutex
	m        map[parseKey]*parseCacheEntry
	lru      queue  // min-atime priority queue of *parseCacheEntry
	clock    uint64 // clock time, incremented when the cache is updated
	nextBase int    // base offset for the next parsed file
}

// parseKey uniquely identifies a parsed Go file.
type parseKey struct {
	file source.FileIdentity
	mode parser.Mode
}

type parseCacheEntry struct {
	key      parseKey
	promise  *memoize.Promise // memoize.Promise[*source.ParsedGoFile]
	atime    uint64           // clock time of last access
	lruIndex int
}

// startParse prepares a parsing pass, creating new promises in the cache for
// any cache misses.
//
// The resulting slice has an entry for every given file handle, though some
// entries may be nil if there was an error reading the file (in which case the
// resulting error will be non-nil).
func (c *parseCache) startParse(mode parser.Mode, fhs ...source.FileHandle) ([]*memoize.Promise, error) {
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

		if e, ok := c.m[key]; ok { // cache hit
			e.atime = c.clock
			heap.Fix(&c.lru, e.lruIndex)
			promises[i] = e.promise
			continue
		}

		uri := fh.URI()
		promise := memoize.NewPromise("parseCache.parse", func(ctx context.Context, _ interface{}) interface{} {
			// Allocate 2*len(content)+parsePadding to allow for re-parsing once
			// inside of parseGoSrc without exceeding the allocated space.
			base, nextBase := c.allocateSpace(2*len(content) + parsePadding)

			pgf, fixes1 := ParseGoSrc(ctx, fileSetWithBase(base), uri, content, mode)
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
				pgf2, fixes2 := ParseGoSrc(ctx, fileSetWithBase(base2), uri, content, mode)

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

// The parse cache is not supported on 32-bit systems, where reservedForParsing
// is too small to be viable.
func parseCacheSupported() bool {
	return bits.UintSize != 32
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
func (c *parseCache) parseFiles(ctx context.Context, fset *token.FileSet, mode parser.Mode, fhs ...source.FileHandle) ([]*source.ParsedGoFile, error) {
	pgfs := make([]*source.ParsedGoFile, len(fhs))

	// Temporary fall-back for 32-bit systems, where reservedForParsing is too
	// small to be viable. We don't actually support 32-bit systems, so this
	// workaround is only for tests and can be removed when we stop running
	// 32-bit TryBots for gopls.
	if bits.UintSize == 32 {
		for i, fh := range fhs {
			var err error
			pgfs[i], err = parseGoImpl(ctx, fset, fh, mode)
			if err != nil {
				return pgfs, err
			}
		}
		return pgfs, nil
	}

	promises, firstErr := c.startParse(mode, fhs...)

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
