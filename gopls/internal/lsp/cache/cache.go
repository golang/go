// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"html/template"
	"os"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/memoize"
)

// New Creates a new cache for gopls operation results, using the given file
// set, shared store, and session options.
//
// Both the fset and store may be nil, but if store is non-nil so must be fset
// (and they must always be used together), otherwise it may be possible to get
// cached data referencing token.Pos values not mapped by the FileSet.
func New(fset *token.FileSet, store *memoize.Store) *Cache {
	index := atomic.AddInt64(&cacheIndex, 1)

	if store != nil && fset == nil {
		panic("non-nil store with nil fset")
	}
	if fset == nil {
		fset = token.NewFileSet()
	}
	if store == nil {
		store = &memoize.Store{}
	}

	c := &Cache{
		id:          strconv.FormatInt(index, 10),
		fset:        fset,
		store:       store,
		fileContent: map[span.URI]*DiskFile{},
	}
	return c
}

type Cache struct {
	id   string
	fset *token.FileSet

	store *memoize.Store

	fileMu      sync.Mutex
	fileContent map[span.URI]*DiskFile
}

// A DiskFile is a file on the filesystem, or a failure to read one.
// It implements the source.FileHandle interface.
type DiskFile struct {
	uri     span.URI
	modTime time.Time
	content []byte
	hash    source.Hash
	err     error
}

func (h *DiskFile) URI() span.URI { return h.uri }

func (h *DiskFile) FileIdentity() source.FileIdentity {
	return source.FileIdentity{
		URI:  h.uri,
		Hash: h.hash,
	}
}

func (h *DiskFile) Saved() bool    { return true }
func (h *DiskFile) Version() int32 { return 0 }

func (h *DiskFile) Read() ([]byte, error) {
	return h.content, h.err
}

// GetFile stats and (maybe) reads the file, updates the cache, and returns it.
func (c *Cache) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	fi, statErr := os.Stat(uri.Filename())
	if statErr != nil {
		return &DiskFile{
			err: statErr,
			uri: uri,
		}, nil
	}

	// We check if the file has changed by comparing modification times. Notably,
	// this is an imperfect heuristic as various systems have low resolution
	// mtimes (as much as 1s on WSL or s390x builders), so we only cache
	// filehandles if mtime is old enough to be reliable, meaning that we don't
	// expect a subsequent write to have the same mtime.
	//
	// The coarsest mtime precision we've seen in practice is 1s, so consider
	// mtime to be unreliable if it is less than 2s old. Capture this before
	// doing anything else.
	recentlyModified := time.Since(fi.ModTime()) < 2*time.Second

	c.fileMu.Lock()
	fh, ok := c.fileContent[uri]
	c.fileMu.Unlock()

	if ok && fh.modTime.Equal(fi.ModTime()) {
		return fh, nil
	}

	fh, err := readFile(ctx, uri, fi) // ~25us
	if err != nil {
		return nil, err
	}
	c.fileMu.Lock()
	if !recentlyModified {
		c.fileContent[uri] = fh
	} else {
		delete(c.fileContent, uri)
	}
	c.fileMu.Unlock()
	return fh, nil
}

// ioLimit limits the number of parallel file reads per process.
var ioLimit = make(chan struct{}, 128)

func readFile(ctx context.Context, uri span.URI, fi os.FileInfo) (*DiskFile, error) {
	select {
	case ioLimit <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-ioLimit }()

	ctx, done := event.Start(ctx, "cache.readFile", tag.File.Of(uri.Filename()))
	_ = ctx
	defer done()

	content, err := os.ReadFile(uri.Filename()) // ~20us
	if err != nil {
		content = nil // just in case
	}
	return &DiskFile{
		modTime: fi.ModTime(),
		uri:     uri,
		content: content,
		hash:    source.HashOf(content),
		err:     err,
	}, nil
}

// NewSession creates a new gopls session with the given cache and options overrides.
//
// The provided optionsOverrides may be nil.
func NewSession(ctx context.Context, c *Cache, optionsOverrides func(*source.Options)) *Session {
	index := atomic.AddInt64(&sessionIndex, 1)
	options := source.DefaultOptions().Clone()
	if optionsOverrides != nil {
		optionsOverrides(options)
	}
	s := &Session{
		id:          strconv.FormatInt(index, 10),
		cache:       c,
		gocmdRunner: &gocommand.Runner{},
		options:     options,
		overlays:    make(map[span.URI]*Overlay),
	}
	event.Log(ctx, "New session", KeyCreateSession.Of(s))
	return s
}

var cacheIndex, sessionIndex, viewIndex int64

func (c *Cache) ID() string                     { return c.id }
func (c *Cache) MemStats() map[reflect.Type]int { return c.store.Stats() }

type packageStat struct {
	id        PackageID
	mode      source.ParseMode
	file      int64
	ast       int64
	types     int64
	typesInfo int64
	total     int64
}

func (c *Cache) PackageStats(withNames bool) template.HTML {
	var packageStats []packageStat
	c.store.DebugOnlyIterate(func(k, v interface{}) {
		switch k.(type) {
		case packageHandleKey:
			v := v.(typeCheckResult)
			if v.pkg == nil {
				break
			}
			typsCost := typesCost(v.pkg.types.Scope())
			typInfoCost := typesInfoCost(v.pkg.typesInfo)
			stat := packageStat{
				id:        v.pkg.id,
				mode:      v.pkg.mode,
				types:     typsCost,
				typesInfo: typInfoCost,
			}
			for _, f := range v.pkg.compiledGoFiles {
				stat.file += int64(len(f.Src))
				stat.ast += astCost(f.File)
			}
			stat.total = stat.file + stat.ast + stat.types + stat.typesInfo
			packageStats = append(packageStats, stat)
		}
	})
	var totalCost int64
	for _, stat := range packageStats {
		totalCost += stat.total
	}
	sort.Slice(packageStats, func(i, j int) bool {
		return packageStats[i].total > packageStats[j].total
	})
	html := "<table><thead><td>Name</td><td>total = file + ast + types + types info</td></thead>\n"
	human := func(n int64) string {
		return fmt.Sprintf("%.2f", float64(n)/(1024*1024))
	}
	var printedCost int64
	for _, stat := range packageStats {
		name := stat.id
		if !withNames {
			name = "-"
		}
		html += fmt.Sprintf("<tr><td>%v (%v)</td><td>%v = %v + %v + %v + %v</td></tr>\n", name, stat.mode,
			human(stat.total), human(stat.file), human(stat.ast), human(stat.types), human(stat.typesInfo))
		printedCost += stat.total
		if float64(printedCost) > float64(totalCost)*.9 {
			break
		}
	}
	html += "</table>\n"
	return template.HTML(html)
}

func astCost(f *ast.File) int64 {
	if f == nil {
		return 0
	}
	var count int64
	ast.Inspect(f, func(_ ast.Node) bool {
		count += 32 // nodes are pretty small.
		return true
	})
	return count
}

func typesCost(scope *types.Scope) int64 {
	cost := 64 + int64(scope.Len())*128 // types.object looks pretty big
	for i := 0; i < scope.NumChildren(); i++ {
		cost += typesCost(scope.Child(i))
	}
	return cost
}

func typesInfoCost(info *types.Info) int64 {
	// Most of these refer to existing objects, with the exception of InitOrder, Selections, and Types.
	cost := 24*len(info.Defs) +
		32*len(info.Implicits) +
		256*len(info.InitOrder) + // these are big, but there aren't many of them.
		32*len(info.Scopes) +
		128*len(info.Selections) + // wild guess
		128*len(info.Types) + // wild guess
		32*len(info.Uses)
	return int64(cost)
}
