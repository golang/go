// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"html/template"
	"io/ioutil"
	"os"
	"reflect"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
)

func New(ctx context.Context, options func(*source.Options)) *Cache {
	index := atomic.AddInt64(&cacheIndex, 1)
	c := &Cache{
		id:          strconv.FormatInt(index, 10),
		fset:        token.NewFileSet(),
		options:     options,
		fileContent: map[span.URI]*fileHandle{},
	}
	return c
}

type Cache struct {
	id      string
	fset    *token.FileSet
	options func(*source.Options)

	store memoize.Store

	fileMu      sync.Mutex
	fileContent map[span.URI]*fileHandle
}

type fileHandle struct {
	modTime time.Time
	uri     span.URI
	bytes   []byte
	hash    string
	err     error
}

func (c *Cache) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	return c.getFile(ctx, uri)
}

func (c *Cache) getFile(ctx context.Context, uri span.URI) (*fileHandle, error) {
	fi, statErr := os.Stat(uri.Filename())
	if statErr != nil {
		return &fileHandle{err: statErr}, nil
	}

	c.fileMu.Lock()
	fh, ok := c.fileContent[uri]
	c.fileMu.Unlock()
	if ok && fh.modTime.Equal(fi.ModTime()) {
		return fh, nil
	}

	select {
	case ioLimit <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-ioLimit }()

	fh = readFile(ctx, uri, fi.ModTime())
	c.fileMu.Lock()
	c.fileContent[uri] = fh
	c.fileMu.Unlock()
	return fh, nil
}

// ioLimit limits the number of parallel file reads per process.
var ioLimit = make(chan struct{}, 128)

func readFile(ctx context.Context, uri span.URI, modTime time.Time) *fileHandle {
	ctx, done := event.Start(ctx, "cache.readFile", tag.File.Of(uri.Filename()))
	_ = ctx
	defer done()

	data, err := ioutil.ReadFile(uri.Filename())
	if err != nil {
		return &fileHandle{
			modTime: modTime,
			err:     err,
		}
	}
	return &fileHandle{
		modTime: modTime,
		uri:     uri,
		bytes:   data,
		hash:    hashContents(data),
	}
}

func (c *Cache) NewSession(ctx context.Context) *Session {
	index := atomic.AddInt64(&sessionIndex, 1)
	s := &Session{
		cache:       c,
		id:          strconv.FormatInt(index, 10),
		options:     source.DefaultOptions(),
		overlays:    make(map[span.URI]*overlay),
		gocmdRunner: &gocommand.Runner{},
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

func (h *fileHandle) Hash() string {
	return h.hash
}

func (h *fileHandle) FileIdentity() source.FileIdentity {
	return source.FileIdentity{
		URI:  h.uri,
		Hash: h.hash,
		Kind: h.Kind(),
	}
}

func (h *fileHandle) Read() ([]byte, error) {
	return h.bytes, h.err
}

func hashContents(contents []byte) string {
	return fmt.Sprintf("%x", sha256.Sum256(contents))
}

var cacheIndex, sessionIndex, viewIndex int64

func (c *Cache) ID() string                     { return c.id }
func (c *Cache) MemStats() map[reflect.Type]int { return c.store.Stats() }

type packageStat struct {
	id        packageID
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
			v := v.(*packageData)
			if v.pkg == nil {
				break
			}
			var typsCost, typInfoCost int64
			if v.pkg.types != nil {
				typsCost = typesCost(v.pkg.types.Scope())
			}
			if v.pkg.typesInfo != nil {
				typInfoCost = typesInfoCost(v.pkg.typesInfo)
			}
			stat := packageStat{
				id:        v.pkg.m.id,
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
	ast.Inspect(f, func(n ast.Node) bool {
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
