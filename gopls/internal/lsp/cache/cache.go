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
	"reflect"
	"sort"
	"strconv"
	"sync/atomic"

	"golang.org/x/tools/gopls/internal/lsp/source"
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
		id:         strconv.FormatInt(index, 10),
		fset:       fset,
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
	id   string
	fset *token.FileSet

	store *memoize.Store

	*memoizedFS // implements source.FileSource
}

// NewSession creates a new gopls session with the given cache and options overrides.
//
// The provided optionsOverrides may be nil.
//
// TODO(rfindley): move this to session.go.
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
		overlayFS:   newOverlayFS(c),
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
