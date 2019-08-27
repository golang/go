// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"errors"
	"sync"
	"time"

	"golang.org/x/tools/godoc/analysis"
	"golang.org/x/tools/godoc/util"
	"golang.org/x/tools/godoc/vfs"
)

// A Corpus holds all the state related to serving and indexing a
// collection of Go code.
//
// Construct a new Corpus with NewCorpus, then modify options,
// then call its Init method.
type Corpus struct {
	fs vfs.FileSystem

	// Verbose logging.
	Verbose bool

	// IndexEnabled controls whether indexing is enabled.
	IndexEnabled bool

	// IndexFiles specifies a glob pattern specifying index files.
	// If not empty, the index is read from these files in sorted
	// order.
	IndexFiles string

	// IndexThrottle specifies the indexing throttle value
	// between 0.0 and 1.0. At 0.0, the indexer always sleeps.
	// At 1.0, the indexer never sleeps. Because 0.0 is useless
	// and redundant with setting IndexEnabled to false, the
	// zero value for IndexThrottle means 0.9.
	IndexThrottle float64

	// IndexInterval specifies the time to sleep between reindexing
	// all the sources.
	// If zero, a default is used. If negative, the index is only
	// built once.
	IndexInterval time.Duration

	// IndexDocs enables indexing of Go documentation.
	// This will produce search results for exported types, functions,
	// methods, variables, and constants, and will link to the godoc
	// documentation for those identifiers.
	IndexDocs bool

	// IndexGoCode enables indexing of Go source code.
	// This will produce search results for internal and external identifiers
	// and will link to both declarations and uses of those identifiers in
	// source code.
	IndexGoCode bool

	// IndexFullText enables full-text indexing.
	// This will provide search results for any matching text in any file that
	// is indexed, including non-Go files (see whitelisted in index.go).
	// Regexp searching is supported via full-text indexing.
	IndexFullText bool

	// MaxResults optionally specifies the maximum results for indexing.
	MaxResults int

	// SummarizePackage optionally specifies a function to
	// summarize a package. It exists as an optimization to
	// avoid reading files to parse package comments.
	//
	// If SummarizePackage returns false for ok, the caller
	// ignores all return values and parses the files in the package
	// as if SummarizePackage were nil.
	//
	// If showList is false, the package is hidden from the
	// package listing.
	SummarizePackage func(pkg string) (summary string, showList, ok bool)

	// IndexDirectory optionally specifies a function to determine
	// whether the provided directory should be indexed.  The dir
	// will be of the form "/src/cmd/6a", "/doc/play",
	// "/src/io", etc.
	// If nil, all directories are indexed if indexing is enabled.
	IndexDirectory func(dir string) bool

	// Send a value on this channel to trigger a metadata refresh.
	// It is buffered so that if a signal is not lost if sent
	// during a refresh.
	refreshMetadataSignal chan bool

	// file system information
	fsTree      util.RWValue // *Directory tree of packages, updated with each sync (but sync code is removed now)
	fsModified  util.RWValue // timestamp of last call to invalidateIndex
	docMetadata util.RWValue // mapping from paths to *Metadata

	// SearchIndex is the search index in use.
	searchIndex util.RWValue

	// Analysis is the result of type and pointer analysis.
	Analysis analysis.Result

	// flag to check whether a corpus is initialized or not
	initMu   sync.RWMutex
	initDone bool

	// pkgAPIInfo contains the information about which package API
	// features were added in which version of Go.
	pkgAPIInfo apiVersions
}

// NewCorpus returns a new Corpus from a filesystem.
// The returned corpus has all indexing enabled and MaxResults set to 1000.
// Change or set any options on Corpus before calling the Corpus.Init method.
func NewCorpus(fs vfs.FileSystem) *Corpus {
	c := &Corpus{
		fs:                    fs,
		refreshMetadataSignal: make(chan bool, 1),

		MaxResults:    1000,
		IndexEnabled:  true,
		IndexDocs:     true,
		IndexGoCode:   true,
		IndexFullText: true,
	}
	return c
}

func (c *Corpus) CurrentIndex() (*Index, time.Time) {
	v, t := c.searchIndex.Get()
	idx, _ := v.(*Index)
	return idx, t
}

func (c *Corpus) FSModifiedTime() time.Time {
	_, ts := c.fsModified.Get()
	return ts
}

// Init initializes Corpus, once options on Corpus are set.
// It must be called before any subsequent method calls.
func (c *Corpus) Init() error {
	if err := c.initFSTree(); err != nil {
		return err
	}
	c.updateMetadata()
	go c.refreshMetadataLoop()

	c.initMu.Lock()
	c.initDone = true
	c.initMu.Unlock()
	return nil
}

func (c *Corpus) initFSTree() error {
	dir := c.newDirectory("/", -1)
	if dir == nil {
		return errors.New("godoc: corpus fstree is nil")
	}
	c.fsTree.Set(dir)
	c.invalidateIndex()
	return nil
}
