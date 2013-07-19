// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"errors"
	pathpkg "path"
	"time"

	"code.google.com/p/go.tools/godoc/util"
	"code.google.com/p/go.tools/godoc/vfs"
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

	IndexThrottle float64

	// MaxResults optionally specifies the maximum results for indexing.
	// The default is 1000.
	MaxResults int

	testDir string // TODO(bradfitz,adg): migrate old godoc flag? looks unused.

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
}

// NewCorpus returns a new Corpus from a filesystem.
// Set any options on Corpus before calling the Corpus.Init method.
func NewCorpus(fs vfs.FileSystem) *Corpus {
	c := &Corpus{
		fs: fs,
		refreshMetadataSignal: make(chan bool, 1),

		MaxResults:   1000,
		IndexEnabled: true,
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
	// TODO(bradfitz): do this in a goroutine because newDirectory might block for a long time?
	// It used to be sometimes done in a goroutine before, at least in HTTP server mode.
	if err := c.initFSTree(); err != nil {
		return err
	}
	c.updateMetadata()
	go c.refreshMetadataLoop()
	return nil
}

func (c *Corpus) initFSTree() error {
	dir := c.newDirectory(pathpkg.Join("/", c.testDir), -1)
	if dir == nil {
		return errors.New("godoc: corpus fstree is nil")
	}
	c.fsTree.Set(dir)
	c.invalidateIndex()
	return nil
}
