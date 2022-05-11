// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vulncheck

import (
	"encoding/json"
	"go/build"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	"golang.org/x/vuln/client"
	"golang.org/x/vuln/osv"
)

// copy from x/vuln/cmd/govulncheck/cache.go

// The cache uses a single JSON index file for each vulnerability database
// which contains the map from packages to the time the last
// vulnerability for that package was added/modified and the time that
// the index was retrieved from the vulnerability database. The JSON
// format is as follows:
//
// $GOPATH/pkg/mod/cache/download/vulndb/{db hostname}/indexes/index.json
//   {
//       Retrieved time.Time
//       Index client.DBIndex
//   }
//
// Each package also has a JSON file which contains the array of vulnerability
// entries for the package. The JSON format is as follows:
//
// $GOPATH/pkg/mod/cache/download/vulndb/{db hostname}/{import path}/vulns.json
//   []*osv.Entry

// fsCache is a thread-safe file-system cache implementing osv.Cache
//
// TODO: use something like cmd/go/internal/lockedfile for thread safety?
type fsCache struct {
	mu      sync.Mutex
	rootDir string
}

// use cfg.GOMODCACHE available in cmd/go/internal?
var defaultCacheRoot = filepath.Join(build.Default.GOPATH, "/pkg/mod/cache/download/vulndb")

func defaultCache() *fsCache {
	return &fsCache{rootDir: defaultCacheRoot}
}

type cachedIndex struct {
	Retrieved time.Time
	Index     client.DBIndex
}

func (c *fsCache) ReadIndex(dbName string) (client.DBIndex, time.Time, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	b, err := ioutil.ReadFile(filepath.Join(c.rootDir, dbName, "index.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, time.Time{}, nil
		}
		return nil, time.Time{}, err
	}
	var index cachedIndex
	if err := json.Unmarshal(b, &index); err != nil {
		return nil, time.Time{}, err
	}
	return index.Index, index.Retrieved, nil
}

func (c *fsCache) WriteIndex(dbName string, index client.DBIndex, retrieved time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	path := filepath.Join(c.rootDir, dbName)
	if err := os.MkdirAll(path, 0755); err != nil {
		return err
	}
	j, err := json.Marshal(cachedIndex{
		Index:     index,
		Retrieved: retrieved,
	})
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(path, "index.json"), j, 0666); err != nil {
		return err
	}
	return nil
}

func (c *fsCache) ReadEntries(dbName string, p string) ([]*osv.Entry, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	b, err := ioutil.ReadFile(filepath.Join(c.rootDir, dbName, p, "vulns.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var entries []*osv.Entry
	if err := json.Unmarshal(b, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

func (c *fsCache) WriteEntries(dbName string, p string, entries []*osv.Entry) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	path := filepath.Join(c.rootDir, dbName, p)
	if err := os.MkdirAll(path, 0777); err != nil {
		return err
	}
	j, err := json.Marshal(entries)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(path, "vulns.json"), j, 0666); err != nil {
		return err
	}
	return nil
}
