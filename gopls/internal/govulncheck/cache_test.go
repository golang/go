// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package govulncheck

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/vuln/client"
	"golang.org/x/vuln/osv"
)

func TestCache(t *testing.T) {
	tmpDir := t.TempDir()

	cache := &FSCache{rootDir: tmpDir}
	dbName := "vulndb.golang.org"

	_, _, err := cache.ReadIndex(dbName)
	if err != nil {
		t.Fatalf("ReadIndex failed for non-existent database: %v", err)
	}

	if err = os.Mkdir(filepath.Join(tmpDir, dbName), 0777); err != nil {
		t.Fatalf("os.Mkdir failed: %v", err)
	}
	_, _, err = cache.ReadIndex(dbName)
	if err != nil {
		t.Fatalf("ReadIndex failed for database without cached index: %v", err)
	}

	now := time.Now()
	expectedIdx := client.DBIndex{
		"a.vuln.example.com": time.Time{}.Add(time.Hour),
		"b.vuln.example.com": time.Time{}.Add(time.Hour * 2),
		"c.vuln.example.com": time.Time{}.Add(time.Hour * 3),
	}
	if err = cache.WriteIndex(dbName, expectedIdx, now); err != nil {
		t.Fatalf("WriteIndex failed to write index: %v", err)
	}

	idx, retrieved, err := cache.ReadIndex(dbName)
	if err != nil {
		t.Fatalf("ReadIndex failed for database with cached index: %v", err)
	}
	if !reflect.DeepEqual(idx, expectedIdx) {
		t.Errorf("ReadIndex returned unexpected index, got:\n%s\nwant:\n%s", idx, expectedIdx)
	}
	if !retrieved.Equal(now) {
		t.Errorf("ReadIndex returned unexpected retrieved: got %s, want %s", retrieved, now)
	}

	if _, err = cache.ReadEntries(dbName, "vuln.example.com"); err != nil {
		t.Fatalf("ReadEntires failed for non-existent package: %v", err)
	}

	expectedEntries := []*osv.Entry{
		{ID: "001"},
		{ID: "002"},
		{ID: "003"},
	}
	if err := cache.WriteEntries(dbName, "vuln.example.com", expectedEntries); err != nil {
		t.Fatalf("WriteEntries failed: %v", err)
	}

	entries, err := cache.ReadEntries(dbName, "vuln.example.com")
	if err != nil {
		t.Fatalf("ReadEntries failed for cached package: %v", err)
	}
	if !reflect.DeepEqual(entries, expectedEntries) {
		t.Errorf("ReadEntries returned unexpected entries, got:\n%v\nwant:\n%v", entries, expectedEntries)
	}
}

func TestConcurrency(t *testing.T) {
	tmpDir := t.TempDir()

	cache := &FSCache{rootDir: tmpDir}
	dbName := "vulndb.golang.org"

	g := new(errgroup.Group)
	for i := 0; i < 1000; i++ {
		i := i
		g.Go(func() error {
			id := i % 5
			p := fmt.Sprintf("example.com/package%d", id)

			entries, err := cache.ReadEntries(dbName, p)
			if err != nil {
				return err
			}

			err = cache.WriteEntries(dbName, p, append(entries, &osv.Entry{ID: fmt.Sprint(id)}))
			if err != nil {
				return err
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		t.Errorf("error in parallel cache entries read/write: %v", err)
	}

	// sanity checking
	for i := 0; i < 5; i++ {
		id := fmt.Sprint(i)
		p := fmt.Sprintf("example.com/package%s", id)

		es, err := cache.ReadEntries(dbName, p)
		if err != nil {
			t.Fatalf("failed to read entries: %v", err)
		}
		for _, e := range es {
			if e.ID != id {
				t.Errorf("want %s ID for vuln entry; got %s", id, e.ID)
			}
		}
	}

	// do similar for cache index
	start := time.Now()
	for i := 0; i < 1000; i++ {
		i := i
		g.Go(func() error {
			id := i % 5
			p := fmt.Sprintf("package%v", id)

			idx, _, err := cache.ReadIndex(dbName)
			if err != nil {
				return err
			}

			if idx == nil {
				idx = client.DBIndex{}
			}

			// sanity checking
			if rt, ok := idx[p]; ok && rt.Before(start) {
				return fmt.Errorf("unexpected past time in index: %v before start %v", rt, start)
			}

			now := time.Now()
			idx[p] = now
			if err := cache.WriteIndex(dbName, idx, now); err != nil {
				return err
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		t.Errorf("error in parallel cache index read/write: %v", err)
	}
}
