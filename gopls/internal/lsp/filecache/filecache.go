// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The filecache package provides a file-based shared durable blob cache.
//
// The cache is a machine-global mapping from (kind string, key
// [32]byte) to []byte, where kind is an identifier describing the
// namespace or purpose (e.g. "analysis"), and key is a SHA-256 digest
// of the recipe of the value. (It need not be the digest of the value
// itself, so you can query the cache without knowing what value the
// recipe would produce.)
//
// The space budget of the cache can be controlled by [SetBudget].
// Cache entries may be evicted at any time or in any order.
//
// The Get and Set operations are concurrency-safe.
package filecache

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/tools/internal/robustio"
)

// Get retrieves from the cache and returns a newly allocated
// copy of the value most recently supplied to Set(kind, key),
// possibly by another process.
// Get returns ErrNotFound if the value was not found.
func Get(kind string, key [32]byte) ([]byte, error) {
	name := filename(kind, key)
	data, err := robustio.ReadFile(name)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	// Update file time for use by LRU eviction.
	// (This turns every read into a write operation.
	// If this is a performance problem, we should
	// touch the files aynchronously.)
	//
	// (Traditionally the access time would be updated
	// automatically, but for efficiency most POSIX systems have
	// for many years set the noatime mount option to avoid every
	// open or read operation entailing a metadata write.)
	now := time.Now()
	if err := os.Chtimes(name, now, now); err != nil {
		return nil, fmt.Errorf("failed to update access time: %w", err)
	}

	return data, nil
}

// ErrNotFound is the distinguished error
// returned by Get when the key is not found.
var ErrNotFound = fmt.Errorf("not found")

// Set updates the value in the cache.
func Set(kind string, key [32]byte, value []byte) error {
	name := filename(kind, key)
	if err := os.MkdirAll(filepath.Dir(name), 0700); err != nil {
		return err
	}

	// The sequence below uses rename to achieve atomic cache
	// updates even with concurrent processes.
	var cause error
	for try := 0; try < 3; try++ {
		tmpname := fmt.Sprintf("%s.tmp.%d", name, rand.Int())
		tmp, err := os.OpenFile(tmpname, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			if os.IsExist(err) {
				// Create raced with another thread (or stale file).
				// Try again.
				cause = err
				continue
			}
			return err
		}

		_, err = tmp.Write(value)
		if closeErr := tmp.Close(); err == nil {
			err = closeErr // prefer error from write over close
		}
		if err != nil {
			os.Remove(tmp.Name()) // ignore error
			return err
		}

		err = robustio.Rename(tmp.Name(), name)
		if err == nil {
			return nil // success
		}
		cause = err

		// Rename raced with another thread. Try again.
		os.Remove(tmp.Name()) // ignore error
	}
	return cause
}

var budget int64 = 1e9 // 1GB

// SetBudget sets a soft limit on disk usage of the cache (in bytes)
// and returns the previous value. Supplying a negative value queries
// the current value without changing it.
//
// If two gopls processes have different budgets, the one with the
// lower budget will collect garbage more actively, but both will
// observe the effect.
func SetBudget(new int64) (old int64) {
	if new < 0 {
		return atomic.LoadInt64(&budget)
	}
	return atomic.SwapInt64(&budget, new)
}

// --- implementation ----

// filename returns the cache entry of the specified kind and key.
//
// A typical cache entry is a file name such as:
//
//	$HOME/Library/Caches / gopls / VVVVVVVV / kind / KK / KKKK...KKKK
//
// The portions separated by spaces are as follows:
// - The user's preferred cache directory; the default value varies by OS.
// - The constant "gopls".
// - The "version", 32 bits of the digest of the gopls executable.
// - The kind or purpose of this cache subtree (e.g. "analysis").
// - The first 8 bits of the key, to avoid huge directories.
// - The full 256 bits of the key.
//
// Once a file is written its contents are never modified, though it
// may be atomically replaced or removed.
//
// New versions of gopls are free to reorganize the contents of the
// version directory as needs evolve.  But all versions of gopls must
// in perpetuity treat the "gopls" directory in a common fashion.
//
// In particular, each gopls process attempts to garbage collect
// the entire gopls directory so that newer binaries can clean up
// after older ones: in the development cycle especially, new
// new versions may be created frequently.
func filename(kind string, key [32]byte) string {
	hex := fmt.Sprintf("%x", key)
	return filepath.Join(getCacheDir(), kind, hex[:2], hex)
}

// getCacheDir returns the persistent cache directory of all processes
// running this version of the gopls executable.
//
// It must incorporate the hash of the executable so that we needn't
// worry about incompatible changes to the file format or changes to
// the algorithm that produced the index.
func getCacheDir() string {
	cacheDirOnce.Do(func() {
		// Use user's preferred cache directory.
		userDir := os.Getenv("GOPLS_CACHE")
		if userDir == "" {
			var err error
			userDir, err = os.UserCacheDir()
			if err != nil {
				userDir = os.TempDir()
			}
		}
		goplsDir := filepath.Join(userDir, "gopls")

		// UserCacheDir may return a nonexistent directory
		// (in which case we must create it, which may fail),
		// or it may return a non-writable directory, in
		// which case we should ideally respect the user's express
		// wishes (e.g. XDG_CACHE_HOME) and not write somewhere else.
		// Sadly UserCacheDir doesn't currently let us distinguish
		// such intent from accidental misconfiguraton such as HOME=/
		// in a CI builder. So, we check whether the gopls subdirectory
		// can be created (or already exists) and not fall back to /tmp.
		// See also https://github.com/golang/go/issues/57638.
		if os.MkdirAll(goplsDir, 0700) != nil {
			goplsDir = filepath.Join(os.TempDir(), "gopls")
		}

		// Start the garbage collector.
		go gc(goplsDir)

		// Compute the hash of this executable (~20ms) and create a subdirectory.
		hash, err := hashExecutable()
		if err != nil {
			log.Fatalf("can't hash gopls executable: %v", err)
		}
		// Use only 32 bits of the digest to avoid unwieldy filenames.
		// It's not an adversarial situation.
		cacheDir = filepath.Join(goplsDir, fmt.Sprintf("%x", hash[:4]))
		if err := os.MkdirAll(cacheDir, 0700); err != nil {
			log.Fatalf("can't create cache: %v", err)
		}
	})
	return cacheDir
}

var (
	cacheDirOnce sync.Once
	cacheDir     string // only accessed by getCacheDir
)

func hashExecutable() (hash [32]byte, err error) {
	exe, err := os.Executable()
	if err != nil {
		return hash, err
	}
	f, err := os.Open(exe)
	if err != nil {
		return hash, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return hash, fmt.Errorf("can't read executable: %w", err)
	}
	h.Sum(hash[:0])
	return hash, nil
}

// gc runs forever, periodically deleting files from the gopls
// directory until the space budget is no longer exceeded, and also
// deleting files older than the maximum age, regardless of budget.
//
// One gopls process may delete garbage created by a different gopls
// process, possibly running a different version of gopls, possibly
// running concurrently.
func gc(goplsDir string) {
	const period = 1 * time.Minute         // period between collections
	const statDelay = 1 * time.Millisecond // delay between stats to smooth out I/O
	const maxAge = 7 * 24 * time.Hour      // max time since last access before file is deleted

	for {
		// Enumerate all files in the cache.
		type item struct {
			path string
			stat os.FileInfo
		}
		var files []item
		var total int64 // bytes
		_ = filepath.Walk(goplsDir, func(path string, stat os.FileInfo, err error) error {
			// TODO(adonovan): opt: also collect empty directories,
			// as they typically occupy around 1KB.
			if err == nil && !stat.IsDir() {
				files = append(files, item{path, stat})
				total += stat.Size()
				time.Sleep(statDelay)
			}
			return nil
		})

		// Sort oldest files first.
		sort.Slice(files, func(i, j int) bool {
			return files[i].stat.ModTime().Before(files[j].stat.ModTime())
		})

		// Delete oldest files until we're under budget.
		// Unconditionally delete files we haven't used in ages.
		budget := atomic.LoadInt64(&budget)
		for _, file := range files {
			age := time.Since(file.stat.ModTime())
			if total > budget || age > maxAge {
				if false { // debugging
					log.Printf("deleting stale file %s (%dB, age %v)",
						file.path, file.stat.Size(), age)
				}
				os.Remove(filepath.Join(goplsDir, file.path)) // ignore error
				total -= file.stat.Size()
			}
		}

		time.Sleep(period)
	}
}
