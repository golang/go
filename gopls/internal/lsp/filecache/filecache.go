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
// Note that "du -sh $GOPLSCACHE" may report a disk usage
// figure that is rather larger (e.g. 50%) than the budget because
// it rounds up partial disk blocks.
//
// The Get and Set operations are concurrency-safe.
package filecache

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/lockedfile"
)

// Start causes the filecache to initialize and start garbage gollection.
//
// Start is automatically called by the first call to Get, but may be called
// explicitly to pre-initialize the cache.
func Start() {
	go getCacheDir()
}

// Get retrieves from the cache and returns a newly allocated
// copy of the value most recently supplied to Set(kind, key),
// possibly by another process.
// Get returns ErrNotFound if the value was not found.
func Get(kind string, key [32]byte) ([]byte, error) {
	name := filename(kind, key)
	data, err := lockedfile.Read(name)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrNotFound
		}
		return nil, err
	}

	// Verify that the Write was complete
	// by checking the recorded length.
	if len(data) < 8+4 {
		return nil, ErrNotFound // cache entry is incomplete
	}
	length, value, checksum := data[:8], data[8:len(data)-4], data[len(data)-4:]
	if binary.LittleEndian.Uint64(length) != uint64(len(value)) {
		return nil, ErrNotFound // cache entry is incomplete (or too long!)
	}

	// Check for corruption and print the entire file content; see
	// issue #59289. TODO(adonovan): stop printing the entire file
	// once we've seen enough reports to understand the pattern.
	if binary.LittleEndian.Uint32(checksum) != crc32.ChecksumIEEE(value) {
		return nil, bug.Errorf("internal error in filecache.Get(%q, %x): invalid checksum at end of %d-byte file %s:\n%q",
			kind, key, len(data), name, data)
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

	return value, nil
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

	// In the unlikely event of a short write (e.g. ENOSPC)
	// followed by process termination (e.g. a power cut), we
	// don't want a reader to see a short file, so we record
	// the expected length first and verify it in Get.
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(value)))

	// Occasional file corruption (presence of zero bytes in JSON
	// files) has been reported on macOS (see issue #59289),
	// assumed due to a nonatomicity problem in the file system.
	// Ideally the macOS kernel would be fixed, or lockedfile
	// would implement a workaround (since its job is to provide
	// reliable the mutual exclusion primitive that allows
	// cooperating gopls processes to implement transactional
	// file replacement), but for now we add an extra integrity
	// check: a 32-bit checksum at the end.
	var checksum [4]byte
	binary.LittleEndian.PutUint32(checksum[:], crc32.ChecksumIEEE(value))

	// Windows doesn't support atomic rename--we tried MoveFile,
	// MoveFileEx, ReplaceFileEx, and SetFileInformationByHandle
	// of RenameFileInfo, all to no avail--so instead we use
	// advisory file locking, which is only about 2x slower even
	// on POSIX platforms with atomic rename.
	return lockedfile.Write(name, io.MultiReader(
		bytes.NewReader(length[:]),
		bytes.NewReader(value),
		bytes.NewReader(checksum[:])),
		0600)
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
		userDir := os.Getenv("GOPLSCACHE")
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
	const period = 1 * time.Minute // period between collections
	// Sleep statDelay*batchSize between stats to smooth out I/O.
	//
	// The constants below were chosen using the following heuristics:
	//  - 1GB of filecache is on the order of ~100-200k files, in which case
	//    100μs delay per file introduces 10-20s of additional walk time, less
	//    than the 1m gc period.
	//  - Processing batches of stats at once is much more efficient than
	//    sleeping after every stat (due to OS optimizations).
	const statDelay = 100 * time.Microsecond // average delay between stats, to smooth out I/O
	const batchSize = 1000                   // # of stats to process before sleeping
	maxAge := 5 * 24 * time.Hour             // max time since last access before file is deleted

	// This environment variable is set when running under a Go test builder.
	// We use it to trigger much more aggressive cache eviction to prevent
	// filling of the tmp volume by short-lived test processes.
	// A single run of the gopls tests takes on the order of a minute
	// and produces <50MB of cache data, so these are still generous.
	if os.Getenv("GO_BUILDER_NAME") != "" {
		maxAge = 1 * time.Hour
		SetBudget(250 * 1e6) // 250MB
	}

	// The macOS filesystem is strikingly slow, at least on some machines.
	// /usr/bin/find achieves only about 25,000 stats per second
	// at full speed (no pause between items), meaning a large
	// cache may take several minutes to scan.
	// We must ensure that short-lived processes (crucially,
	// tests) are able to make progress sweeping garbage.
	//
	// (gopls' caches should never actually get this big in
	// practice: the example mentioned above resulted from a bug
	// that caused filecache to fail to delete any files.)

	const debug = false

	// Names of all directories found in first pass; nil thereafter.
	dirs := make(map[string]bool)

	for {
		// Enumerate all files in the cache.
		type item struct {
			path string
			stat os.FileInfo
		}
		var files []item
		start := time.Now()
		var total int64 // bytes
		_ = filepath.Walk(goplsDir, func(path string, stat os.FileInfo, err error) error {
			if err != nil {
				return nil // ignore errors
			}
			if stat.IsDir() {
				// Collect (potentially empty) directories.
				if dirs != nil {
					dirs[path] = true
				}
			} else {
				// Unconditionally delete files we haven't used in ages.
				// (We do this here, not in the second loop, so that we
				// perform age-based collection even in short-lived processes.)
				age := time.Since(stat.ModTime())
				if age > maxAge {
					if debug {
						log.Printf("age: deleting stale file %s (%dB, age %v)",
							path, stat.Size(), age)
					}
					os.Remove(path) // ignore error
				} else {
					files = append(files, item{path, stat})
					total += stat.Size()
					if debug && len(files)%1000 == 0 {
						log.Printf("filecache: checked %d files in %v", len(files), time.Since(start))
					}
					if len(files)%batchSize == 0 {
						time.Sleep(batchSize * statDelay)
					}
				}
			}
			return nil
		})

		// Sort oldest files first.
		sort.Slice(files, func(i, j int) bool {
			return files[i].stat.ModTime().Before(files[j].stat.ModTime())
		})

		// Delete oldest files until we're under budget.
		budget := atomic.LoadInt64(&budget)
		for _, file := range files {
			if total < budget {
				break
			}
			if debug {
				age := time.Since(file.stat.ModTime())
				log.Printf("budget: deleting stale file %s (%dB, age %v)",
					file.path, file.stat.Size(), age)
			}
			os.Remove(file.path) // ignore error
			total -= file.stat.Size()
		}

		time.Sleep(period)

		// Once only, delete all directories.
		// This will succeed only for the empty ones,
		// and ensures that stale directories (whose
		// files have been deleted) are removed eventually.
		// They don't take up much space but they do slow
		// down the traversal.
		//
		// We do this after the sleep to minimize the
		// race against Set, which may create a directory
		// that is momentarily empty.
		//
		// (Test processes don't live that long, so
		// this may not be reached on the CI builders.)
		if dirs != nil {
			dirnames := make([]string, 0, len(dirs))
			for dir := range dirs {
				dirnames = append(dirnames, dir)
			}
			dirs = nil

			// Descending length order => children before parents.
			sort.Slice(dirnames, func(i, j int) bool {
				return len(dirnames[i]) > len(dirnames[j])
			})
			var deleted int
			for _, dir := range dirnames {
				if os.Remove(dir) == nil { // ignore error
					deleted++
				}
			}
			if debug {
				log.Printf("deleted %d empty directories", deleted)
			}
		}
	}
}
