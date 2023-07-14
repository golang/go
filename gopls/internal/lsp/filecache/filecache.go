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
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/lru"
)

// Start causes the filecache to initialize and start garbage gollection.
//
// Start is automatically called by the first call to Get, but may be called
// explicitly to pre-initialize the cache.
func Start() {
	go getCacheDir()
}

// As an optimization, use a 100MB in-memory LRU cache in front of filecache
// operations. This reduces I/O for operations such as diagnostics or
// implementations that repeatedly access the same cache entries.
var memCache = lru.New(100 * 1e6)

type memKey struct {
	kind string
	key  [32]byte
}

// Get retrieves from the cache and returns a newly allocated
// copy of the value most recently supplied to Set(kind, key),
// possibly by another process.
// Get returns ErrNotFound if the value was not found.
func Get(kind string, key [32]byte) ([]byte, error) {
	// First consult the read-through memory cache.
	// Note that memory cache hits do not update the times
	// used for LRU eviction of the file-based cache.
	if value := memCache.Get(memKey{kind, key}); value != nil {
		return value.([]byte), nil
	}

	iolimit <- struct{}{}        // acquire a token
	defer func() { <-iolimit }() // release a token

	// Read the index file, which provides the name of the CAS file.
	indexName, err := filename(kind, key)
	if err != nil {
		return nil, err
	}
	indexData, err := os.ReadFile(indexName)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	var valueHash [32]byte
	if copy(valueHash[:], indexData) != len(valueHash) {
		return nil, ErrNotFound // index entry has wrong length
	}

	// Read the CAS file and check its contents match.
	//
	// This ensures integrity in all cases (corrupt or truncated
	// file, short read, I/O error, wrong length, etc) except an
	// engineered hash collision, which is infeasible.
	casName, err := filename(casKind, valueHash)
	if err != nil {
		return nil, err
	}
	value, _ := os.ReadFile(casName) // ignore error
	if sha256.Sum256(value) != valueHash {
		return nil, ErrNotFound // CAS file is missing or has wrong contents
	}

	// Update file times used by LRU eviction.
	//
	// Because this turns a read into a write operation,
	// we follow the approach used in the go command's
	// cache and update the access time only if the
	// existing timestamp is older than one hour.
	//
	// (Traditionally the access time would be updated
	// automatically, but for efficiency most POSIX systems have
	// for many years set the noatime mount option to avoid every
	// open or read operation entailing a metadata write.)
	now := time.Now()
	touch := func(filename string) {
		st, err := os.Stat(filename)
		if err == nil && now.Sub(st.ModTime()) > time.Hour {
			os.Chtimes(filename, now, now) // ignore error
		}
	}
	touch(indexName)
	touch(casName)

	memCache.Set(memKey{kind, key}, value, len(value))

	return value, nil
}

// ErrNotFound is the distinguished error
// returned by Get when the key is not found.
var ErrNotFound = fmt.Errorf("not found")

// Set updates the value in the cache.
func Set(kind string, key [32]byte, value []byte) error {
	memCache.Set(memKey{kind, key}, value, len(value))

	// Set the active event to wake up the GC.
	select {
	case active <- struct{}{}:
	default:
	}

	iolimit <- struct{}{}        // acquire a token
	defer func() { <-iolimit }() // release a token

	// First, add the value to the content-
	// addressable store (CAS), if not present.
	hash := sha256.Sum256(value)
	casName, err := filename(casKind, hash)
	if err != nil {
		return err
	}
	// Does CAS file exist and have correct (complete) content?
	// TODO(adonovan): opt: use mmap for this check.
	if prev, _ := os.ReadFile(casName); !bytes.Equal(prev, value) {
		if err := os.MkdirAll(filepath.Dir(casName), 0700); err != nil {
			return err
		}
		// Avoiding O_TRUNC here is merely an optimization to avoid
		// cache misses when two threads race to write the same file.
		if err := writeFileNoTrunc(casName, value, 0600); err != nil {
			os.Remove(casName) // ignore error
			return err         // e.g. disk full
		}
	}

	// Now write an index entry that refers to the CAS file.
	indexName, err := filename(kind, key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(indexName), 0700); err != nil {
		return err
	}
	if err := writeFileNoTrunc(indexName, hash[:], 0600); err != nil {
		os.Remove(indexName) // ignore error
		return err           // e.g. disk full
	}

	return nil
}

// The active 1-channel is a selectable resettable event
// indicating recent cache activity.
var active = make(chan struct{}, 1)

// writeFileNoTrunc is like os.WriteFile but doesn't truncate until
// after the write, so that racing writes of the same data are idempotent.
func writeFileNoTrunc(filename string, data []byte, perm os.FileMode) error {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, perm)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err == nil {
		err = f.Truncate(int64(len(data)))
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

// reserved kind strings
const (
	casKind = "cas" // content-addressable store files
	bugKind = "bug" // gopls bug reports
)

var iolimit = make(chan struct{}, 128) // counting semaphore to limit I/O concurrency in Set.

var budget int64 = 1e9 // 1GB

// SetBudget sets a soft limit on disk usage of regular files in the
// cache (in bytes) and returns the previous value. Supplying a
// negative value queries the current value without changing it.
//
// If two gopls processes have different budgets, the one with the
// lower budget will collect garbage more actively, but both will
// observe the effect.
//
// Even in the steady state, the storage usage reported by the 'du'
// command may exceed the budget by as much as a factor of 3 due to
// the overheads of directories and the effects of block quantization,
// which are especially pronounced for the small index files.
func SetBudget(new int64) (old int64) {
	if new < 0 {
		return atomic.LoadInt64(&budget)
	}
	return atomic.SwapInt64(&budget, new)
}

// --- implementation ----

// filename returns the name of the cache file of the specified kind and key.
//
// A typical cache file has a name such as:
//
//	$HOME/Library/Caches / gopls / VVVVVVVV / KK / KKKK...KKKK - kind
//
// The portions separated by spaces are as follows:
// - The user's preferred cache directory; the default value varies by OS.
// - The constant "gopls".
// - The "version", 32 bits of the digest of the gopls executable.
// - The first 8 bits of the key, to avoid huge directories.
// - The full 256 bits of the key.
// - The kind or purpose of this cache file (e.g. "analysis").
//
// The kind establishes a namespace for the keys. It is represented as
// a suffix, not a segment, as this significantly reduces the number
// of directories created, and thus the storage overhead.
//
// Previous iterations of the design aimed for the invariant that once
// a file is written, its contents are never modified, though it may
// be atomically replaced or removed. However, not all platforms have
// an atomic rename operation (our first approach), and file locking
// (our second) is a notoriously fickle mechanism.
//
// The current design instead exploits a trick from the cache
// implementation used by the go command: writes of small files are in
// practice atomic (all or nothing) on all platforms.
// (See GOROOT/src/cmd/go/internal/cache/cache.go.)
//
// Russ Cox notes: "all file systems use an rwlock around every file
// system block, including data blocks, so any writes or reads within
// the same block are going to be handled atomically by the FS
// implementation without any need to request file locking explicitly.
// And since the files are so small, there's only one block. (A block
// is at minimum 512 bytes, usually much more.)" And: "all modern file
// systems protect against [partial writes due to power loss] with
// journals."
//
// We use a two-level scheme consisting of an index and a
// content-addressable store (CAS). A single cache entry consists of
// two files. The value of a cache entry is written into the file at
// filename("cas", sha256(value)). Since the value may be arbitrarily
// large, this write is not atomic. That means we must check the
// integrity of the contents read back from the CAS to make sure they
// hash to the expected key. If the CAS file is incomplete or
// inconsistent, we proceed as if it were missing.
//
// Once the CAS file has been written, we write a small fixed-size
// index file at filename(kind, key), using the values supplied by the
// caller. The index file contains the hash that identifies the value
// file in the CAS. (We could add extra metadata to this file, up to
// 512B, the minimum size of a disk block, if later desired, so long
// as the total size remains fixed.) Because the index file is small,
// concurrent writes to it are atomic in practice, even though this is
// not guaranteed by any OS. The fixed size ensures that readers can't
// see a palimpsest when a short new file overwrites a longer old one.
//
// New versions of gopls are free to reorganize the contents of the
// version directory as needs evolve.  But all versions of gopls must
// in perpetuity treat the "gopls" directory in a common fashion.
//
// In particular, each gopls process attempts to garbage collect
// the entire gopls directory so that newer binaries can clean up
// after older ones: in the development cycle especially, new
// versions may be created frequently.
func filename(kind string, key [32]byte) (string, error) {
	base := fmt.Sprintf("%x-%s", key, kind)
	dir, err := getCacheDir()
	if err != nil {
		return "", err
	}
	// Keep the BugReports function consistent with this one.
	return filepath.Join(dir, base[:2], base), nil
}

// getCacheDir returns the persistent cache directory of all processes
// running this version of the gopls executable.
//
// It must incorporate the hash of the executable so that we needn't
// worry about incompatible changes to the file format or changes to
// the algorithm that produced the index.
func getCacheDir() (string, error) {
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
			cacheDirErr = fmt.Errorf("can't hash gopls executable: %v", err)
		}
		// Use only 32 bits of the digest to avoid unwieldy filenames.
		// It's not an adversarial situation.
		cacheDir = filepath.Join(goplsDir, fmt.Sprintf("%x", hash[:4]))
		if err := os.MkdirAll(cacheDir, 0700); err != nil {
			cacheDirErr = fmt.Errorf("can't create cache: %v", err)
		}
	})
	return cacheDir, cacheDirErr
}

var (
	cacheDirOnce sync.Once
	cacheDir     string
	cacheDirErr  error
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
	// period between collections
	//
	// Originally the period was always 1 minute, but this
	// consumed 15% of a CPU core when idle (#61049).
	//
	// The reason for running collections even when idle is so
	// that long lived gopls sessions eventually clean up the
	// caches created by defunct executables.
	const (
		minPeriod = 5 * time.Minute // when active
		maxPeriod = 6 * time.Hour   // when idle
	)

	// Sleep statDelay*batchSize between stats to smooth out I/O.
	//
	// The constants below were chosen using the following heuristics:
	//  - 1GB of filecache is on the order of ~100-200k files, in which case
	//    100μs delay per file introduces 10-20s of additional walk time,
	//    less than the minPeriod.
	//  - Processing batches of stats at once is much more efficient than
	//    sleeping after every stat (due to OS optimizations).
	const statDelay = 100 * time.Microsecond // average delay between stats, to smooth out I/O
	const batchSize = 1000                   // # of stats to process before sleeping
	const maxAge = 5 * 24 * time.Hour        // max time since last access before file is deleted

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
			path  string
			mtime time.Time
			size  int64
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
					files = append(files, item{path, stat.ModTime(), stat.Size()})
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
			return files[i].mtime.Before(files[j].mtime)
		})

		// Delete oldest files until we're under budget.
		budget := atomic.LoadInt64(&budget)
		for _, file := range files {
			if total < budget {
				break
			}
			if debug {
				age := time.Since(file.mtime)
				log.Printf("budget: deleting stale file %s (%dB, age %v)",
					file.path, file.size, age)
			}
			os.Remove(file.path) // ignore error
			total -= file.size
		}
		files = nil // release memory before sleep

		// Wait unconditionally for the minimum period.
		time.Sleep(minPeriod)

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

		// Wait up to the max period,
		// or for Set activity in this process.
		select {
		case <-active:
		case <-time.After(maxPeriod):
		}
	}
}

func init() {
	// Register a handler to durably record this process's first
	// assertion failure in the cache so that we can ask users to
	// share this information via the stats command.
	bug.Handle(func(bug bug.Bug) {
		// Wait for cache init (bugs in tests happen early).
		_, _ = getCacheDir()

		data, err := json.Marshal(bug)
		if err != nil {
			panic(fmt.Sprintf("error marshalling bug %+v: %v", bug, err))
		}

		key := sha256.Sum256(data)
		_ = Set(bugKind, key, data)
	})
}

// BugReports returns a new unordered array of the contents
// of all cached bug reports produced by this executable.
// It also returns the location of the cache directory
// used by this process (or "" on initialization error).
func BugReports() (string, []bug.Bug) {
	// To test this logic, run:
	// $ TEST_GOPLS_BUG=oops gopls bug     # trigger a bug
	// $ gopls stats                       # list the bugs

	dir, err := getCacheDir()
	if err != nil {
		return "", nil // ignore initialization errors
	}
	var result []bug.Bug
	_ = filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return nil // ignore readdir/stat errors
		}
		// Parse the key from each "XXXX-bug" cache file name.
		if !info.IsDir() && strings.HasSuffix(path, bugKind) {
			var key [32]byte
			n, err := hex.Decode(key[:], []byte(filepath.Base(path)[:len(key)*2]))
			if err != nil || n != len(key) {
				return nil // ignore malformed file names
			}
			content, err := Get(bugKind, key)
			if err == nil { // ignore read errors
				var b bug.Bug
				if err := json.Unmarshal(content, &b); err != nil {
					log.Printf("error marshalling bug %q: %v", string(content), err)
				}
				result = append(result, b)
			}
		}
		return nil
	})
	return dir, result
}
