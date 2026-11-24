// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cache implements a build artifact cache.
package cache

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"internal/godebug"
	"io"
	"io/fs"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/mmap"
	"cmd/internal/par"
)

// An ActionID is a cache action key, the hash of a complete description of a
// repeatable computation (command line, environment variables,
// input file contents, executable contents).
type ActionID [HashSize]byte

// An OutputID is a cache output key, the hash of an output of a computation.
type OutputID [HashSize]byte

// Cache is the interface as used by the cmd/go.
type Cache interface {
	// Get returns the cache entry for the provided ActionID.
	// On miss, the error type should be of type *entryNotFoundError.
	//
	// After a successful call to Get, OutputFile(Entry.OutputID) must
	// exist on disk until Close is called (at the end of the process).
	Get(ActionID) (Entry, error)

	// Put adds an item to the cache.
	//
	// The seeker is only used to seek to the beginning. After a call to Put,
	// the seek position is not guaranteed to be in any particular state.
	//
	// As a special case, if the ReadSeeker is of type noVerifyReadSeeker,
	// the verification from GODEBUG=gocacheverify=1 is skipped.
	//
	// After a successful call to Put, OutputFile(OutputID) must
	// exist on disk until Close is called (at the end of the process).
	Put(ActionID, io.ReadSeeker) (_ OutputID, size int64, _ error)

	// Close is called at the end of the go process. Implementations can do
	// cache cleanup work at this phase, or wait for and report any errors from
	// background cleanup work started earlier. Any cache trimming in one
	// process should not cause the invariants of this interface to be
	// violated in another process. Namely, a cache trim from one process should
	// not delete an OutputID from disk that was recently Get or Put from
	// another process. As a rule of thumb, don't trim things used in the last
	// day (with GOEXPERIMENT=trimbuildcache, DiskCache may trim young generation
	// files unused for at least 60 minutes, and refreshes files it accessed every
	// 20 minutes so concurrent commands do not trim them, see youngGenDeleteAge below).
	Close() error

	// OutputFile returns the path on disk where OutputID is stored.
	//
	// It's only called after a successful get or put call so it doesn't need
	// to return an error; it's assumed that if the previous get or put succeeded,
	// it's already on disk.
	OutputFile(OutputID) string

	// FuzzDir returns where fuzz files are stored.
	FuzzDir() string
}

// A DiskCache is a package cache, backed by a file system directory tree.
type DiskCache struct {
	dir    string
	now    func() time.Time
	statfs func(dir string) (avail, total, bsize uint64) // file system blocks (zeros if unknown)

	trimBuildCache bool         // GOEXPERIMENT=trimbuildcache is enabled
	openTime       time.Time    // wall time at Open (youngGenDeleteAge cutoff in Close)
	sizeSample     int64        // sampling stride (defaultSizeSample outside tests)
	sizeMinTarget  int64        // min size trim target (defaultMinTarget outside tests)
	sampleAcc      atomic.Int64 // random offset in [0, sizeSample) plus bytes added
	used           sync.Map     // map[usedKey]struct{} of entries accessed by this process
	refreshOnce    sync.Once    // starts the background keepalive goroutine on first access
}

// Open opens and returns the cache in the given directory.
//
// It is safe for multiple processes on a single machine to use the
// same cache directory in a local file system simultaneously.
// They will coordinate using operating system file locks and may
// duplicate effort but will not corrupt the cache.
//
// However, it is NOT safe for multiple processes on different machines
// to share a cache directory (for example, if the directory were stored
// in a network file system). File locking is notoriously unreliable in
// network file systems and may not suffice to protect the cache.
func Open(dir string) (*DiskCache, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, &fs.PathError{Op: "open", Path: dir, Err: fmt.Errorf("not a directory")}
	}
	for i := 0; i < 256; i++ {
		name := filepath.Join(dir, fmt.Sprintf("%02x", i))
		if err := os.MkdirAll(name, 0o777); err != nil {
			return nil, err
		}
	}
	c := &DiskCache{
		dir:            dir,
		now:            time.Now,
		statfs:         statfs,
		trimBuildCache: cfg.Experiment != nil && cfg.Experiment.TrimBuildCache,
		sizeSample:     defaultSizeSample,
		sizeMinTarget:  defaultMinTarget,
	}
	if c.trimBuildCache {
		c.openTime = c.now().Round(0)
		c.sampleAcc.Store(rand.Int64N(c.sizeSample))
	}
	return c, nil
}

// fileName returns the name of the file corresponding to the given id.
func (c *DiskCache) fileName(id [HashSize]byte, key string) string {
	return filepath.Join(c.dir, fmt.Sprintf("%02x", id[0]), fmt.Sprintf("%x", id)+"-"+key)
}

// An entryNotFoundError indicates that a cache entry was not found, with an
// optional underlying reason.
type entryNotFoundError struct {
	Err error
}

func (e *entryNotFoundError) Error() string {
	if e.Err == nil {
		return "cache entry not found"
	}
	return fmt.Sprintf("cache entry not found: %v", e.Err)
}

func (e *entryNotFoundError) Unwrap() error {
	return e.Err
}

const (
	// action entry file is "v1 <hex id> <hex out> <decimal size space-padded to 20 bytes> <unixnano space-padded to 20 bytes>\n"
	hexSize   = HashSize * 2
	entrySize = 2 + 1 + hexSize + 1 + hexSize + 1 + 20 + 1 + 20 + 1
)

// verify controls whether to run the cache in verify mode.
// In verify mode, the cache always returns errMissing from Get
// but then double-checks in Put that the data being written
// exactly matches any existing entry. This provides an easy
// way to detect program behavior that would have been different
// had the cache entry been returned from Get.
//
// verify is enabled by setting the environment variable
// GODEBUG=gocacheverify=1.
var verify = false

var errVerifyMode = errors.New("gocacheverify=1")

// DebugTest is set when GODEBUG=gocachetest=1 is in the environment.
var DebugTest = false

func init() { initEnv() }

var (
	gocacheverify = godebug.New("gocacheverify")
	gocachehash   = godebug.New("gocachehash")
	gocachetest   = godebug.New("gocachetest")
)

func initEnv() {
	if gocacheverify.Value() == "1" {
		gocacheverify.IncNonDefault()
		verify = true
	}
	if gocachehash.Value() == "1" {
		gocachehash.IncNonDefault()
		debugHash = true
	}
	if gocachetest.Value() == "1" {
		gocachetest.IncNonDefault()
		DebugTest = true
	}
}

// Get looks up the action ID in the cache,
// returning the corresponding output ID and file size, if any.
// Note that finding an output ID does not guarantee that the
// saved file for that output ID is still available.
func (c *DiskCache) Get(id ActionID) (Entry, error) {
	if verify {
		return Entry{}, &entryNotFoundError{Err: errVerifyMode}
	}
	return c.get(id)
}

type Entry struct {
	OutputID OutputID
	Size     int64
	Time     time.Time // when added to cache
}

// get is Get but does not respect verify mode, so that Put can use it.
func (c *DiskCache) get(id ActionID) (Entry, error) {
	missing := func(reason error) (Entry, error) {
		return Entry{}, &entryNotFoundError{Err: reason}
	}
	f, err := os.Open(c.fileName(id, "a"))
	if err != nil {
		return missing(err)
	}
	defer f.Close()
	entry := make([]byte, entrySize+1) // +1 to detect whether f is too long
	if n, err := io.ReadFull(f, entry); n > entrySize {
		return missing(errors.New("too long"))
	} else if err != io.ErrUnexpectedEOF {
		if err == io.EOF {
			return missing(errors.New("file is empty"))
		}
		return missing(err)
	} else if n < entrySize {
		return missing(errors.New("entry file incomplete"))
	}
	if entry[0] != 'v' || entry[1] != '1' || entry[2] != ' ' || entry[3+hexSize] != ' ' || entry[3+hexSize+1+hexSize] != ' ' || entry[3+hexSize+1+hexSize+1+20] != ' ' || entry[entrySize-1] != '\n' {
		return missing(errors.New("invalid header"))
	}
	eid, entry := entry[3:3+hexSize], entry[3+hexSize:]
	eout, entry := entry[1:1+hexSize], entry[1+hexSize:]
	esize, entry := entry[1:1+20], entry[1+20:]
	etime, entry := entry[1:1+20], entry[1+20:]
	var buf [HashSize]byte
	if _, err := hex.Decode(buf[:], eid); err != nil {
		return missing(fmt.Errorf("decoding ID: %v", err))
	} else if buf != id {
		return missing(errors.New("mismatched ID"))
	}
	if _, err := hex.Decode(buf[:], eout); err != nil {
		return missing(fmt.Errorf("decoding output ID: %v", err))
	}
	i := 0
	for i < len(esize) && esize[i] == ' ' {
		i++
	}
	size, err := strconv.ParseInt(string(esize[i:]), 10, 64)
	if err != nil {
		return missing(fmt.Errorf("parsing size: %v", err))
	} else if size < 0 {
		return missing(errors.New("negative size"))
	}
	i = 0
	for i < len(etime) && etime[i] == ' ' {
		i++
	}
	tm, err := strconv.ParseInt(string(etime[i:]), 10, 64)
	if err != nil {
		return missing(fmt.Errorf("parsing timestamp: %v", err))
	} else if tm < 0 {
		return missing(errors.New("negative timestamp"))
	}

	c.markUsed(c.fileName(id, "a"))
	c.noteUsed(id, 'a')
	c.noteUsed(buf, 'd')

	return Entry{buf, size, time.Unix(0, tm)}, nil
}

// GetFile looks up the action ID in the cache and returns
// the name of the corresponding data file.
func GetFile(c Cache, id ActionID) (file string, entry Entry, err error) {
	entry, err = c.Get(id)
	if err != nil {
		return "", Entry{}, err
	}
	file = c.OutputFile(entry.OutputID)
	info, err := os.Stat(file)
	if err != nil {
		return "", Entry{}, &entryNotFoundError{Err: err}
	}
	if info.Size() != entry.Size {
		return "", Entry{}, &entryNotFoundError{Err: errors.New("file incomplete")}
	}
	return file, entry, nil
}

// GetBytes looks up the action ID in the cache and returns
// the corresponding output bytes.
// GetBytes should only be used for data that can be expected to fit in memory.
func GetBytes(c Cache, id ActionID) ([]byte, Entry, error) {
	entry, err := c.Get(id)
	if err != nil {
		return nil, entry, err
	}
	data, _ := os.ReadFile(c.OutputFile(entry.OutputID))
	if sha256.Sum256(data) != entry.OutputID {
		return nil, entry, &entryNotFoundError{Err: errors.New("bad checksum")}
	}
	return data, entry, nil
}

// GetMmap looks up the action ID in the cache and returns
// the corresponding output bytes.
// GetMmap should only be used for data that can be expected to fit in memory.
// The boolean result indicates whether the file was opened.
// If it is true, the caller should avoid attempting
// to write to the file on Windows, because Windows locks
// the open file, and writes to it will fail.
func GetMmap(c Cache, id ActionID) ([]byte, Entry, bool, error) {
	entry, err := c.Get(id)
	if err != nil {
		return nil, entry, false, err
	}
	md, opened, err := mmap.Mmap(c.OutputFile(entry.OutputID))
	if err != nil {
		return nil, Entry{}, opened, err
	}
	if int64(len(md.Data)) != entry.Size {
		return nil, Entry{}, true, &entryNotFoundError{Err: errors.New("file incomplete")}
	}
	return md.Data, entry, true, nil
}

// OutputFile returns the name of the cache file storing output with the given OutputID.
func (c *DiskCache) OutputFile(out OutputID) string {
	file := c.fileName(out, "d")
	isDir := c.markUsed(file)
	c.noteUsed(out, 'd')
	if isDir { // => cached executable
		entries, err := os.ReadDir(file)
		if err != nil {
			return fmt.Sprintf("DO NOT USE - missing binary cache entry: %v", err)
		}
		if len(entries) != 1 {
			return "DO NOT USE - invalid binary cache entry"
		}
		return filepath.Join(file, entries[0].Name())
	}
	return file
}

// Time constants for cache expiration.
//
// We set the mtime on a cache file on each use, but at most one per mtimeInterval (1 hour),
// to avoid causing many unnecessary inode updates. The mtimes therefore
// roughly reflect "time of last use" but may in fact be older by at most an hour.
//
// We scan the cache for entries to delete at most once per trimInterval (1 day).
//
// When we do scan the cache, we delete entries that have not been used for
// at least trimLimit (5 days). Statistics gathered from a month of usage by
// Go developers found that essentially all reuse of cached entries happened
// within 5 days of the previous reuse. See golang.org/issue/22990.
//
// Update (GOEXPERIMENT=trimbuildcache):
// Some users report very large caches when running builds over and over.
// See go.dev/issue/29561.
// As an attempt to identify and remove non-useful cache entries earlier,
// we split the cache logically into a young and tenured30m generation
// and trigger trims based on cache size growth (see size.txt below).
// The young generation consists of files that were generated and used
// in the first 30 minutes after their creation but not reused since.
// Young generation files that reach 90 minutes old (youngGenDeleteAge) can be removed
// during a size triggered trim (so every deleted file went unused for at least 60 minutes).
// A young generation file that is reused after reaching the age of 30 minutes (usedAfter30m)
// is promoted to the tenured30m generation, becoming a tenured30m file.
// (The name tenured30m leaves room for a later longer horizon tenured generation.)
// Tenured30m files that reach 5 days + 1 hour old (trimLimit + mtimeInterval) can be removed.
//
// Generations are encoded in file mtimes: newly written files get a whole minute mtime (:00),
// and tenured30m files get a :30 mtime on promotion and later mtimeInterval refreshes.
// Any mtime not on a whole minute (including files written by older Go) is treated as tenured30m,
// and rewriting an already cached entry counts as a use rather than resetting its mark.
//
// Size triggered trims run at most once per minTrimSpacing (1 hour) when a command adds
// sampled bytes and the estimated size in size.txt reaches targetSize(base). Close measures
// youngGenDeleteAge from openTime so a long go command never deletes its own output, and a
// long running DiskCache refreshes accessed files every refreshInterval (20 minutes) without
// tenuring young files so concurrent go commands do not trim them either.
// Daily trims (trimInterval, 24 hours) still only delete files unused for 5 days + 1 hour
// and never delete young generation files early.
// On Linux, a nearly full disk (see lowDiskFree) also triggers a trim at most once per
// minTrimSpacing, even if no bytes were added. It deletes any file whose mtime is at
// least youngGenDeleteAge (90 minutes) old regardless of generation. Since running go
// commands refresh files they use every refreshInterval (20 minutes), both young and
// tenured30m files in use are refreshed before their mtime reaches 80 minutes.
// For more rationale, see https://go.dev/cl/XXXXX.
const (
	mtimeInterval = 1 * time.Hour
	trimInterval  = 24 * time.Hour
	trimClockSkew = 1 * time.Hour // tolerate future trim.txt timestamps up to this much
	trimLimit     = 5 * 24 * time.Hour

	minTrimSpacing    = 1 * time.Hour    // min time from any trim to a size triggered trim
	usedAfter30m      = 30 * time.Minute // reuse at this age promotes to tenured30m
	youngGenDeleteAge = 90 * time.Minute // size trim delete age for young generation files
	refreshInterval   = 20 * time.Minute // how often a long running process refreshes accessed files
)

// Size constants for estimating cache size in size.txt (GOEXPERIMENT=trimbuildcache).
//
// To avoid updating size.txt on every small build, each go command initializes sampleAcc to
// a random offset U in [0, defaultSizeSample) and adds newly written logical bytes to it.
// After adding B bytes, Trim/Close/exit adds floor((B+U)/defaultSizeSample) * defaultSizeSample
// to size.txt and keeps the remainder. The expected increment is B, while a command adding
// no bytes never touches size.txt and one adding 1 MiB updates it only once in 64 runs.
//
// Each trim scan replaces the estimate with the measured size of the entries it kept,
// plus any bytes other go commands added during the scan, and records as the baseline
// ("base") the kept bytes a size triggered trim could not delete at that moment
// (tenured30m files, young generation files under 90 minutes old, and undeletable files).
// A new or cleaned cache has base 0, so its first size triggered trim runs when the
// estimate reaches defaultMinTarget (5 GiB).
//
// Completed trim scans and 'go clean -cache' also append a line to cache.log (see log.go).
const (
	defaultSizeSample int64 = 64 << 20 // 64 MiB
	defaultMinTarget  int64 = 5 << 30  // 5 GiB, floor for targetSize
	maxSizeBound      int64 = 1 << 60  // cap size arithmetic well clear of overflow
)

// addBytes records that n new logical bytes were written to the cache.
func (c *DiskCache) addBytes(n int64) {
	if c.trimBuildCache && n > 0 {
		c.sampleAcc.Add(n)
	}
}

// takeSampledAdded removes and returns the whole sampling strides from sampleAcc.
func (c *DiskCache) takeSampledAdded() int64 {
	stride := c.sizeSample
	for {
		v := c.sampleAcc.Load()
		if v < stride {
			return 0
		}
		if c.sampleAcc.CompareAndSwap(v, v%stride) {
			return (v / stride) * stride
		}
	}
}

// flush records any pending sampled bytes in size.txt without scanning the cache.
// It runs via base.AtExit, so it must not report errors or exit.
func (c *DiskCache) flush() {
	if !c.trimBuildCache || c.sampleAcc.Load() < c.sizeSample {
		return
	}
	_ = c.withTrimLock(func(*lockedfile.File) {
		if delta := c.takeSampledAdded(); delta > 0 {
			c.addSizeLocked(delta)
		}
	})
}

// CleanCache resets trim metadata after 'go clean -cache' removes the cache entries in dir.
// It truncates trim.txt (so an ongoing trim in another process won't overwrite the reset),
// resets an existing size.txt to zero, and logs the clean in cache.log.
// It creates nothing in an empty or nonexistent cache.
func CleanCache(dir string) {
	exists := func(name string) bool {
		_, err := os.Stat(filepath.Join(dir, name))
		return err == nil
	}
	if !exists("size.txt") && !exists("trim.txt") && !exists("cache.log") {
		return
	}
	_ = withTrimLockDir(dir, func(trimFile *lockedfile.File) {
		_ = trimFile.Truncate(0)
		if _, sizeExists, ok := readSize(dir); sizeExists && ok {
			writeSizeLocked(dir, cacheSizeState{})
		}
		appendLogLocked(dir, time.Now(), "clean -cache")
	})
}

// isInterrupted reports whether the go command has received an interrupt.
func isInterrupted() bool {
	select {
	case <-base.Interrupted:
		return true
	default:
		return false
	}
}

// withTrimLock calls fn while holding the exclusive lock on trim.txt.
// See withTrimLockDir.
func (c *DiskCache) withTrimLock(fn func(trimFile *lockedfile.File)) error {
	return withTrimLockDir(c.dir, fn)
}

// withTrimLockDir calls fn while holding an exclusive lock on dir/trim.txt,
// which serializes all reads and writes of trim.txt, size.txt, and cache.log.
// Within fn, trim.txt must be accessed only through trimFile: opening and closing
// it separately can drop a POSIX fcntl lock (which is per process, not per fd),
// and locking it again would deadlock.
//
// If locking fails (e.g., a read-only cache), fn is not called and nil is returned,
// except when file locking is unsupported on the platform.
func withTrimLockDir(dir string, fn func(trimFile *lockedfile.File)) error {
	f, err := lockedfile.Edit(filepath.Join(dir, "trim.txt"))
	if err != nil {
		if errors.Is(err, errors.ErrUnsupported) {
			return err
		}
		return nil
	}
	defer f.Close()
	fn(f)
	return nil
}

// trimDue reports whether a trim is due at now given the contents of trim.txt and
// the required interval. A missing or corrupt timestamp, or one more than
// trimClockSkew in the future, is treated as due.
func trimDue(data []byte, now time.Time, interval time.Duration) bool {
	last, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	if err != nil {
		return true
	}
	d := now.Sub(time.Unix(last, 0))
	return d >= interval || d <= -trimClockSkew
}

// writeTrimTimestamp overwrites trimFile with Unix timestamp t and returns what it wrote.
func writeTrimTimestamp(trimFile *lockedfile.File, t int64) ([]byte, error) {
	data := strconv.AppendInt(nil, t, 10)
	if _, err := trimFile.WriteAt(data, 0); err != nil {
		return nil, err
	}
	if err := trimFile.Truncate(int64(len(data))); err != nil {
		return nil, err
	}
	return data, nil
}

// cacheSizeState holds the parsed contents of size.txt ("key value\n" lines).
type cacheSizeState struct {
	estimated    int64 // estimated logical size of cache entries, in bytes
	base         int64 // baseline size from the last trim scan (0 if none)
	hasEstimated bool  // size.txt held a valid estimate
}

// readSize reads dir/size.txt. Callers that modify size.txt must hold the trim.txt lock.
//
// A missing file returns exists=false, ok=true, and a read error on an existing file returns
// exists=true, ok=false (callers must not overwrite it). Lines without a trailing newline
// (such as from a torn write), unknown keys, and invalid values are ignored. For duplicate
// keys, the first valid line wins.
func readSize(dir string) (st cacheSizeState, exists, ok bool) {
	data, err := os.ReadFile(filepath.Join(dir, "size.txt"))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return cacheSizeState{}, false, true
		}
		return cacheSizeState{}, true, false
	}
	rest := string(data)
	var hasBase bool
	for {
		line, after, found := strings.Cut(rest, "\n")
		if !found {
			break
		}
		rest = after
		key, val, found := strings.Cut(strings.TrimSpace(line), " ")
		if !found {
			continue
		}
		val = strings.TrimSpace(val)
		switch key {
		case "estimated", "base":
			n, err := strconv.ParseInt(val, 10, 64)
			if err != nil || n < 0 {
				continue
			}
			n = min(n, maxSizeBound)
			if key == "estimated" && !st.hasEstimated {
				st.estimated, st.hasEstimated = n, true
			} else if key == "base" && !hasBase {
				st.base, hasBase = n, true
			}
		}
	}
	return st, true, true
}

// writeSizeLocked writes st to dir/size.txt while holding the trim.txt lock.
// It rewrites in place (avoiding renames over open files on Windows) and writes
// before truncating so an ENOSPC failure does not leave an empty file.
// Every write starts with both keys, so any trailing bytes left by a crash
// before Truncate are ignored by readSize and dropped on the next write.
func writeSizeLocked(dir string, st cacheSizeState) {
	buf := fmt.Appendf(nil, "base %d\nestimated %d\n", st.base, st.estimated)
	f, err := os.OpenFile(filepath.Join(dir, "size.txt"), os.O_WRONLY|os.O_CREATE, 0o666)
	if err != nil {
		return
	}
	if _, err := f.Write(buf); err == nil {
		_ = f.Truncate(int64(len(buf)))
	}
	_ = f.Close()
}

// addSizeLocked adds delta to an existing estimate in size.txt while holding the
// trim.txt lock, returning the updated state. If size.txt has no valid estimate,
// it does nothing (the next trim scan records one). If unreadable, it returns ok=false.
func (c *DiskCache) addSizeLocked(delta int64) (st cacheSizeState, ok bool) {
	st, _, ok = readSize(c.dir)
	if ok && st.hasEstimated && delta > 0 {
		st.estimated = min(st.estimated+delta, maxSizeBound)
		writeSizeLocked(c.dir, st)
	}
	return st, ok
}

// targetSize returns the cache size threshold for the next size triggered trim.
// Modeled lightly on GOGC, it scales with base: 1.5x up to 20 GiB and 1.25x above,
// floored at c.sizeMinTarget. For more rationale, see https://go.dev/cl/XXXXX.
// TODO(thepudds): consider smoothing out 1.5x to 1.25x transition, but not critical and important to have two growth factors.
func (c *DiskCache) targetSize(base int64) int64 {
	const high = 20 << 30
	if base > high {
		return max(c.sizeMinTarget, base+base/4)
	}
	return max(c.sizeMinTarget, base+base/2)
}

// lowDiskBytes is the low disk threshold, unless 5% of the file system is smaller.
const lowDiskBytes = 512 << 20

// lowDiskFree reports whether the file system holding the cache is nearly full, with less
// than lowDiskBytes or 5% of its blocks (whichever is smaller) available to unprivileged
// users, and if so how many bytes are available. It is best effort and false outside Linux.
func (c *DiskCache) lowDiskFree() (free int64, low bool) {
	avail, total, bsize := c.statfs(c.dir)
	// avail > (lowDiskBytes-1)/bsize is avail*bsize >= lowDiskBytes without overflow.
	if bsize == 0 || avail >= total/20 || avail > (lowDiskBytes-1)/bsize {
		return 0, false
	}
	return int64(avail * bsize), true
}

// writeTime returns the mtime for a newly written cache file: the current
// whole minute (:00) under GOEXPERIMENT=trimbuildcache, or c.now().
func (c *DiskCache) writeTime() time.Time {
	if c.trimBuildCache {
		return c.now().Truncate(time.Minute)
	}
	return c.now()
}

// isYoungGen reports whether mt carries the young generation mark (a whole minute, :00).
// Any other mtime belongs to the tenured30m generation.
func isYoungGen(mt time.Time) bool {
	return mt.Nanosecond() == 0 && mt.Unix()%60 == 0
}

// markUsed makes a best-effort attempt to update mtime on file,
// so that mtime reflects cache access time.
//
// Because the reflection only needs to be approximate,
// and to reduce the amount of disk activity caused by using
// cache entries, used only updates the mtime if the current
// mtime is more than an hour old. This heuristic eliminates
// nearly all of the mtime updates that would otherwise happen,
// while still keeping the mtimes useful for cache trimming.
//
// With GOEXPERIMENT=trimbuildcache, a young generation file is updated
// once it is at least usedAfter30m old, promoting it to the tenured30m
// generation by setting mtime to the current minute minus 30 seconds.
//
// markUsed reports whether the file is a directory (an executable cache entry).
func (c *DiskCache) markUsed(file string) (isDir bool) {
	info, err := os.Stat(file)
	if err != nil {
		return false
	}
	interval := mtimeInterval
	if c.trimBuildCache && isYoungGen(info.ModTime()) {
		interval = usedAfter30m
	}
	if now := c.now(); now.Sub(info.ModTime()) >= interval {
		if c.trimBuildCache {
			now = now.Truncate(time.Minute).Add(-30 * time.Second)
		}
		os.Chtimes(file, now, now)
	}
	return info.IsDir()
}

// usedKey identifies a cache file ("<id>-a" or "<id>-d") accessed by this process.
type usedKey struct {
	id   [HashSize]byte
	kind byte
}

// noteUsed records a cache file accessed by this DiskCache and starts the
// background keepalive goroutine on first access.
func (c *DiskCache) noteUsed(id [HashSize]byte, kind byte) {
	if !c.trimBuildCache {
		return
	}
	c.used.Store(usedKey{id: id, kind: kind}, struct{}{})
	c.refreshOnce.Do(func() {
		go func() {
			for !isInterrupted() {
				time.Sleep(refreshInterval)
				c.refreshUsed()
			}
		}()
	})
}

// refreshUsed refreshes every cache file accessed through this DiskCache so
// concurrent go commands do not trim them while this process is still running.
// Young generation files keep their :00 mark instead of being promoted to
// tenured30m, so a long command does not tenure its own single use outputs.
func (c *DiskCache) refreshUsed() {
	c.used.Range(func(k, _ any) bool {
		if isInterrupted() {
			return false
		}
		u := k.(usedKey)
		file := c.fileName(u.id, string(u.kind))
		info, err := os.Stat(file)
		if err != nil {
			return true
		}
		if mt := info.ModTime(); isYoungGen(mt) {
			if now := c.now(); now.Sub(mt) >= mtimeInterval {
				now = now.Truncate(time.Minute)
				os.Chtimes(file, now, now)
			}
			return true
		}
		c.markUsed(file)
		return true
	})
}

func (c *DiskCache) Close() error {
	return c.trim(c.openTime) // never delete this go command's own output
}

// Trim removes old cache entries that are likely not to be reused.
func (c *DiskCache) Trim() error {
	return c.trim(time.Time{})
}

// trim implements Trim and Close for GOEXPERIMENT=trimbuildcache, flushing sampled
// bytes to size.txt, running a daily, size, or low disk triggered scan when due, and logging
// the result. Young generation and low disk delete ages are measured as of youngGenAsOf (or now if zero).
func (c *DiskCache) trim(youngGenAsOf time.Time) error {
	if !c.trimBuildCache {
		return c.trimLegacy()
	}
	// Strip the monotonic clock for wall clock comparisons with trim.txt and file
	// mtimes, keeping it in start only to measure scan duration.
	start := c.now()
	now := start.Round(0)
	if youngGenAsOf.IsZero() || youngGenAsOf.After(now) {
		youngGenAsOf = now
	}

	// If there are no sampled bytes to add to size.txt, check whether a daily or low disk
	// trim is due under a shared lock, avoiding an exclusive lock in the common case.
	if c.sampleAcc.Load() < c.sizeSample {
		if isInterrupted() {
			return nil
		}
		if data, err := lockedfile.Read(filepath.Join(c.dir, "trim.txt")); err == nil && !trimDue(data, now, trimInterval) {
			if !trimDue(data, now, minTrimSpacing) {
				return nil
			}
			if _, low := c.lowDiskFree(); !low {
				return nil
			}
		}
	}

	var (
		walk     bool
		sizeTrim bool           // size triggered trim (daily 5 day trim only if neither this nor lowDisk)
		lowDisk  bool           // low disk triggered trim, ignoring tenure
		free     int64          // bytes available on a nearly full disk (lowDisk only)
		stamp    []byte         // trim.txt content written by this scan
		sizeOK   bool           // size.txt was readable before the scan
		before   cacheSizeState // size.txt state before the scan
	)
	err := c.withTrimLock(func(trimFile *lockedfile.File) {
		added := c.takeSampledAdded()
		if isInterrupted() {
			if added > 0 {
				c.addSizeLocked(added)
			}
			return
		}
		data, _ := io.ReadAll(trimFile)
		var (
			st cacheSizeState
			ok bool
		)
		if trimDue(data, now, minTrimSpacing) {
			free, lowDisk = c.lowDiskFree()
		}
		switch {
		case added > 0:
			// This command grew the cache: update size.txt and check the size trigger.
			st, ok = c.addSizeLocked(added)
			sizeTrim = ok && st.hasEstimated && st.estimated >= c.targetSize(st.base) &&
				trimDue(data, now, minTrimSpacing)
			if !sizeTrim && !lowDisk && !trimDue(data, now, trimInterval) {
				return
			}
		case lowDisk, trimDue(data, now, trimInterval):
			// Low disk or daily 24 hour trim (the latter never deletes young generation files early).
			st, _, ok = readSize(c.dir)
		default:
			return
		}

		// Write the new timestamp before scanning, so that other go
		// commands do not start scanning at the same time (#76314).
		// This is best effort: a full disk is all the more reason to scan.
		stamp = data
		if newStamp, err := writeTrimTimestamp(trimFile, now.Unix()); err == nil {
			stamp = newStamp
		}
		walk, sizeOK, before = true, ok, st
	})
	if err != nil || !walk {
		return err
	}

	// Trim the 256 subdirectories in parallel without holding the lock.
	// We subtract an additional mtimeInterval
	// to account for the imprecision of our "last used" mtimes.
	cutoff := now.Add(-trimLimit - mtimeInterval)
	classify := func(mt time.Time) (remove, inBase bool) {
		switch {
		case mt.Before(cutoff):
			return true, false // unused for 5 days, either generation
		case !isYoungGen(mt):
			// Tenured30m generation: only a low disk trim deletes it early.
			return lowDisk && youngGenAsOf.Sub(mt) >= youngGenDeleteAge, true
		}
		// Young generation: only a size or low disk triggered trim deletes it early,
		// and it counts in the baseline while too young to delete as of now.
		return (sizeTrim || lowDisk) && youngGenAsOf.Sub(mt) >= youngGenDeleteAge, now.Sub(mt) < youngGenDeleteAge
	}
	limit := 8 // lower cap outside Linux for small fd limits
	if runtime.GOOS == "linux" {
		limit = 32
	}
	q := par.NewQueue(min(runtime.GOMAXPROCS(0), limit))
	var perDir [256]trimStats
	for i := range 256 {
		q.Add(func() {
			if isInterrupted() {
				return
			}
			subdir := filepath.Join(c.dir, fmt.Sprintf("%02x", i))
			c.trimSubdir(subdir, classify, &perDir[i])
		})
	}
	<-q.Idle()
	if isInterrupted() {
		return nil // we don't want incomplete results
	}
	var stats trimStats
	for _, s := range perDir {
		stats.add(s)
	}
	end := c.now()
	elapsed := end.Sub(start)
	end = end.Round(0)

	return c.withTrimLock(func(trimFile *lockedfile.File) {
		// If trim.txt changed during the scan (e.g., from 'go clean -cache' or
		// another trim), our scan results are stale. Reject an empty stamp too,
		// since it would match an empty trim.txt left by 'go clean -cache'.
		if data, _ := io.ReadAll(trimFile); len(stamp) == 0 || !bytes.Equal(data, stamp) {
			return
		}
		base := min(stats.baseBytes, maxSizeBound)
		// Keep any bytes that other go commands added during the scan.
		if sizeOK {
			if st, _, ok := readSize(c.dir); ok {
				var concAdded int64
				if before.hasEstimated && st.hasEstimated {
					concAdded = max(0, st.estimated-before.estimated)
				}
				st.estimated = min(stats.keptBytes+concAdded, maxSizeBound)
				st.base = base
				writeSizeLocked(c.dir, st)
			}
		}
		_, _ = writeTrimTimestamp(trimFile, end.Unix())
		trigger := "24h"
		switch {
		case lowDisk:
			trigger = "low disk"
		case sizeTrim:
			trigger = "size"
		}
		actual := stats.keptBytes + stats.removedBytes
		var estNote, freeNote string
		if before.hasEstimated {
			estNote = formatEstimateDelta(before.estimated, actual)
		}
		if lowDisk {
			freeNote = fmt.Sprintf(" free=%s (consider providing more space)", formatBytes(free))
		}
		appendLogLocked(c.dir, end, "trim: %s trigger: %s -> %s in %s, removed %d of %d files, base=%s next=%s%s%s",
			trigger, formatBytes(actual), formatBytes(stats.keptBytes),
			elapsed.Round(time.Millisecond), stats.removedFiles, stats.keptFiles+stats.removedFiles,
			formatBytes(base), formatBytes(c.targetSize(base)), estNote, freeNote)
	})
}

// trimLegacy implements Trim for GOEXPERIMENT=notrimbuildcache.
func (c *DiskCache) trimLegacy() error {
	now := c.now()

	// We maintain in dir/trim.txt the time of the last completed cache trim.
	// If the cache has been trimmed recently enough, do nothing.
	// This is the common case.
	// If the trim file is corrupt, detected if the file can't be parsed, or the
	// trim time is too far in the future, attempt the trim anyway. It's possible that
	// the cache was full when the corruption happened. Attempting a trim on
	// an empty cache is cheap, so there wouldn't be a big performance hit in that case.
	// Check to see if we need a trim. Do this check separately from the lockedfile.Transform
	// so that we can skip getting an exclusive lock in the common case.
	if data, err := lockedfile.Read(filepath.Join(c.dir, "trim.txt")); err == nil {
		if !trimDue(data, now, trimInterval) {
			return nil
		}
	}

	errFileChanged := errors.New("file changed")

	// Write the new timestamp before we start trimming to reduce the chance that multiple invocations
	// try to trim at the same time, causing contention in CI (#76314).
	err := lockedfile.Transform(filepath.Join(c.dir, "trim.txt"), func(data []byte) ([]byte, error) {
		if !trimDue(data, now, trimInterval) {
			// The timestamp in the file no longer meets the criteria for us to
			// do a trim. It must have been updated by another go command invocation
			// since we last read it. Skip the trim.
			return nil, errFileChanged
		}
		return fmt.Appendf(nil, "%d", now.Unix()), nil
	})
	if errors.Is(err, errors.ErrUnsupported) {
		return err
	}
	if errors.Is(err, errFileChanged) {
		// Skip the trim because we don't need it anymore.
		return nil
	}

	// Trim each of the 256 subdirectories.
	// We subtract an additional mtimeInterval
	// to account for the imprecision of our "last used" mtimes.
	cutoff := now.Add(-trimLimit - mtimeInterval)
	classify := func(mt time.Time) (remove, inBase bool) { return mt.Before(cutoff), false }
	for i := 0; i < 256; i++ {
		subdir := filepath.Join(c.dir, fmt.Sprintf("%02x", i))
		c.trimSubdir(subdir, classify, nil)
	}

	return nil
}

// trimStats tracks the counts and logical byte sizes of entries seen by a trim scan.
type trimStats struct {
	keptFiles, keptBytes       int64
	removedFiles, removedBytes int64
	baseBytes                  int64 // kept bytes counted toward the baseline
}

func (s *trimStats) add(o trimStats) {
	s.keptFiles += o.keptFiles
	s.keptBytes += o.keptBytes
	s.removedFiles += o.removedFiles
	s.removedBytes += o.removedBytes
	s.baseBytes += o.baseBytes
}

// trimSubdir trims a single cache subdirectory, accumulating counts into stats if non-nil.
func (c *DiskCache) trimSubdir(subdir string, classify func(mt time.Time) (remove, inBase bool), stats *trimStats) {
	// Read all directory entries from subdir before removing
	// any files, in case removing files invalidates the file offset
	// in the directory scan. Also, ignore error from f.Readdirnames,
	// because we don't care about reporting the error and we still
	// want to process any entries found before the error.
	f, err := os.Open(subdir)
	if err != nil {
		return
	}
	names, _ := f.Readdirnames(-1)
	f.Close()

	for _, name := range names {
		// Remove only cache entries (xxxx-a and xxxx-d).
		if !strings.HasSuffix(name, "-a") && !strings.HasSuffix(name, "-d") {
			continue
		}
		entry := filepath.Join(subdir, name)
		info, err := os.Stat(entry)
		if err != nil {
			continue
		}
		var size int64
		if stats != nil {
			size = info.Size()
			if info.IsDir() {
				size = dirFilesSize(entry)
			}
		}
		remove, inBase := classify(info.ModTime())
		if remove {
			if info.IsDir() { // executable cache entry
				err = os.RemoveAll(entry)
			} else {
				err = os.Remove(entry)
			}
			if err == nil || errors.Is(err, fs.ErrNotExist) {
				if stats != nil {
					stats.removedFiles++
					stats.removedBytes += size
				}
				continue
			}
			inBase = true // could not be deleted
		}
		if stats != nil {
			stats.keptFiles++
			stats.keptBytes += size
			if inBase {
				stats.baseBytes += size
			}
		}
	}
}

// dirFilesSize returns the size of regular files directly in dir (for cached executables).
func dirFilesSize(dir string) int64 {
	entries, _ := os.ReadDir(dir)
	var total int64
	for _, e := range entries {
		if info, err := e.Info(); err == nil && info.Mode().IsRegular() {
			total += info.Size()
		}
	}
	return total
}

// putIndexEntry adds an entry to the cache recording that executing the action
// with the given id produces an output with the given output id (hash) and size.
func (c *DiskCache) putIndexEntry(id ActionID, out OutputID, size int64, allowVerify bool) error {
	// Note: We expect that for one reason or another it may happen
	// that repeating an action produces a different output hash
	// (for example, if the output contains a time stamp or temp dir name).
	// While not ideal, this is also not a correctness problem, so we
	// don't make a big deal about it. In particular, we leave the action
	// cache entries writable specifically so that they can be overwritten.
	//
	// Setting GODEBUG=gocacheverify=1 does make a big deal:
	// in verify mode we are double-checking that the cache entries
	// are entirely reproducible. As just noted, this may be unrealistic
	// in some cases but the check is also useful for shaking out real bugs.
	entry := fmt.Sprintf("v1 %x %x %20d %20d\n", id, out, size, time.Now().UnixNano())
	if verify && allowVerify {
		old, err := c.get(id)
		if err == nil && (old.OutputID != out || old.Size != size) {
			// panic to show stack trace, so we can see what code is generating this cache entry.
			msg := fmt.Sprintf("go: internal cache error: cache verify failed: id=%x changed:<<<\n%s\n>>>\nold: %x %d\nnew: %x %d", id, reverseHash(id), out, size, old.OutputID, old.Size)
			panic(msg)
		}
	}
	file := c.fileName(id, "a")
	var prev fs.FileInfo
	if c.trimBuildCache {
		// Stat before writing to count new entries and check any prior generation mark.
		prev, _ = os.Stat(file)
	}

	// Copy file to cache directory.
	mode := os.O_WRONLY | os.O_CREATE
	f, err := os.OpenFile(file, mode, 0o666)
	if err != nil {
		return err
	}
	_, err = f.WriteString(entry)
	if err == nil {
		// Truncate the file only *after* writing it.
		// (This should be a no-op, but truncate just in case of previous corruption.)
		//
		// This differs from os.WriteFile, which truncates to 0 *before* writing
		// via os.O_TRUNC. Truncating only after writing ensures that a second write
		// of the same content to the same file is idempotent, and does not — even
		// temporarily! — undo the effect of the first write.
		err = f.Truncate(int64(len(entry)))
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		// TODO(bcmills): This Remove potentially races with another go command writing to file.
		// Can we eliminate it?
		os.Remove(file)
		return err
	}
	mark := c.writeTime()
	if c.trimBuildCache && prev != nil {
		// Rewriting an existing index entry counts as a use (matching copyFile):
		// promote or keep a young generation mark, and refresh an older tenured30m
		// mark. Skip tenuring if mt > tenured, which can happen when a concurrent
		// cold write has not yet reached Chtimes and still has its raw OS mtime.
		mt := prev.ModTime()
		tenured := mark.Add(-30 * time.Second)
		switch {
		case isYoungGen(mt):
			if mark.Sub(mt) >= usedAfter30m {
				mark = tenured
			} else if !mt.After(mark) {
				mark = mt
			}
		case !mt.After(tenured):
			mark = tenured
		}
	}
	os.Chtimes(file, mark, mark)
	if prev == nil {
		c.addBytes(entrySize)
	}
	c.noteUsed(id, 'a')
	c.noteUsed(out, 'd')

	return nil
}

// noVerifyReadSeeker is an io.ReadSeeker wrapper sentinel type
// that says that Cache.Put should skip the verify check
// (from GODEBUG=gocacheverify=1).
type noVerifyReadSeeker struct {
	io.ReadSeeker
}

// Put stores the given output in the cache as the output for the action ID.
// It may read file twice. The content of file must not change between the two passes.
func (c *DiskCache) Put(id ActionID, file io.ReadSeeker) (OutputID, int64, error) {
	wrapper, isNoVerify := file.(noVerifyReadSeeker)
	if isNoVerify {
		file = wrapper.ReadSeeker
	}
	return c.put(id, "", file, !isNoVerify)
}

// PutExecutable is used to store the output as the output for the action ID into a
// file with the given base name, with the executable mode bit set.
// It may read file twice. The content of file must not change between the two passes.
func (c *DiskCache) PutExecutable(id ActionID, name string, file io.ReadSeeker) (OutputID, int64, error) {
	if name == "" {
		panic("PutExecutable called without a name")
	}
	wrapper, isNoVerify := file.(noVerifyReadSeeker)
	if isNoVerify {
		file = wrapper.ReadSeeker
	}
	return c.put(id, name, file, !isNoVerify)
}

// PutNoVerify is like Put but disables the verify check
// when GODEBUG=gocacheverify=1 is set.
// It is meant for data that is OK to cache but that we expect to vary slightly from run to run,
// like test output containing times and the like.
func PutNoVerify(c Cache, id ActionID, file io.ReadSeeker) (OutputID, int64, error) {
	return c.Put(id, noVerifyReadSeeker{file})
}

func (c *DiskCache) put(id ActionID, executableName string, file io.ReadSeeker, allowVerify bool) (OutputID, int64, error) {
	// Compute output ID.
	h := sha256.New()
	if _, err := file.Seek(0, 0); err != nil {
		return OutputID{}, 0, err
	}
	size, err := io.Copy(h, file)
	if err != nil {
		return OutputID{}, 0, err
	}
	var out OutputID
	h.Sum(out[:0])

	// Copy to cached output file (if not already present).
	fileMode := fs.FileMode(0o666)
	if executableName != "" {
		fileMode = 0o777
	}
	if err := c.copyFile(file, executableName, out, size, fileMode); err != nil {
		return out, size, err
	}

	// Add to cache index.
	return out, size, c.putIndexEntry(id, out, size, allowVerify)
}

// PutBytes stores the given bytes in the cache as the output for the action ID.
func PutBytes(c Cache, id ActionID, data []byte) error {
	_, _, err := c.Put(id, bytes.NewReader(data))
	return err
}

// copyFile copies file into the cache, expecting it to have the given
// output ID and size, if that file is not present already.
func (c *DiskCache) copyFile(file io.ReadSeeker, executableName string, out OutputID, size int64, perm os.FileMode) error {
	name := c.fileName(out, "d") // TODO(matloob): use a different suffix for the executable cache?
	entry := name
	info, err := os.Stat(name)
	if executableName != "" {
		// This is an executable file. The file at name won't hold the output itself, but will
		// be a directory that holds the output, named according to executableName. Check to see
		// if the directory already exists, and if it does not, create it. Then reset name
		// to the name we want the output written to.
		if err != nil {
			if !os.IsNotExist(err) {
				return err
			}
			if err := os.Mkdir(name, 0o777); err != nil {
				return err
			}
			if info, err = os.Stat(name); err != nil {
				return err
			}
		}
		if !info.IsDir() {
			return errors.New("internal error: invalid binary cache entry: not a directory")
		}

		// directory exists. now set name to the inner file
		name = filepath.Join(name, executableName)
		info, err = os.Stat(name)
	}
	var oldSize int64 // logical bytes already in the file, not counted again
	if err == nil {
		oldSize = info.Size()
	}
	if err == nil && info.Size() == size {
		// Check hash.
		if f, err := os.Open(name); err == nil {
			h := sha256.New()
			io.Copy(h, f)
			f.Close()
			var out2 OutputID
			h.Sum(out2[:0])
			if out == out2 {
				if c.trimBuildCache {
					c.markUsed(entry) // rewriting counts as a use
				}
				return nil
			}
		}
		// Hash did not match. Fall through and rewrite file.
	}

	// Copy file to cache directory.
	mode := os.O_RDWR | os.O_CREATE
	if err == nil && info.Size() > size { // shouldn't happen but fix in case
		mode |= os.O_TRUNC
	}
	f, err := os.OpenFile(name, mode, perm)
	if err != nil {
		if base.IsETXTBSY(err) {
			// This file is being used by an executable. It must have
			// already been written by another go process and then run.
			// return without an error.
			if c.trimBuildCache {
				c.markUsed(entry)
			}
			return nil
		}
		return err
	}
	defer f.Close()
	if size == 0 {
		// File now exists with correct size.
		// Only one possible zero-length file, so contents are OK too.
		// Early return here makes sure there's a "last byte" for code below.
		if c.trimBuildCache {
			if err := f.Close(); err != nil {
				os.Remove(name)
				return err
			}
			now := c.writeTime()
			os.Chtimes(name, now, now)
			if executableName != "" {
				// trimSubdir and markUsed check the entry directory's mtime.
				os.Chtimes(entry, now, now)
			}
		}
		return nil
	}

	// From here on, if any of the I/O writing the file fails,
	// we make a best-effort attempt to truncate the file f
	// before returning, to avoid leaving bad bytes in the file.

	// Copy file to f, but also into h to double-check hash.
	if _, err := file.Seek(0, 0); err != nil {
		f.Truncate(0)
		return err
	}
	h := sha256.New()
	w := io.MultiWriter(f, h)
	if _, err := io.CopyN(w, file, size-1); err != nil {
		f.Truncate(0)
		return err
	}
	// Check last byte before writing it; writing it will make the size match
	// what other processes expect to find and might cause them to start
	// using the file.
	buf := make([]byte, 1)
	if _, err := file.Read(buf); err != nil {
		f.Truncate(0)
		return err
	}
	h.Write(buf)
	sum := h.Sum(nil)
	if !bytes.Equal(sum, out[:]) {
		f.Truncate(0)
		return fmt.Errorf("file content changed underfoot")
	}

	// Commit cache file entry.
	if _, err := f.Write(buf); err != nil {
		f.Truncate(0)
		return err
	}
	if err := f.Close(); err != nil {
		// Data might not have been written,
		// but file may look like it is the right size.
		// To be extra careful, remove cached file.
		os.Remove(name)
		return err
	}
	now := c.writeTime()
	os.Chtimes(name, now, now) // sets young generation mark (and used by tests)
	if c.trimBuildCache && executableName != "" {
		// trimSubdir and markUsed check the entry directory's mtime.
		os.Chtimes(entry, now, now)
	}
	c.addBytes(size - oldSize)

	return nil
}

// FuzzDir returns a subdirectory within the cache for storing fuzzing data.
// The subdirectory may not exist.
//
// This directory is managed by the internal/fuzz package. Files in this
// directory aren't removed by the 'go clean -cache' command or by Trim.
// They may be removed with 'go clean -fuzzcache'.
//
// TODO(#48526): make Trim remove unused files from this directory.
func (c *DiskCache) FuzzDir() string {
	return filepath.Join(c.dir, "fuzz")
}
