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
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/mmap"
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
	// After a success call to Get, OutputFile(Entry.OutputID) must
	// exist on disk for until Close is called (at the end of the process).
	Get(ActionID) (Entry, error)

	// Put adds an item to the cache.
	//
	// The seeker is only used to seek to the beginning. After a call to Put,
	// the seek position is not guaranteed to be in any particular state.
	//
	// As a special case, if the ReadSeeker is of type noVerifyReadSeeker,
	// the verification from GODEBUG=goverifycache=1 is skipped.
	//
	// After a success call to Get, OutputFile(Entry.OutputID) must
	// exist on disk for until Close is called (at the end of the process).
	Put(ActionID, io.ReadSeeker) (_ OutputID, size int64, _ error)

	// Close is called at the end of the go process. Implementations can do
	// cache cleanup work at this phase, or wait for and report any errors from
	// background cleanup work started earlier. Any cache trimming should in one
	// process should not violate cause the invariants of this interface to be
	// violated in another process. Namely, a cache trim from one process should
	// not delete an ObjectID from disk that was recently Get or Put from
	// another process. As a rule of thumb, don't trim things used in the last
	// day.
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

// A Cache is a package cache, backed by a file system directory tree.
type DiskCache struct {
	dir string
	now func() time.Time
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
		dir: dir,
		now: time.Now,
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
func GetMmap(c Cache, id ActionID) ([]byte, Entry, error) {
	entry, err := c.Get(id)
	if err != nil {
		return nil, entry, err
	}
	md, err := mmap.Mmap(c.OutputFile(entry.OutputID))
	if err != nil {
		return nil, Entry{}, err
	}
	if int64(len(md.Data)) != entry.Size {
		return nil, Entry{}, &entryNotFoundError{Err: errors.New("file incomplete")}
	}
	return md.Data, entry, nil
}

// OutputFile returns the name of the cache file storing output with the given OutputID.
func (c *DiskCache) OutputFile(out OutputID) string {
	file := c.fileName(out, "d")
	isExecutable := c.markUsed(file)
	if isExecutable {
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
const (
	mtimeInterval = 1 * time.Hour
	trimInterval  = 24 * time.Hour
	trimLimit     = 5 * 24 * time.Hour
)

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
// markUsed reports whether the file is a directory (an executable cache entry).
func (c *DiskCache) markUsed(file string) (isExecutable bool) {
	info, err := os.Stat(file)
	if err == nil && c.now().Sub(info.ModTime()) < mtimeInterval {
		return info.IsDir()
	}
	os.Chtimes(file, c.now(), c.now())
	return info.IsDir()
}

func (c *DiskCache) Close() error { return c.Trim() }

// Trim removes old cache entries that are likely not to be reused.
func (c *DiskCache) Trim() error {
	now := c.now()

	// We maintain in dir/trim.txt the time of the last completed cache trim.
	// If the cache has been trimmed recently enough, do nothing.
	// This is the common case.
	// If the trim file is corrupt, detected if the file can't be parsed, or the
	// trim time is too far in the future, attempt the trim anyway. It's possible that
	// the cache was full when the corruption happened. Attempting a trim on
	// an empty cache is cheap, so there wouldn't be a big performance hit in that case.
	if data, err := lockedfile.Read(filepath.Join(c.dir, "trim.txt")); err == nil {
		if t, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64); err == nil {
			lastTrim := time.Unix(t, 0)
			if d := now.Sub(lastTrim); d < trimInterval && d > -mtimeInterval {
				return nil
			}
		}
	}

	// Trim each of the 256 subdirectories.
	// We subtract an additional mtimeInterval
	// to account for the imprecision of our "last used" mtimes.
	cutoff := now.Add(-trimLimit - mtimeInterval)
	for i := 0; i < 256; i++ {
		subdir := filepath.Join(c.dir, fmt.Sprintf("%02x", i))
		c.trimSubdir(subdir, cutoff)
	}

	// Ignore errors from here: if we don't write the complete timestamp, the
	// cache will appear older than it is, and we'll trim it again next time.
	var b bytes.Buffer
	fmt.Fprintf(&b, "%d", now.Unix())
	if err := lockedfile.Write(filepath.Join(c.dir, "trim.txt"), &b, 0o666); err != nil {
		return err
	}

	return nil
}

// trimSubdir trims a single cache subdirectory.
func (c *DiskCache) trimSubdir(subdir string, cutoff time.Time) {
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
		if err == nil && info.ModTime().Before(cutoff) {
			if info.IsDir() { // executable cache entry
				os.RemoveAll(entry)
				continue
			}
			os.Remove(entry)
		}
	}
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
	os.Chtimes(file, c.now(), c.now()) // mainly for tests

	return nil
}

// noVerifyReadSeeker is an io.ReadSeeker wrapper sentinel type
// that says that Cache.Put should skip the verify check
// (from GODEBUG=goverifycache=1).
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
// when GODEBUG=goverifycache=1 is set.
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
	if err == nil && info.Size() == size {
		// Check hash.
		if f, err := os.Open(name); err == nil {
			h := sha256.New()
			io.Copy(h, f)
			f.Close()
			var out2 OutputID
			h.Sum(out2[:0])
			if out == out2 {
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
			return nil
		}
		return err
	}
	defer f.Close()
	if size == 0 {
		// File now exists with correct size.
		// Only one possible zero-length file, so contents are OK too.
		// Early return here makes sure there's a "last byte" for code below.
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
	os.Chtimes(name, c.now(), c.now()) // mainly for tests

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
