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
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
)

// An ActionID is a cache action key, the hash of a complete description of a
// repeatable computation (command line, environment variables,
// input file contents, executable contents).
type ActionID [HashSize]byte

// An OutputID is a cache output key, the hash of an output of a computation.
type OutputID [HashSize]byte

// A Cache is a package cache, backed by a file system directory tree.
type Cache struct {
	dir string
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
//
func Open(dir string) (*Cache, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, &os.PathError{Op: "open", Path: dir, Err: fmt.Errorf("not a directory")}
	}
	for i := 0; i < 256; i++ {
		name := filepath.Join(dir, fmt.Sprintf("%02x", i))
		if err := os.MkdirAll(name, 0777); err != nil {
			return nil, err
		}
	}
	c := &Cache{dir: dir}
	return c, nil
}

// fileName returns the name of the file corresponding to the given id.
func (c *Cache) fileName(id [HashSize]byte, key string) string {
	return filepath.Join(c.dir, fmt.Sprintf("%02x", id[0]), fmt.Sprintf("%x", id)+"-"+key)
}

var errMissing = errors.New("cache entry not found")

const (
	// action entry file is "v1 <hex id> <hex out> <decimal size space-padded to 20 bytes>\n"
	hexSize   = HashSize * 2
	entrySize = 2 + 1 + hexSize + 1 + hexSize + 1 + 20 + 1
)

// Get looks up the action ID in the cache,
// returning the corresponding output ID and file size, if any.
// Note that finding an output ID does not guarantee that the
// saved file for that output ID is still available.
func (c *Cache) Get(id ActionID) (OutputID, int64, error) {
	missing := func() (OutputID, int64, error) {
		// TODO: log miss
		return OutputID{}, 0, errMissing
	}
	f, err := os.Open(c.fileName(id, "a"))
	if err != nil {
		return missing()
	}
	defer f.Close()
	entry := make([]byte, entrySize+1) // +1 to detect whether f is too long
	if n, err := io.ReadFull(f, entry); n != entrySize || err != io.ErrUnexpectedEOF {
		return missing()
	}
	if entry[0] != 'v' || entry[1] != '1' || entry[2] != ' ' || entry[3+hexSize] != ' ' || entry[3+hexSize+1+64] != ' ' || entry[entrySize-1] != '\n' {
		return missing()
	}
	eid, eout, esize := entry[3:3+hexSize], entry[3+hexSize+1:3+hexSize+1+hexSize], entry[3+hexSize+1+hexSize+1:entrySize-1]
	var buf [HashSize]byte
	if _, err := hex.Decode(buf[:], eid); err != nil || buf != id {
		return missing()
	}
	if _, err := hex.Decode(buf[:], eout); err != nil {
		return missing()
	}
	i := 0
	for i < len(esize) && esize[i] == ' ' {
		i++
	}
	size, err := strconv.ParseInt(string(esize[i:]), 10, 64)
	if err != nil || size < 0 {
		return missing()
	}

	// TODO: Update modtime of f to give a signal about recently used?
	// TODO: log hit

	return buf, size, nil
}

// OutputFile returns the name of the cache file storing output with the given OutputID.
func (c *Cache) OutputFile(out OutputID) string {
	return c.fileName(out, "d")
}

// putIndexEntry adds an entry to the cache recording that executing the action
// with the given id produces an output with the given output id (hash) and size.
func (c *Cache) putIndexEntry(id ActionID, out OutputID, size int64) error {
	// Note: We expect that for one reason or another it may happen
	// that repeating an action produces a different output hash
	// (for example, if the output contains a time stamp or temp dir name).
	// While not ideal, this is also not a correctness problem, so we
	// don't make a big deal about it. In particular, we leave the action
	// cache entries writable specifically so that they can be overwritten.
	entry := []byte(fmt.Sprintf("v1 %x %x %20d\n", id, out, size))
	return ioutil.WriteFile(c.fileName(id, "a"), entry, 0666)
}

// Put stores the given output in the cache as the output for the action ID.
// It may read file twice. The content of file must not change between the two passes.
func (c *Cache) Put(id ActionID, file io.ReadSeeker) (OutputID, int64, error) {
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
	if err := c.copyFile(file, out, size); err != nil {
		return out, size, err
	}

	// Add to cache index.
	// TODO: log put
	return out, size, c.putIndexEntry(id, out, size)
}

// copyFile copies file into the cache, expecting it to have the given
// output ID and size, if that file is not present already.
func (c *Cache) copyFile(file io.ReadSeeker, out OutputID, size int64) error {
	name := c.fileName(out, "d")
	info, err := os.Stat(name)
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
	f, err := os.OpenFile(name, mode, 0666)
	if err != nil {
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

	return nil
}
