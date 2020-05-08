// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"hash"
	"io"
	"os"
	"runtime"
	"sync"
)

var debugHash = false // set when GODEBUG=gocachehash=1

// HashSize is the number of bytes in a hash.
const HashSize = 32

// A Hash provides access to the canonical hash function used to index the cache.
// The current implementation uses salted SHA256, but clients must not assume this.
type Hash struct {
	h    hash.Hash
	name string        // for debugging
	buf  *bytes.Buffer // for verify
}

// hashSalt is a salt string added to the beginning of every hash
// created by NewHash. Using the Go version makes sure that different
// versions of the go command (or even different Git commits during
// work on the development branch) do not address the same cache
// entries, so that a bug in one version does not affect the execution
// of other versions. This salt will result in additional ActionID files
// in the cache, but not additional copies of the large output files,
// which are still addressed by unsalted SHA256.
var hashSalt = []byte(runtime.Version())

// Subkey returns an action ID corresponding to mixing a parent
// action ID with a string description of the subkey.
func Subkey(parent ActionID, desc string) ActionID {
	h := sha256.New()
	h.Write([]byte("subkey:"))
	h.Write(parent[:])
	h.Write([]byte(desc))
	var out ActionID
	h.Sum(out[:0])
	if debugHash {
		fmt.Fprintf(os.Stderr, "HASH subkey %x %q = %x\n", parent, desc, out)
	}
	if verify {
		hashDebug.Lock()
		hashDebug.m[out] = fmt.Sprintf("subkey %x %q", parent, desc)
		hashDebug.Unlock()
	}
	return out
}

// NewHash returns a new Hash.
// The caller is expected to Write data to it and then call Sum.
func NewHash(name string) *Hash {
	h := &Hash{h: sha256.New(), name: name}
	if debugHash {
		fmt.Fprintf(os.Stderr, "HASH[%s]\n", h.name)
	}
	h.Write(hashSalt)
	if verify {
		h.buf = new(bytes.Buffer)
	}
	return h
}

// Write writes data to the running hash.
func (h *Hash) Write(b []byte) (int, error) {
	if debugHash {
		fmt.Fprintf(os.Stderr, "HASH[%s]: %q\n", h.name, b)
	}
	if h.buf != nil {
		h.buf.Write(b)
	}
	return h.h.Write(b)
}

// Sum returns the hash of the data written previously.
func (h *Hash) Sum() [HashSize]byte {
	var out [HashSize]byte
	h.h.Sum(out[:0])
	if debugHash {
		fmt.Fprintf(os.Stderr, "HASH[%s]: %x\n", h.name, out)
	}
	if h.buf != nil {
		hashDebug.Lock()
		if hashDebug.m == nil {
			hashDebug.m = make(map[[HashSize]byte]string)
		}
		hashDebug.m[out] = h.buf.String()
		hashDebug.Unlock()
	}
	return out
}

// In GODEBUG=gocacheverify=1 mode,
// hashDebug holds the input to every computed hash ID,
// so that we can work backward from the ID involved in a
// cache entry mismatch to a description of what should be there.
var hashDebug struct {
	sync.Mutex
	m map[[HashSize]byte]string
}

// reverseHash returns the input used to compute the hash id.
func reverseHash(id [HashSize]byte) string {
	hashDebug.Lock()
	s := hashDebug.m[id]
	hashDebug.Unlock()
	return s
}

var hashFileCache struct {
	sync.Mutex
	m map[string][HashSize]byte
}

// FileHash returns the hash of the named file.
// It caches repeated lookups for a given file,
// and the cache entry for a file can be initialized
// using SetFileHash.
// The hash used by FileHash is not the same as
// the hash used by NewHash.
func FileHash(file string) ([HashSize]byte, error) {
	hashFileCache.Lock()
	out, ok := hashFileCache.m[file]
	hashFileCache.Unlock()

	if ok {
		return out, nil
	}

	h := sha256.New()
	f, err := os.Open(file)
	if err != nil {
		if debugHash {
			fmt.Fprintf(os.Stderr, "HASH %s: %v\n", file, err)
		}
		return [HashSize]byte{}, err
	}
	_, err = io.Copy(h, f)
	f.Close()
	if err != nil {
		if debugHash {
			fmt.Fprintf(os.Stderr, "HASH %s: %v\n", file, err)
		}
		return [HashSize]byte{}, err
	}
	h.Sum(out[:0])
	if debugHash {
		fmt.Fprintf(os.Stderr, "HASH %s: %x\n", file, out)
	}

	SetFileHash(file, out)
	return out, nil
}

// SetFileHash sets the hash returned by FileHash for file.
func SetFileHash(file string, sum [HashSize]byte) {
	hashFileCache.Lock()
	if hashFileCache.m == nil {
		hashFileCache.m = make(map[string][HashSize]byte)
	}
	hashFileCache.m[file] = sum
	hashFileCache.Unlock()
}
