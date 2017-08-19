// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"crypto/sha256"
	"fmt"
	"hash"
	"io"
	"os"
	"sync"
)

var debugHash = os.Getenv("GOCMDDEBUGHASH") == "1"

// HashSize is the number of bytes in a hash.
const HashSize = 32

// A Hash provides access to the canonical hash function used to index the cache.
// The current implementation uses SHA256, but clients must not assume this.
type Hash struct {
	h    hash.Hash
	name string // for debugging
}

// NewHash returns a new Hash.
// The caller is expected to Write data to it and then call Sum.
func NewHash(name string) *Hash {
	h := &Hash{h: sha256.New(), name: name}
	if debugHash {
		fmt.Fprintf(os.Stderr, "HASH[%s]\n", h.name)
	}
	return h
}

// Write writes data to the running hash.
func (h *Hash) Write(b []byte) (int, error) {
	if debugHash {
		fmt.Fprintf(os.Stderr, "HASH[%s]: %q\n", h.name, b)
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
	return out
}

var hashFileCache struct {
	sync.Mutex
	m map[string][HashSize]byte
}

// HashFile returns the hash of the named file.
// It caches repeated lookups for a given file,
// and the cache entry for a file can be initialized
// using SetFileHash.
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
