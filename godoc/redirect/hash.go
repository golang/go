// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides a compact encoding of
// a map of Mercurial hashes to Git hashes.

package redirect

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
)

// hashMap is a map of Mercurial hashes to Git hashes.
type hashMap struct {
	file    *os.File
	entries int
}

// newHashMap takes a file handle that contains a map of Mercurial to Git
// hashes. The file should be a sequence of pairs of little-endian encoded
// uint32s, representing a hgHash and a gitHash respectively.
// The sequence must be sorted by hgHash.
// The file must remain open for as long as the returned hashMap is used.
func newHashMap(f *os.File) (*hashMap, error) {
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return &hashMap{file: f, entries: int(fi.Size() / 8)}, nil
}

// Lookup finds an hgHash in the map that matches the given prefix, and returns
// its corresponding gitHash. The prefix must be at least 8 characters long.
func (m *hashMap) Lookup(s string) gitHash {
	if m == nil {
		return 0
	}
	hg, err := hgHashFromString(s)
	if err != nil {
		return 0
	}
	var git gitHash
	b := make([]byte, 8)
	sort.Search(m.entries, func(i int) bool {
		n, err := m.file.ReadAt(b, int64(i*8))
		if err != nil {
			panic(err)
		}
		if n != 8 {
			panic(io.ErrUnexpectedEOF)
		}
		v := hgHash(binary.LittleEndian.Uint32(b[:4]))
		if v == hg {
			git = gitHash(binary.LittleEndian.Uint32(b[4:]))
		}
		return v >= hg
	})
	return git
}

// hgHash represents the lower (leftmost) 32 bits of a Mercurial hash.
type hgHash uint32

func (h hgHash) String() string {
	return intToHash(int64(h))
}

func hgHashFromString(s string) (hgHash, error) {
	if len(s) < 8 {
		return 0, fmt.Errorf("string too small: len(s) = %d", len(s))
	}
	hash := s[:8]
	i, err := strconv.ParseInt(hash, 16, 64)
	if err != nil {
		return 0, err
	}
	return hgHash(i), nil
}

// gitHash represents the leftmost 28 bits of a Git hash in its upper 28 bits,
// and it encodes hash's repository in the lower 4  bits.
type gitHash uint32

func (h gitHash) Hash() string {
	return intToHash(int64(h))[:7]
}

func (h gitHash) Repo() string {
	return repo(h & 0xF).String()
}

func intToHash(i int64) string {
	s := strconv.FormatInt(i, 16)
	if len(s) < 8 {
		s = strings.Repeat("0", 8-len(s)) + s
	}
	return s
}

// repo represents a Go Git repository.
type repo byte

const (
	repoGo repo = iota
	repoBlog
	repoCrypto
	repoExp
	repoImage
	repoMobile
	repoNet
	repoSys
	repoTalks
	repoText
	repoTools
)

func (r repo) String() string {
	return map[repo]string{
		repoGo:     "go",
		repoBlog:   "blog",
		repoCrypto: "crypto",
		repoExp:    "exp",
		repoImage:  "image",
		repoMobile: "mobile",
		repoNet:    "net",
		repoSys:    "sys",
		repoTalks:  "talks",
		repoText:   "text",
		repoTools:  "tools",
	}[r]
}
