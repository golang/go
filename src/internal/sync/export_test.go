// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"internal/abi"
	"unsafe"
)

// NewBadHashTrieMap creates a new HashTrieMap for the provided key and value
// but with an intentionally bad hash function.
func NewBadHashTrieMap[K, V comparable]() *HashTrieMap[K, V] {
	// Stub out the good hash function with a terrible one.
	// Everything should still work as expected.
	var m HashTrieMap[K, V]
	m.init()
	m.keyHash = func(_ unsafe.Pointer, _ uintptr) uintptr {
		return 0
	}
	return &m
}

// NewTruncHashTrieMap creates a new HashTrieMap for the provided key and value
// but with an intentionally bad hash function.
func NewTruncHashTrieMap[K, V comparable]() *HashTrieMap[K, V] {
	// Stub out the good hash function with a terrible one.
	// Everything should still work as expected.
	var m HashTrieMap[K, V]
	var mx map[string]int
	mapType := abi.TypeOf(mx).MapType()
	hasher := mapType.Hasher
	m.keyHash = func(p unsafe.Pointer, n uintptr) uintptr {
		return hasher(p, n) & ((uintptr(1) << 4) - 1)
	}
	return &m
}
