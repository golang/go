// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go declarations for malloc.
// The actual functions are written in C
// and part of the runtime library.

package malloc

type Stats struct {
	Alloc	uint64;
	Sys	uint64;
	Stacks	uint64;
	InusePages	uint64;
	NextGC	uint64;
	EnableGC	bool;
}

func Alloc(uintptr) *byte
func Free(*byte)
func GetStats() *Stats
func Lookup(*byte) (*byte, uintptr)
func GC()
