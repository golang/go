// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go declarations for malloc.
// The actual functions are written in C
// and part of the runtime library.

package malloc

export type Stats struct {
	alloc	uint64;
	sys	uint64;
};

export func Alloc(uint64) *byte;
export func Free(*byte);
export func GetStats() *Stats;
export func Lookup(*byte) (*byte, uint64);
