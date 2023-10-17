// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows cryptographically secure pseudorandom number
// generator.

package rand

import (
	"internal/syscall/windows"
)

func init() { Reader = &rngReader{} }

type rngReader struct{}

func (r *rngReader) Read(b []byte) (n int, err error) {
	// RtlGenRandom only returns 1<<32-1 bytes at a time. We only read at
	// most 1<<31-1 bytes at a time so that  this works the same on 32-bit
	// and 64-bit systems.
	if err := batched(windows.RtlGenRandom, 1<<31-1)(b); err != nil {
		return 0, err
	}
	return len(b), nil
}
