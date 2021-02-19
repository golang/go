// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows cryptographically secure pseudorandom number
// generator.

package rand

import (
	"internal/syscall/windows"
	"os"
)

func init() { Reader = &rngReader{} }

type rngReader struct{}

func (r *rngReader) Read(b []byte) (n int, err error) {
	// RtlGenRandom only accepts 2**32-1 bytes at a time, so truncate.
	inputLen := uint32(len(b))

	if inputLen == 0 {
		return 0, nil
	}

	err = windows.RtlGenRandom(b)
	if err != nil {
		return 0, os.NewSyscallError("RtlGenRandom", err)
	}
	return int(inputLen), nil
}
