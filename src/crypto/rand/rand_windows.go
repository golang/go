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

func (r *rngReader) Read(b []byte) (int, error) {
	if err := windows.ProcessPrng(b); err != nil {
		return 0, err
	}
	return len(b), nil
}
