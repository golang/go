// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rand implements a cryptographically secure
// pseudorandom number generator.
package rand

import (
	"io"
	"os"
)

// Reader is a global, shared instance of a cryptographically
// strong pseudo-random generator.
// On Unix-like systems, Reader reads from /dev/urandom.
// On Windows systems, Reader uses the CryptGenRandom API.
var Reader io.Reader

// Read is a helper function that calls Reader.Read.
func Read(b []byte) (n int, err os.Error) { return Reader.Read(b) }
