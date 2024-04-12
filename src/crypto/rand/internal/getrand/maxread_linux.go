// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package getrand

// Per the manpage:
//
//	When reading from the urandom source, a maximum of 33554431 bytes
//	is returned by a single call to getrandom() on systems where int
//	has a size of 32 bits.
const maxGetRandomRead = (1 << 25) - 1
