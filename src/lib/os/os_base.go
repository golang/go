// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// Support types and routines for OS library

export func StringToBytes(b *[]byte, s string) bool {
	if len(s) >= len(b) {
		return false
	}
	for i := 0; i < len(s); i++ {
		b[i] = s[i]
	}
	b[len(s)] = '\000';	// not necessary - memory is zeroed - but be explicit
	return true
}
