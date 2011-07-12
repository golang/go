// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

// The file contains the generic version of updateCastagnoli which just calls
// the software implementation.

func updateCastagnoli(crc uint32, p []byte) uint32 {
	return update(crc, castagnoliTable, p)
}
