// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix_test

// stringsFromByteSlice converts a sequence of attributes to a []string.
// On Darwin, each entry is a NULL-terminated string.
func stringsFromByteSlice(buf []byte) []string {
	var result []string
	off := 0
	for i, b := range buf {
		if b == 0 {
			result = append(result, string(buf[off:i]))
			off = i + 1
		}
	}
	return result
}
