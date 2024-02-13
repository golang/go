// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !android

package runtime

//go:nosplit
func writeErr(b []byte) {
	if len(b) > 0 {
		writeErrData(&b[0], int32(len(b)))
	}
}
