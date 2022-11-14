// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !android

package runtime

import "unsafe"

func writeErr(b []byte) {
	write(2, unsafe.Pointer(&b[0]), int32(len(b)))
}
