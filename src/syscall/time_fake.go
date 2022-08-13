// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build faketime

package syscall

import "unsafe"

const faketime = true

// When faketime is enabled, we redirect writes to FDs 1 and 2 through
// the runtime's write function, since that adds the framing that
// reports the emulated time.

//go:linkname runtimeWrite runtime.write
func runtimeWrite(fd uintptr, p unsafe.Pointer, n int32) int32

func faketimeWrite(fd int, p []byte) int {
	var pp *byte
	if len(p) > 0 {
		pp = &p[0]
	}
	return int(runtimeWrite(uintptr(fd), unsafe.Pointer(pp), int32(len(p))))
}
