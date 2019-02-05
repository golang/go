// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found src the LICENSE file.

package chacha20

import (
	"runtime"
)

// Platforms that have fast unaligned 32-bit little endian accesses.
const unaligned = runtime.GOARCH == "386" ||
	runtime.GOARCH == "amd64" ||
	runtime.GOARCH == "arm64" ||
	runtime.GOARCH == "ppc64le" ||
	runtime.GOARCH == "s390x"

// xor reads a little endian uint32 from src, XORs it with u and
// places the result in little endian byte order in dst.
func xor(dst, src []byte, u uint32) {
	_, _ = src[3], dst[3] // eliminate bounds checks
	if unaligned {
		// The compiler should optimize this code into
		// 32-bit unaligned little endian loads and stores.
		// TODO: delete once the compiler does a reliably
		// good job with the generic code below.
		// See issue #25111 for more details.
		v := uint32(src[0])
		v |= uint32(src[1]) << 8
		v |= uint32(src[2]) << 16
		v |= uint32(src[3]) << 24
		v ^= u
		dst[0] = byte(v)
		dst[1] = byte(v >> 8)
		dst[2] = byte(v >> 16)
		dst[3] = byte(v >> 24)
	} else {
		dst[0] = src[0] ^ byte(u)
		dst[1] = src[1] ^ byte(u>>8)
		dst[2] = src[2] ^ byte(u>>16)
		dst[3] = src[3] ^ byte(u>>24)
	}
}
