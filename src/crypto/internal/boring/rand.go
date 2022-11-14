// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto && linux && (amd64 || arm64) && !android && !cmd_go_bootstrap && !msan

package boring

// #include "goboringcrypto.h"
import "C"
import "unsafe"

type randReader int

func (randReader) Read(b []byte) (int, error) {
	// Note: RAND_bytes should never fail; the return value exists only for historical reasons.
	// We check it even so.
	if len(b) > 0 && C._goboringcrypto_RAND_bytes((*C.uint8_t)(unsafe.Pointer(&b[0])), C.size_t(len(b))) == 0 {
		return 0, fail("RAND_bytes")
	}
	return len(b), nil
}

const RandReader = randReader(0)
