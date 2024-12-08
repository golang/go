// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(linux && (amd64 || arm64 || arm64be || ppc64 || ppc64le || loong64 || s390x))

package runtime

import _ "unsafe"

//go:linkname vgetrandom
func vgetrandom(p []byte, flags uint32) (ret int, supported bool) {
	return -1, false
}

func vgetrandomPutState(state uintptr) {}

func vgetrandomInit() {}
