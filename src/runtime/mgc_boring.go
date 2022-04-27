// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package runtime

import "unsafe"

//go:linkname boring_registerCache crypto/internal/boring.registerCache
func boring_registerCache(p unsafe.Pointer) {
	boringCaches = append(boringCaches, p)
}
