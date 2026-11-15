// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || 386 || ppc64le || s390x

package rt

import "unsafe"

func viewAsAddr[T any](b []byte) *T {
	return (*T)(unsafe.Pointer(unsafe.SliceData(b)))
}
