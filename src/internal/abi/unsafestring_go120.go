// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20
// +build go1.20

package abi

import "unsafe"

func unsafeStringFor(b *byte, l int) string {
	return unsafe.String(b, l)
}

func unsafeSliceFor(b *byte, l int) []byte {
	return unsafe.Slice(b, l)
}
