// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64

package crc64

func archFold(crc uint64, tab *Table, p []byte) (uint64, []byte) {
	return crc, p
}
