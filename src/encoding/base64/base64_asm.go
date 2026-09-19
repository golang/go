// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64 && !purego

package base64

//go:noescape
func encodeChunk(encode *[64]byte, dst, src *byte, n int)
