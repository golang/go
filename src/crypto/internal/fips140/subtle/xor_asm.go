// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64 || loong64 || mips || mipsle || mips64 || mips64le || ppc64 || ppc64le || riscv64) && !purego

package subtle

//go:noescape
func xorBytes(dst, a, b *byte, n int)
