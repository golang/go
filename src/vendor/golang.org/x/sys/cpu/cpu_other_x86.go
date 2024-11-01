// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64p32 || (amd64 && (!darwin || !gc))

package cpu

func darwinSupportsAVX512() bool {
	panic("only implemented for gc && amd64 && darwin")
}
