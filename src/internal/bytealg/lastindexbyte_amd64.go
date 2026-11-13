// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !plan9

package bytealg

//go:noescape
func LastIndexByte(b []byte, c byte) int

//go:noescape
func LastIndexByteString(s string, c byte) int
