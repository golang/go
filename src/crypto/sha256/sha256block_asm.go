// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (386 || loong64 || riscv64) && !purego

package sha256

//go:noescape
func block(dig *digest, p []byte)
