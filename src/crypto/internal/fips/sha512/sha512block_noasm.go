// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !arm64 && !loong64 && !ppc64 && !ppc64le && !riscv64 && !s390x) || purego

package sha512

func block(dig *Digest, p []byte) {
	blockGeneric(dig, p)
}
