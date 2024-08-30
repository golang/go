// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!386 && !amd64 && !arm64 && !ppc64 && !ppc64le && !riscv64 && !s390x) || purego

package sha256

func block(dig *digest, p []byte) {
	blockGeneric(dig, p)
}
