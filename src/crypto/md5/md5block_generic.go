// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !386 && !arm && !ppc64le && !ppc64 && !s390x && !arm64) || purego

package md5

const haveAsm = false

func block(dig *digest, p []byte) {
	blockGeneric(dig, p)
}
