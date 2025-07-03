// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!386 && !amd64 && !arm && !arm64 && !loong64 && !s390x) || purego

package sha1

func block(dig *digest, p []byte) {
	blockGeneric(dig, p)
}
