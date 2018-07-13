// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build !386,!amd64,!arm,!arm64

package runtime

func vdsoauxv(tag, val uintptr) {
}
