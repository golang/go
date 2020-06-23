// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!s390x,!arm64,!ppc64le

package elliptic

var (
	p256 p256Curve
)

func initP256Arch() {
	// Use pure Go implementation.
	p256 = p256Curve{p256Params}
}
