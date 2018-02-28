// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!gccgo,!appengine

package cipherhw

// defined in asm_amd64.s
func hasAESNI() bool

// AESGCMSupport returns true if the Go standard library supports AES-GCM in
// hardware.
func AESGCMSupport() bool {
	return hasAESNI()
}
