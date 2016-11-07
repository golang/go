// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,!gccgo,!appengine

package cipherhw

// hasHWSupport reports whether the AES-128, AES-192 and AES-256 cipher message
// (KM) function codes are supported. Note that this function is expensive.
// defined in asm_s390x.s
func hasHWSupport() bool

var hwSupport = hasHWSupport()

func AESGCMSupport() bool {
	return hwSupport
}
