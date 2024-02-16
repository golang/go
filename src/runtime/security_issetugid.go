// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || illumos || netbsd || openbsd || solaris

package runtime

// secureMode is only ever mutated in schedinit, so we don't need to worry about
// synchronization primitives.
var secureMode bool

func initSecureMode() {
	secureMode = issetugid() == 1
}

func isSecureMode() bool {
	return secureMode
}
