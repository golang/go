// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !linux

package runtime

// sigFromUser reports whether the signal was sent because of a call
// to kill.
//
//go:nosplit
func (c *sigctxt) sigFromUser() bool {
	return c.sigcode() == _SI_USER
}

// sigFromSeccomp reports whether the signal was sent from seccomp.
//
//go:nosplit
func (c *sigctxt) sigFromSeccomp() bool {
	return false
}
