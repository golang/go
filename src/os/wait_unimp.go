// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly js,wasm netbsd openbsd solaris

package os

// blockUntilWaitable attempts to block until a call to p.Wait will
// succeed immediately, and reports whether it has done so.
// It does not actually call p.Wait.
// This version is used on systems that do not implement waitid,
// or where we have not implemented it yet.
func (p *Process) blockUntilWaitable() (bool, error) {
	return false, nil
}
