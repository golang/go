// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// aix, darwin, js/wasm, openbsd and solaris don't implement
// waitid/wait6. netbsd implements wait6, but that is causing test
// failures, see issue #48789.

//go:build aix || darwin || (js && wasm) || netbsd || openbsd || solaris

package os

// blockUntilWaitable attempts to block until a call to p.Wait will
// succeed immediately, and reports whether it has done so.
// It does not actually call p.Wait.
// This version is used on systems that do not implement waitid,
// or where we have not implemented it yet. Note that this is racy:
// a call to Process.Signal can in an extremely unlikely case send a
// signal to the wrong process, see issue #13987.
func (p *Process) blockUntilWaitable() (bool, error) {
	return false, nil
}
