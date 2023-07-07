// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// aix, darwin, js/wasm, openbsd, solaris and wasip1/wasm don't implement
// waitid/wait6.

//go:build aix || darwin || (js && wasm) || openbsd || solaris || wasip1

package syscall

// blockUntilWaitable attempts to block until a call to Wait4 will
// succeed immediately, and reports whether it has done so.
// It does not actually call Wait4.
// This version is used on systems that do not implement waitid,
// or where we have not implemented it yet. Note that this is racy:
// a call to Process.Signal can in an extremely unlikely case send a
// signal to the wrong process, see issue #13987.
func blockUntilWaitable(pid int) (int, error) {
	return 0, nil
}
