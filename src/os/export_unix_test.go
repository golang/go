// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package os

var SplitPath = splitPath

// LockSignal blocks calls to p.Signal that use the PID, rather than a
// process handle, after they have checked the PID and before they send
// the signal. It returns a function that unblocks them.
func (p *Process) LockSignal() (unlock func()) {
	p.sigMu.Lock()
	return p.sigMu.Unlock
}
