// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// pause sets SP to newsp and pauses the execution of Go's WebAssembly
// code until an event is triggered, or call back into Go.
//
// Note: the epilogue of pause pops 8 bytes from the stack, so when
// returning to the host, the SP is newsp+8.
// If we want to set the SP such that when it calls back into Go, the
// Go function appears to be called from pause's caller's caller, then
// call pause with newsp = getcallersp()-16 (another 8 is the return
// PC pushed to the stack).
func pause(newsp uintptr)
