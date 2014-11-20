// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Call a Windows function with stdcall conventions,
// and switch to os stack during the call.
func asmstdcall(fn unsafe.Pointer)

func getlasterror() uint32
func setlasterror(err uint32)

// Function to be called by windows CreateThread
// to start new os thread.
func tstart_stdcall(newm *m) uint32

func ctrlhandler(_type uint32) uint32

// TODO(brainman): should not need those
const (
	_NSIG = 65
)
