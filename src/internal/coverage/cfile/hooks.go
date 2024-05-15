// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfile

import _ "unsafe"

// InitHook is invoked from the main package "init" routine in
// programs built with "-cover". This function is intended to be
// called only by the compiler (via runtime/coverage.initHook).
//
// If 'istest' is false, it indicates we're building a regular program
// ("go build -cover ..."), in which case we immediately try to write
// out the meta-data file, and register emitCounterData as an exit
// hook.
//
// If 'istest' is true (indicating that the program in question is a
// Go test binary), then we tentatively queue up both emitMetaData and
// emitCounterData as exit hooks. In the normal case (e.g. regular "go
// test -cover" run) the testmain.go boilerplate will run at the end
// of the test, write out the coverage percentage, and then invoke
// MarkProfileEmitted to indicate that no more work needs to be
// done. If however that call is never made, this is a sign that the
// test binary is being used as a replacement binary for the tool
// being tested, hence we do want to run exit hooks when the program
// terminates.
func InitHook(istest bool) {
	// Note: hooks are run in reverse registration order, so
	// register the counter data hook before the meta-data hook
	// (in the case where two hooks are needed).
	runOnNonZeroExit := true
	runtime_addExitHook(emitCounterData, runOnNonZeroExit)
	if istest {
		runtime_addExitHook(emitMetaData, runOnNonZeroExit)
	} else {
		emitMetaData()
	}
}

//go:linkname runtime_addExitHook runtime.addExitHook
func runtime_addExitHook(f func(), runOnNonZeroExit bool)
