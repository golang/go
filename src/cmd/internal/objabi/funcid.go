// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

// A FuncID identifies particular functions that need to be treated
// specially by the runtime.
// Note that in some situations involving plugins, there may be multiple
// copies of a particular special runtime function.
// Note: this list must match the list in runtime/symtab.go.
type FuncID uint8

const (
	FuncID_normal FuncID = iota // not a special function
	FuncID_runtime_main
	FuncID_goexit
	FuncID_jmpdefer
	FuncID_mcall
	FuncID_morestack
	FuncID_mstart
	FuncID_rt0_go
	FuncID_asmcgocall
	FuncID_sigpanic
	FuncID_runfinq
	FuncID_gcBgMarkWorker
	FuncID_systemstack_switch
	FuncID_systemstack
	FuncID_cgocallback_gofunc
	FuncID_gogo
	FuncID_externalthreadhandler
	FuncID_debugCallV1
)
