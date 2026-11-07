// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

// Cooperative frame contract
//
// This API supports code generated for the running Go toolchain and architecture,
// not a persistent binary format or an FFI for opaque blocking code. Query ABI
// descriptions again when recompiling with another toolchain. Publish code,
// relocations and every safepoint map before publishing function entries; retire
// a region only after no execution, stack frame or callable reference can use it.
//
// GC and stack copying stop JIT invocations at described cooperative safepoints.
// Every live Go pointer must be in a mapped stack slot or independently rooted
// through ordinary Go references. Pointer registers must be spilled before a Go
// call can stop the goroutine. Integer and floating-point payloads must not be
// marked as pointers. Stack copying updates mapped stack pointers and saved frame
// pointers; addresses disguised as integers are not relocated.
//
// Addressable JIT stack objects, including nonescaping closure contexts, are
// supported by describing all their live pointer fields directly in the owning
// frame's map. They are roots for that safepoint regardless of object reachability.
// There are no reachability-dependent JIT stack-object descriptors. Slots must
// contain valid initialized pointers or nil while mapped, and must be removed
// from maps or cleared before reuse. Such objects cannot escape the invocation
// or be retained by another goroutine. Go callbacks may borrow them synchronously.
// A mapped pointer into an ordinary Go stack object uses the runtime's normal
// stack-object reachability processing, including conservative-root provenance.
//
// Arbitrary JIT PCs, including partial prologues and write barriers, are not
// asynchronous preemption points. The runtime's unknown-function checks reject
// asynchronous preemption and debugger call injection there. Signal tracebacks
// may be incomplete at PCs without a published unwind recipe; signals do not
// turn such PCs into GC safepoints. If an injected Go register-save frame is
// followed by a described JIT frame, conservative scanning is propagated for
// that immediate parent and for stack objects reached from conservative roots.
// Synchronous hardware faults at generated PCs are not specified as recoverable.
//
// Every unbounded generated loop must promptly poll Preempt and reach Yield or
// an equivalent actual scheduling boundary through a described Go call. Calls
// to arbitrary nosplit Go leaves and returning alone do not guarantee yielding.
// Blocking foreign calls still need appropriate scheduler transitions such as
// cgo. No JIT-owned Go callbacks run during pointer scanning or relocation.
// Ordinary Go functions called by JIT code may allocate and block normally.
// Describe is different: it may run during runtime stack formatting, must not
// allocate or grow its stack, and must not call diagnostic snapshot functions.
//
// Globals and pointer constants belong in typed, normally reachable Go objects.
// Executable mappings and embedded uintptr constants are not roots. Generated
// pointer writes to Go heap objects must implement Go's write barriers or call
// typed Go helpers; freshly allocated objects are not exempt. Metadata validation
// checks structural errors only. It cannot discover a missing root, an incorrect
// live interval, or instructions which violate this contract.

package jit
