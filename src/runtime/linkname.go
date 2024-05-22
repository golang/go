// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe"

// used in time and internal/poll
//go:linkname nanotime

// used in internal/godebug and syscall
//go:linkname write

// used in internal/runtime/atomic
//go:linkname goarm

// used by cgo
//go:linkname cgocall
//go:linkname _cgo_panic_internal
//go:linkname cgoAlwaysFalse
//go:linkname cgoUse
//go:linkname cgoCheckPointer
//go:linkname cgoCheckResult
//go:linkname cgoNoCallback
//go:linkname gobytes
//go:linkname gostringn

// used in plugin
//go:linkname doInit

// used in math/bits
//go:linkname overflowError
//go:linkname divideError

// used in runtime/coverage and in tests
//go:linkname addExitHook

// used in tests
//go:linkname extraMInUse
//go:linkname getm
//go:linkname blockevent
//go:linkname haveHighResSleep
//go:linkname blockUntilEmptyFinalizerQueue
//go:linkname lockedOSThread
