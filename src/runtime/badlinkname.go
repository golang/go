// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe"

// As of Go 1.22, the symbols below are found to be pulled via
// linkname in the wild. We provide a push linkname here, to
// keep them accessible with pull linknames.
// This may change in the future. Please do not depend on them
// in new code.

//go:linkname add
//go:linkname atomicwb
//go:linkname callers
//go:linkname chanbuf
//go:linkname cputicks
//go:linkname entersyscallblock
//go:linkname fastexprand
//go:linkname gopanic
//go:linkname gopark
//go:linkname goready
//go:linkname goyield
//go:linkname nilinterhash
//go:linkname noescape
//go:linkname procPin
//go:linkname procUnpin
//go:linkname sched
//go:linkname startTheWorld
//go:linkname stopTheWorld
//go:linkname stringHash
//go:linkname typedmemmove
//go:linkname typedslicecopy
//go:linkname typehash
//go:linkname wakep
