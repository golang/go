// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall

import _ "unsafe"

// As of Go 1.22, the symbols below are found to be pulled via
// linkname in the wild. We provide a push linkname here, to
// keep them accessible with pull linknames.
// This may change in the future. Please do not depend on them
// in new code.

// golang.org/x/sys linknames getsockopt.
// Do not remove or change the type signature.
//
//go:linkname getsockopt

//go:linkname setsockopt
