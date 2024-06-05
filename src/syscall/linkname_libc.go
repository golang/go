// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || (openbsd && !mips64) || solaris

package syscall

import _ "unsafe"

// used by internal/poll
//go:linkname writev
