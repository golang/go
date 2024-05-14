// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package syscall

import _ "unsafe"

// used by internal/syscall/unix
//go:linkname ioctlPtr

// used by x/net/route
//go:linkname sysctl
