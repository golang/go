// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/syscall/unix"
)

const rootUnsupportedOpenFlags = unix.O_NOFOLLOW_ANY | unix.O_RESOLVE_BENEATH
