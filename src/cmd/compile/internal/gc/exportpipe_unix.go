// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package gc

import (
	"os"
	"syscall"
)

// Use O_NONBLOCK so opening a FIFO fails immediately with ENXIO instead of
// hanging indefinitely if the parent build tool exited without unlinking it.
const exportPipeOpenFlags = os.O_WRONLY | syscall.O_NONBLOCK
