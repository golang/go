// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
)

func (ph *processHandle) closeHandle() {
	syscall.Close(int(ph.handle))
}
