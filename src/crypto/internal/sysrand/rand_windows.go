// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sysrand

import "internal/syscall/windows"

func read(b []byte) error {
	return windows.ProcessPrng(b)
}
