// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "internal/syscall/windows"

func supportsUnixSocket() bool {
	return windows.SupportUnixSocket()
}
