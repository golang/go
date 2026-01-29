// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux

package runtime

import _ "unsafe" // for go:linkname

//go:linkname syscall_runtimeClearenv syscall.runtimeClearenv
func syscall_runtimeClearenv(env map[string]int) {
	// The system doesn't have clearenv(3) so emulate it by unsetting all of
	// the variables manually.
	for k := range env {
		syscall_runtimeUnsetenv(k)
	}
}
