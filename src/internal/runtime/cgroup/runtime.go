// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgroup

import (
	_ "unsafe" // for linkname
)

// Functions below pushed from runtime.

//go:linkname throw
func throw(s string)
