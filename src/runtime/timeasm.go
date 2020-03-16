// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Declarations for operating systems implementing time.now directly in assembly.

// +build windows

package runtime

import _ "unsafe"

//go:linkname time_now time.now
func time_now() (sec int64, nsec int32, mono int64)
