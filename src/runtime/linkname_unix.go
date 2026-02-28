// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime

import _ "unsafe"

// used in internal/syscall/unix
//go:linkname fcntl
