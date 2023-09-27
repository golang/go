// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package os

// supportsCloseOnExec reports whether the platform supports the
// O_CLOEXEC flag.
const supportsCloseOnExec = false
