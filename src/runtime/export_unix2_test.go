// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !linux

package runtime

// for linux close-on-exec implemented in runtime/internal/syscall
var Closeonexec = closeonexec
