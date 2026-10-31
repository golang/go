// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package os

import "internal/syscall/unix"

// rootOpenDirFlags holds additional flags used when opening the
// intermediate directories of a Root operation.
//
// On Linux, intermediate directories are opened with O_PATH.
// Creating an entry in a directory requires only write and search
// permission on the directory, not read permission, and an O_PATH
// open requires no access mode on the directory itself. Using O_PATH
// therefore preserves the permission semantics of the underlying
// operations (see issue #81605), instead of additionally requiring
// read permission on every intermediate directory.
//
// O_PATH descriptors support all of the *at operations performed
// relative to intermediate directories.
const rootOpenDirFlags = unix.O_PATH
