// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (unix && !linux) || wasip1

package os

// rootOpenDirFlags holds additional flags used when opening the
// intermediate directories of a Root operation.
//
// It is zero on platforms without an equivalent of Linux's O_PATH.
const rootOpenDirFlags = 0
