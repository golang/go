// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.synctest

package synctest

import "internal/synctest"

// Run is deprecated.
//
// Deprecated: Use Test instead.
func Run(f func()) {
	synctest.Run(f)
}
