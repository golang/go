// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gccgo

package build

import "runtime"

// getToolDir returns the default value of ToolDir.
func getToolDir() string {
	return envOr("GCCGOTOOLDIR", runtime.GCCGOTOOLDIR)
}
