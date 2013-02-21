// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows plan9

package interp

import (
	"exp/ssa"
)

func ext۰syscall۰Kill(fn *ssa.Function, args []value) value {
	panic("syscall.Kill not yet implemented")
}

func ext۰syscall۰Write(fn *ssa.Function, args []value) value {
	panic("syscall.Write not yet implemented")
}
