// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package interp

import (
	"syscall"

	"code.google.com/p/go.tools/ssa"
)

func ext۰syscall۰Close(fn *ssa.Function, args []value) value {
	panic("syscall.Close not yet implemented")
}
func ext۰syscall۰Fstat(fn *ssa.Function, args []value) value {
	panic("syscall.Fstat not yet implemented")
}
func ext۰syscall۰Kill(fn *ssa.Function, args []value) value {
	panic("syscall.Kill not yet implemented")
}
func ext۰syscall۰Lstat(fn *ssa.Function, args []value) value {
	panic("syscall.Lstat not yet implemented")
}
func ext۰syscall۰Open(fn *ssa.Function, args []value) value {
	panic("syscall.Open not yet implemented")
}
func ext۰syscall۰ParseDirent(fn *ssa.Function, args []value) value {
	panic("syscall.ParseDirent not yet implemented")
}
func ext۰syscall۰Read(fn *ssa.Function, args []value) value {
	panic("syscall.Read not yet implemented")
}
func ext۰syscall۰ReadDirent(fn *ssa.Function, args []value) value {
	panic("syscall.ReadDirent not yet implemented")
}
func ext۰syscall۰Stat(fn *ssa.Function, args []value) value {
	panic("syscall.Stat not yet implemented")
}
func ext۰syscall۰Write(fn *ssa.Function, args []value) value {
	// func Write(fd int, p []byte) (n int, err error)
	n, err := write(args[0].(int), valueToBytes(args[1]))
	return tuple{n, wrapError(err)}
}
