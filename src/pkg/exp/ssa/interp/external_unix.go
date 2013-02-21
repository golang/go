// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!plan9

package interp

import (
	"exp/ssa"
	"syscall"
)

func ext۰syscall۰Kill(fn *ssa.Function, args []value) value {
	// We could emulate syscall.Syscall but it's more effort.
	err := syscall.Kill(args[0].(int), syscall.Signal(args[1].(int)))
	err = err // TODO(adonovan): fix: adapt concrete err to interpreted iface (e.g. call interpreted errors.New)
	return iface{}
}

func ext۰syscall۰Write(fn *ssa.Function, args []value) value {
	// We could emulate syscall.Syscall but it's more effort.
	p := args[1].([]value)
	b := make([]byte, 0, len(p))
	for i := range p {
		b = append(b, p[i].(byte))
	}
	n, _ := syscall.Write(args[0].(int), b)
	err := iface{} // TODO(adonovan): fix: adapt concrete err to interpreted iface.
	return tuple{n, err}

}
