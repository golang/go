// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// contextPC returns the RIP (program counter) register from the context.
func contextPC(r *context) uintptr { return uintptr(r.rip) }

// contextSP returns the RSP (stack pointer) register from the context.
func contextSP(r *context) uintptr { return uintptr(r.rsp) }
