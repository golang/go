// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

// A place that an ssa variable can reside.
type Location interface {
	Name() string // name to use in assembly templates: %rax, 16(%rsp), ...
}

// A Register is a machine register, like %rax.
// They are numbered densely from 0 (for each architecture).
type Register struct {
	Num  int32
	name string
}

func (r *Register) Name() string {
	return r.name
}

// A LocalSlot is a location in the stack frame.
// It is (possibly a subpiece of) a PPARAM, PPARAMOUT, or PAUTO ONAME node.
type LocalSlot struct {
	N    GCNode // an ONAME *gc.Node representing a variable on the stack
	Type Type   // type of slot
	Off  int64  // offset of slot in N
}

func (s LocalSlot) Name() string {
	if s.Off == 0 {
		return fmt.Sprintf("%s[%s]", s.N, s.Type)
	}
	return fmt.Sprintf("%s+%d[%s]", s.N, s.Off, s.Type)
}
