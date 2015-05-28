// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
)

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
type LocalSlot struct {
	Idx int64 // offset in locals area (distance up from SP)
}

func (s *LocalSlot) Name() string {
	return fmt.Sprintf("%d(SP)", s.Idx)
}
