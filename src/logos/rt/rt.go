// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rt is the loGo/s support package: helpers the compiler's
// lowering calls, so lowered files import only logos/rt (plus unsafe
// where a milestone needs it).
package rt

import (
	"context"
	"errors"
)

// Site records one error-propagation site: the file and line where a ?
// error path recorded the error. See TraceErr and ErrorTrace.
type Site struct {
	File string
	Line int
}

// Cold marks a branch unlikely. The backend lowers it to BranchUnlikely;
// it emits no code and costs nothing at run time.
func Cold() {}

// Join combines two errors, matching errors.Join semantics. The lowering
// keeps a nil result alone and calls Join only for two non-nil errors.
func Join(a, b error) error {
	return errors.Join(a, b)
}

// Context aliases context.Context so lowered code spells the group
// context without importing context itself.
type Context = context.Context
