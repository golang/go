// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Placeholder for generated glue to come later
package amd64

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
)

func ssaGenSIMDValue(s *ssagen.State, v *ssa.Value) bool {
	switch v.Op {
	default:
		return false
	}
	return true
}
