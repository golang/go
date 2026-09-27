// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package weak

import "internal/abi"

type M map[uint64]uint64

// Emit a noalg descriptor for [8]uint64.
var Sink any = M{}

//go:noinline
func FoldedFlags() abi.TFlag {
	return abi.TypeFor[[8]uint64]().TFlag
}
