// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import "cmd/internal/sys"

// logosRegisterCold registers the logos/rt.Cold intrinsic on every
// architecture (ssa_logos.go), so the expected table of TestIntrinsics lists
// it for each of them.
func init() {
	for _, arch := range sys.Archs {
		wantIntrinsics[testIntrinsicKey{arch.Name, "logos/rt", "Cold"}] = struct{}{}
	}
}
