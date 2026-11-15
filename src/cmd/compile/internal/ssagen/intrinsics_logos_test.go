// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

func init() {
	wantIntrinsics[testIntrinsicKey{"amd64", "runtime", "logosArenaBump"}] = struct{}{}
	wantIntrinsics[testIntrinsicKey{"amd64", "runtime", "logosRegionBump"}] = struct{}{}
}
