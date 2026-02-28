// errorcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gofrontend used to accept this.

package p

func F2(a int32) bool {
	return a == C	// ERROR "invalid|incompatible"
}

const C = uint32(34)
