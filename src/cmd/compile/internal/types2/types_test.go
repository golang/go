// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "sync/atomic"

func init() {
	acceptMethodTypeParams = true
}

// Upon calling ResetId, nextId starts with 1 again.
// It may be called concurrently. This is only needed
// for tests where we may want to have a consistent
// numbering for each individual test case.
func ResetId() { atomic.StoreUint32(&lastId, 0) }
