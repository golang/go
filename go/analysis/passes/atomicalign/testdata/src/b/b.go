// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !arm,!386

package testdata

import (
	"sync/atomic"
)

func nonAffectedArchs() {
	var s struct {
		_ bool
		a uint64
	}
	atomic.SwapUint64(&s.a, 9) // ok on 64-bit architectures
}
