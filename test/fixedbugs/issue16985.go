// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 16985: intrinsified AMD64 atomic ops should clobber flags

package main

import "sync/atomic"

var count uint32

func main() {
	buffer := []byte("T")
	for i := 0; i < len(buffer); {
		atomic.AddUint32(&count, 1)
		_ = buffer[i]
		i++
		i++
	}

	for i := 0; i < len(buffer); {
		atomic.CompareAndSwapUint32(&count, 0, 1)
		_ = buffer[i]
		i++
		i++
	}

	for i := 0; i < len(buffer); {
		atomic.SwapUint32(&count, 1)
		_ = buffer[i]
		i++
		i++
	}
}
