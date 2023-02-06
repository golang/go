// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386

package sys

func TrailingZeros64(x uint64) int
func TrailingZeros32(x uint32) int
func TrailingZeros8(x uint8) int
