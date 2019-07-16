// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386

package sys

func Ctz64(x uint64) int
func Ctz32(x uint32) int
func Ctz8(x uint8) int
func Bswap64(x uint64) uint64
func Bswap32(x uint32) uint32
