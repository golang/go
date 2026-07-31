// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import "cmd/compile/internal/ssa"

func sdivisible16(c int16) ssa.SdivisibleData { return ssa.Sdivisible(16, int64(c)) }

func sdivisible32(c int32) ssa.SdivisibleData { return ssa.Sdivisible(32, int64(c)) }

func sdivisible64(c int64) ssa.SdivisibleData { return ssa.Sdivisible(64, c) }

func sdivisible8(c int8) ssa.SdivisibleData { return ssa.Sdivisible(8, int64(c)) }

func sdivisibleOK32(c int32) bool { return ssa.SdivisibleOK(32, int64(c)) }

func sdivisibleOK64(c int64) bool { return ssa.SdivisibleOK(64, c) }

func udivisible16(c int16) ssa.UdivisibleData { return ssa.Udivisible(16, int64(c)) }

func udivisible32(c int32) ssa.UdivisibleData { return ssa.Udivisible(32, int64(c)) }

func udivisible64(c int64) ssa.UdivisibleData { return ssa.Udivisible(64, c) }

func udivisible8(c int8) ssa.UdivisibleData { return ssa.Udivisible(8, int64(c)) }

func udivisibleOK32(c int32) bool { return ssa.UdivisibleOK(32, int64(c)) }

func udivisibleOK64(c int64) bool { return ssa.UdivisibleOK(64, c) }
