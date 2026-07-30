// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

func sdivisible16(c int16) SdivisibleData { return Sdivisible(16, int64(c)) }

func sdivisible32(c int32) SdivisibleData { return Sdivisible(32, int64(c)) }

func sdivisible64(c int64) SdivisibleData { return Sdivisible(64, c) }

func sdivisible8(c int8) SdivisibleData { return Sdivisible(8, int64(c)) }

func sdivisibleOK32(c int32) bool { return SdivisibleOK(32, int64(c)) }

func sdivisibleOK64(c int64) bool { return SdivisibleOK(64, c) }

func udivisible16(c int16) UdivisibleData { return Udivisible(16, int64(c)) }

func udivisible32(c int32) UdivisibleData { return Udivisible(32, int64(c)) }

func udivisible64(c int64) UdivisibleData { return Udivisible(64, c) }

func udivisible8(c int8) UdivisibleData { return Udivisible(8, int64(c)) }

func udivisibleOK32(c int32) bool { return UdivisibleOK(32, int64(c)) }

func udivisibleOK64(c int64) bool { return UdivisibleOK(64, c) }
