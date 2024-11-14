// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import "testing"

// Benchmark multiplication of an integer by various constants.
//
// The comment above each sub-benchmark provides an example of how the
// target multiplication operation might be implemented using shift
// (multiplication by a power of 2), addition and subtraction
// operations. It is platform-dependent whether these transformations
// are actually applied.

var (
	mulSinkI32 int32
	mulSinkI64 int64
	mulSinkU32 uint32
	mulSinkU64 uint64
)

func BenchmarkMulconstI32(b *testing.B) {
	// 3x = 2x + x
	b.Run("3", func { b ->
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 3
		}
		mulSinkI32 = x
	})
	// 5x = 4x + x
	b.Run("5", func { b ->
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 5
		}
		mulSinkI32 = x
	})
	// 12x = 8x + 4x
	b.Run("12", func { b ->
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 12
		}
		mulSinkI32 = x
	})
	// 120x = 128x - 8x
	b.Run("120", func { b ->
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 120
		}
		mulSinkI32 = x
	})
	// -120x = 8x - 120x
	b.Run("-120", func { b ->
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= -120
		}
		mulSinkI32 = x
	})
	// 65537x = 65536x + x
	b.Run("65537", func { b ->
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 65537
		}
		mulSinkI32 = x
	})
	// 65538x = 65536x + 2x
	b.Run("65538", func { b ->
		x := int32(1)
		for i := 0; i < b.N; i++ {
			x *= 65538
		}
		mulSinkI32 = x
	})
}

func BenchmarkMulconstI64(b *testing.B) {
	// 3x = 2x + x
	b.Run("3", func { b ->
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 3
		}
		mulSinkI64 = x
	})
	// 5x = 4x + x
	b.Run("5", func { b ->
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 5
		}
		mulSinkI64 = x
	})
	// 12x = 8x + 4x
	b.Run("12", func { b ->
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 12
		}
		mulSinkI64 = x
	})
	// 120x = 128x - 8x
	b.Run("120", func { b ->
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 120
		}
		mulSinkI64 = x
	})
	// -120x = 8x - 120x
	b.Run("-120", func { b ->
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= -120
		}
		mulSinkI64 = x
	})
	// 65537x = 65536x + x
	b.Run("65537", func { b ->
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 65537
		}
		mulSinkI64 = x
	})
	// 65538x = 65536x + 2x
	b.Run("65538", func { b ->
		x := int64(1)
		for i := 0; i < b.N; i++ {
			x *= 65538
		}
		mulSinkI64 = x
	})
}

func BenchmarkMulconstU32(b *testing.B) {
	// 3x = 2x + x
	b.Run("3", func { b ->
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 3
		}
		mulSinkU32 = x
	})
	// 5x = 4x + x
	b.Run("5", func { b ->
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 5
		}
		mulSinkU32 = x
	})
	// 12x = 8x + 4x
	b.Run("12", func { b ->
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 12
		}
		mulSinkU32 = x
	})
	// 120x = 128x - 8x
	b.Run("120", func { b ->
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 120
		}
		mulSinkU32 = x
	})
	// 65537x = 65536x + x
	b.Run("65537", func { b ->
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 65537
		}
		mulSinkU32 = x
	})
	// 65538x = 65536x + 2x
	b.Run("65538", func { b ->
		x := uint32(1)
		for i := 0; i < b.N; i++ {
			x *= 65538
		}
		mulSinkU32 = x
	})
}

func BenchmarkMulconstU64(b *testing.B) {
	// 3x = 2x + x
	b.Run("3", func { b ->
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 3
		}
		mulSinkU64 = x
	})
	// 5x = 4x + x
	b.Run("5", func { b ->
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 5
		}
		mulSinkU64 = x
	})
	// 12x = 8x + 4x
	b.Run("12", func { b ->
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 12
		}
		mulSinkU64 = x
	})
	// 120x = 128x - 8x
	b.Run("120", func { b ->
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 120
		}
		mulSinkU64 = x
	})
	// 65537x = 65536x + x
	b.Run("65537", func { b ->
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 65537
		}
		mulSinkU64 = x
	})
	// 65538x = 65536x + 2x
	b.Run("65538", func { b ->
		x := uint64(1)
		for i := 0; i < b.N; i++ {
			x *= 65538
		}
		mulSinkU64 = x
	})
}
