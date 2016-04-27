// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package runtime_test

import "testing"

const N = 20

func BenchmarkMakeSlice(b *testing.B) {
	var x []byte
	for i := 0; i < b.N; i++ {
		x = make([]byte, 32)
		_ = x
	}
}

func BenchmarkGrowSliceBytes(b *testing.B) {
	b.StopTimer()
	var x = make([]byte, 9)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_ = append([]byte(nil), x...)
	}
}

func BenchmarkGrowSliceInts(b *testing.B) {
	b.StopTimer()
	var x = make([]int, 9)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_ = append([]int(nil), x...)
	}
}

func BenchmarkGrowSlicePtr(b *testing.B) {
	b.StopTimer()
	var x = make([]*byte, 9)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_ = append([]*byte(nil), x...)
	}
}

type struct24 struct{ a, b, c int64 }

func BenchmarkGrowSliceStruct24Bytes(b *testing.B) {
	b.StopTimer()
	var x = make([]struct24, 9)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_ = append([]struct24(nil), x...)
	}
}

func BenchmarkAppend(b *testing.B) {
	b.StopTimer()
	x := make([]int, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		for j := 0; j < N; j++ {
			x = append(x, j)
		}
	}
}

func BenchmarkAppendGrowByte(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x []byte
		for j := 0; j < 1<<20; j++ {
			x = append(x, byte(j))
		}
	}
}

func BenchmarkAppendGrowString(b *testing.B) {
	var s string
	for i := 0; i < b.N; i++ {
		var x []string
		for j := 0; j < 1<<20; j++ {
			x = append(x, s)
		}
	}
}

func benchmarkAppendBytes(b *testing.B, length int) {
	b.StopTimer()
	x := make([]byte, 0, N)
	y := make([]byte, length)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		x = append(x, y...)
	}
}

func BenchmarkAppend1Byte(b *testing.B) {
	benchmarkAppendBytes(b, 1)
}

func BenchmarkAppend4Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 4)
}

func BenchmarkAppend7Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 7)
}

func BenchmarkAppend8Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 8)
}

func BenchmarkAppend15Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 15)
}

func BenchmarkAppend16Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 16)
}

func BenchmarkAppend32Bytes(b *testing.B) {
	benchmarkAppendBytes(b, 32)
}

func benchmarkAppendStr(b *testing.B, str string) {
	b.StopTimer()
	x := make([]byte, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		x = append(x, str...)
	}
}

func BenchmarkAppendStr1Byte(b *testing.B) {
	benchmarkAppendStr(b, "1")
}

func BenchmarkAppendStr4Bytes(b *testing.B) {
	benchmarkAppendStr(b, "1234")
}

func BenchmarkAppendStr8Bytes(b *testing.B) {
	benchmarkAppendStr(b, "12345678")
}

func BenchmarkAppendStr16Bytes(b *testing.B) {
	benchmarkAppendStr(b, "1234567890123456")
}

func BenchmarkAppendStr32Bytes(b *testing.B) {
	benchmarkAppendStr(b, "12345678901234567890123456789012")
}

func BenchmarkAppendSpecialCase(b *testing.B) {
	b.StopTimer()
	x := make([]int, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		for j := 0; j < N; j++ {
			if len(x) < cap(x) {
				x = x[:len(x)+1]
				x[len(x)-1] = j
			} else {
				x = append(x, j)
			}
		}
	}
}

var x []int

func f() int {
	x[:1][0] = 3
	return 2
}

func TestSideEffectOrder(t *testing.T) {
	x = make([]int, 0, 10)
	x = append(x, 1, f())
	if x[0] != 1 || x[1] != 2 {
		t.Error("append failed: ", x[0], x[1])
	}
}

func TestAppendOverlap(t *testing.T) {
	x := []byte("1234")
	x = append(x[1:], x...) // p > q in runtime·appendslice.
	got := string(x)
	want := "2341234"
	if got != want {
		t.Errorf("overlap failed: got %q want %q", got, want)
	}
}

func benchmarkCopySlice(b *testing.B, l int) {
	s := make([]byte, l)
	buf := make([]byte, 4096)
	var n int
	for i := 0; i < b.N; i++ {
		n = copy(buf, s)
	}
	b.SetBytes(int64(n))
}

func benchmarkCopyStr(b *testing.B, l int) {
	s := string(make([]byte, l))
	buf := make([]byte, 4096)
	var n int
	for i := 0; i < b.N; i++ {
		n = copy(buf, s)
	}
	b.SetBytes(int64(n))
}

func BenchmarkCopy1Byte(b *testing.B)    { benchmarkCopySlice(b, 1) }
func BenchmarkCopy2Byte(b *testing.B)    { benchmarkCopySlice(b, 2) }
func BenchmarkCopy4Byte(b *testing.B)    { benchmarkCopySlice(b, 4) }
func BenchmarkCopy8Byte(b *testing.B)    { benchmarkCopySlice(b, 8) }
func BenchmarkCopy12Byte(b *testing.B)   { benchmarkCopySlice(b, 12) }
func BenchmarkCopy16Byte(b *testing.B)   { benchmarkCopySlice(b, 16) }
func BenchmarkCopy32Byte(b *testing.B)   { benchmarkCopySlice(b, 32) }
func BenchmarkCopy128Byte(b *testing.B)  { benchmarkCopySlice(b, 128) }
func BenchmarkCopy1024Byte(b *testing.B) { benchmarkCopySlice(b, 1024) }

func BenchmarkCopy1String(b *testing.B)    { benchmarkCopyStr(b, 1) }
func BenchmarkCopy2String(b *testing.B)    { benchmarkCopyStr(b, 2) }
func BenchmarkCopy4String(b *testing.B)    { benchmarkCopyStr(b, 4) }
func BenchmarkCopy8String(b *testing.B)    { benchmarkCopyStr(b, 8) }
func BenchmarkCopy12String(b *testing.B)   { benchmarkCopyStr(b, 12) }
func BenchmarkCopy16String(b *testing.B)   { benchmarkCopyStr(b, 16) }
func BenchmarkCopy32String(b *testing.B)   { benchmarkCopyStr(b, 32) }
func BenchmarkCopy128String(b *testing.B)  { benchmarkCopyStr(b, 128) }
func BenchmarkCopy1024String(b *testing.B) { benchmarkCopyStr(b, 1024) }

var (
	sByte []byte
	s1Ptr []uintptr
	s2Ptr [][2]uintptr
	s3Ptr [][3]uintptr
	s4Ptr [][4]uintptr
)

// BenchmarkAppendInPlace tests the performance of append
// when the result is being written back to the same slice.
// In order for the in-place optimization to occur,
// the slice must be referred to by address;
// using a global is an easy way to trigger that.
// We test the "grow" and "no grow" paths separately,
// but not the "normal" (occasionally grow) path,
// because it is a blend of the other two.
// We use small numbers and small sizes in an attempt
// to avoid benchmarking memory allocation and copying.
// We use scalars instead of pointers in an attempt
// to avoid benchmarking the write barriers.
// We benchmark four common sizes (byte, pointer, string/interface, slice),
// and one larger size.
func BenchmarkAppendInPlace(b *testing.B) {
	b.Run("NoGrow", func(b *testing.B) {
		const C = 128

		b.Run("Byte", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sByte = make([]byte, C)
				for j := 0; j < C; j++ {
					sByte = append(sByte, 0x77)
				}
			}
		})

		b.Run("1Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s1Ptr = make([]uintptr, C)
				for j := 0; j < C; j++ {
					s1Ptr = append(s1Ptr, 0x77)
				}
			}
		})

		b.Run("2Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s2Ptr = make([][2]uintptr, C)
				for j := 0; j < C; j++ {
					s2Ptr = append(s2Ptr, [2]uintptr{0x77, 0x88})
				}
			}
		})

		b.Run("3Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s3Ptr = make([][3]uintptr, C)
				for j := 0; j < C; j++ {
					s3Ptr = append(s3Ptr, [3]uintptr{0x77, 0x88, 0x99})
				}
			}
		})

		b.Run("4Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s4Ptr = make([][4]uintptr, C)
				for j := 0; j < C; j++ {
					s4Ptr = append(s4Ptr, [4]uintptr{0x77, 0x88, 0x99, 0xAA})
				}
			}
		})

	})

	b.Run("Grow", func(b *testing.B) {
		const C = 5

		b.Run("Byte", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sByte = make([]byte, 0)
				for j := 0; j < C; j++ {
					sByte = append(sByte, 0x77)
					sByte = sByte[:cap(sByte)]
				}
			}
		})

		b.Run("1Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s1Ptr = make([]uintptr, 0)
				for j := 0; j < C; j++ {
					s1Ptr = append(s1Ptr, 0x77)
					s1Ptr = s1Ptr[:cap(s1Ptr)]
				}
			}
		})

		b.Run("2Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s2Ptr = make([][2]uintptr, 0)
				for j := 0; j < C; j++ {
					s2Ptr = append(s2Ptr, [2]uintptr{0x77, 0x88})
					s2Ptr = s2Ptr[:cap(s2Ptr)]
				}
			}
		})

		b.Run("3Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s3Ptr = make([][3]uintptr, 0)
				for j := 0; j < C; j++ {
					s3Ptr = append(s3Ptr, [3]uintptr{0x77, 0x88, 0x99})
					s3Ptr = s3Ptr[:cap(s3Ptr)]
				}
			}
		})

		b.Run("4Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s4Ptr = make([][4]uintptr, 0)
				for j := 0; j < C; j++ {
					s4Ptr = append(s4Ptr, [4]uintptr{0x77, 0x88, 0x99, 0xAA})
					s4Ptr = s4Ptr[:cap(s4Ptr)]
				}
			}
		})

	})
}
