// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"testing"
)

const N = 20

func BenchmarkMakeSliceCopy(b *testing.B) {
	const length = 32
	var bytes = make([]byte, 8*length)
	var ints = make([]int, length)
	var ptrs = make([]*byte, length)
	b.Run("mallocmove", func(b *testing.B) {
		b.Run("Byte", func(b *testing.B) {
			var x []byte
			for i := 0; i < b.N; i++ {
				x = make([]byte, len(bytes))
				copy(x, bytes)
			}
		})
		b.Run("Int", func(b *testing.B) {
			var x []int
			for i := 0; i < b.N; i++ {
				x = make([]int, len(ints))
				copy(x, ints)
			}
		})
		b.Run("Ptr", func(b *testing.B) {
			var x []*byte
			for i := 0; i < b.N; i++ {
				x = make([]*byte, len(ptrs))
				copy(x, ptrs)
			}

		})
	})
	b.Run("makecopy", func(b *testing.B) {
		b.Run("Byte", func(b *testing.B) {
			var x []byte
			for i := 0; i < b.N; i++ {
				x = make([]byte, 8*length)
				copy(x, bytes)
			}
		})
		b.Run("Int", func(b *testing.B) {
			var x []int
			for i := 0; i < b.N; i++ {
				x = make([]int, length)
				copy(x, ints)
			}
		})
		b.Run("Ptr", func(b *testing.B) {
			var x []*byte
			for i := 0; i < b.N; i++ {
				x = make([]*byte, length)
				copy(x, ptrs)
			}

		})
	})
	b.Run("nilappend", func(b *testing.B) {
		b.Run("Byte", func(b *testing.B) {
			var x []byte
			for i := 0; i < b.N; i++ {
				x = append([]byte(nil), bytes...)
				_ = x
			}
		})
		b.Run("Int", func(b *testing.B) {
			var x []int
			for i := 0; i < b.N; i++ {
				x = append([]int(nil), ints...)
				_ = x
			}
		})
		b.Run("Ptr", func(b *testing.B) {
			var x []*byte
			for i := 0; i < b.N; i++ {
				x = append([]*byte(nil), ptrs...)
				_ = x
			}
		})
	})
}

type (
	struct24 struct{ a, b, c int64 }
	struct32 struct{ a, b, c, d int64 }
	struct40 struct{ a, b, c, d, e int64 }
)

func BenchmarkMakeSlice(b *testing.B) {
	const length = 2
	b.Run("Byte", func(b *testing.B) {
		var x []byte
		for i := 0; i < b.N; i++ {
			x = make([]byte, length, 2*length)
			_ = x
		}
	})
	b.Run("Int16", func(b *testing.B) {
		var x []int16
		for i := 0; i < b.N; i++ {
			x = make([]int16, length, 2*length)
			_ = x
		}
	})
	b.Run("Int", func(b *testing.B) {
		var x []int
		for i := 0; i < b.N; i++ {
			x = make([]int, length, 2*length)
			_ = x
		}
	})
	b.Run("Ptr", func(b *testing.B) {
		var x []*byte
		for i := 0; i < b.N; i++ {
			x = make([]*byte, length, 2*length)
			_ = x
		}
	})
	b.Run("Struct", func(b *testing.B) {
		b.Run("24", func(b *testing.B) {
			var x []struct24
			for i := 0; i < b.N; i++ {
				x = make([]struct24, length, 2*length)
				_ = x
			}
		})
		b.Run("32", func(b *testing.B) {
			var x []struct32
			for i := 0; i < b.N; i++ {
				x = make([]struct32, length, 2*length)
				_ = x
			}
		})
		b.Run("40", func(b *testing.B) {
			var x []struct40
			for i := 0; i < b.N; i++ {
				x = make([]struct40, length, 2*length)
				_ = x
			}
		})

	})
}

func BenchmarkGrowSlice(b *testing.B) {
	b.Run("Byte", func(b *testing.B) {
		x := make([]byte, 9)
		for i := 0; i < b.N; i++ {
			_ = append([]byte(nil), x...)
		}
	})
	b.Run("Int16", func(b *testing.B) {
		x := make([]int16, 9)
		for i := 0; i < b.N; i++ {
			_ = append([]int16(nil), x...)
		}
	})
	b.Run("Int", func(b *testing.B) {
		x := make([]int, 9)
		for i := 0; i < b.N; i++ {
			_ = append([]int(nil), x...)
		}
	})
	b.Run("Ptr", func(b *testing.B) {
		x := make([]*byte, 9)
		for i := 0; i < b.N; i++ {
			_ = append([]*byte(nil), x...)
		}
	})
	b.Run("Struct", func(b *testing.B) {
		b.Run("24", func(b *testing.B) {
			x := make([]struct24, 9)
			for i := 0; i < b.N; i++ {
				_ = append([]struct24(nil), x...)
			}
		})
		b.Run("32", func(b *testing.B) {
			x := make([]struct32, 9)
			for i := 0; i < b.N; i++ {
				_ = append([]struct32(nil), x...)
			}
		})
		b.Run("40", func(b *testing.B) {
			x := make([]struct40, 9)
			for i := 0; i < b.N; i++ {
				_ = append([]struct40(nil), x...)
			}
		})

	})
}

var (
	SinkIntSlice        []int
	SinkIntPointerSlice []*int
)

func BenchmarkExtendSlice(b *testing.B) {
	var length = 4 // Use a variable to prevent stack allocation of slices.
	b.Run("IntSlice", func(b *testing.B) {
		s := make([]int, 0, length)
		for i := 0; i < b.N; i++ {
			s = append(s[:0:length/2], make([]int, length)...)
		}
		SinkIntSlice = s
	})
	b.Run("PointerSlice", func(b *testing.B) {
		s := make([]*int, 0, length)
		for i := 0; i < b.N; i++ {
			s = append(s[:0:length/2], make([]*int, length)...)
		}
		SinkIntPointerSlice = s
	})
	b.Run("NoGrow", func(b *testing.B) {
		s := make([]int, 0, length)
		for i := 0; i < b.N; i++ {
			s = append(s[:0:length], make([]int, length)...)
		}
		SinkIntSlice = s
	})
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

func BenchmarkAppendSlice(b *testing.B) {
	for _, length := range []int{1, 4, 7, 8, 15, 16, 32} {
		b.Run(fmt.Sprint(length, "Bytes"), func(b *testing.B) {
			x := make([]byte, 0, N)
			y := make([]byte, length)
			for i := 0; i < b.N; i++ {
				x = x[0:0]
				x = append(x, y...)
			}
		})
	}
}

var (
	blackhole []byte
)

func BenchmarkAppendSliceLarge(b *testing.B) {
	for _, length := range []int{1 << 10, 4 << 10, 16 << 10, 64 << 10, 256 << 10, 1024 << 10} {
		y := make([]byte, length)
		b.Run(fmt.Sprint(length, "Bytes"), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				blackhole = nil
				blackhole = append(blackhole, y...)
			}
		})
	}
}

func BenchmarkAppendStr(b *testing.B) {
	for _, str := range []string{
		"1",
		"1234",
		"12345678",
		"1234567890123456",
		"12345678901234567890123456789012",
	} {
		b.Run(fmt.Sprint(len(str), "Bytes"), func(b *testing.B) {
			x := make([]byte, 0, N)
			for i := 0; i < b.N; i++ {
				x = x[0:0]
				x = append(x, str...)
			}
		})
	}
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

func BenchmarkCopy(b *testing.B) {
	for _, l := range []int{1, 2, 4, 8, 12, 16, 32, 128, 1024} {
		buf := make([]byte, 4096)
		b.Run(fmt.Sprint(l, "Byte"), func(b *testing.B) {
			s := make([]byte, l)
			var n int
			for i := 0; i < b.N; i++ {
				n = copy(buf, s)
			}
			b.SetBytes(int64(n))
		})
		b.Run(fmt.Sprint(l, "String"), func(b *testing.B) {
			s := string(make([]byte, l))
			var n int
			for i := 0; i < b.N; i++ {
				n = copy(buf, s)
			}
			b.SetBytes(int64(n))
		})
	}
}

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
