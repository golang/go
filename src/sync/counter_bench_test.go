// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"math"
	"runtime"
	"sync/atomic"
	"testing"
	"unsafe"

	"internal/cpu"
	"sync"
)

// Sub-benchmarks use the "key=value" naming convention so benchstat can
// group and compare results across dimensions:
//   - impl:  counter implementation ("atomic" or "counter")
//   - op:    operation, can include a mixture of ops
//     ("Add", "Value", "value:add=1:99", "value:add=10:90")
//   - value: counter value type ("int64", "uint64", or "float64")
//   - par:   parallelism ("single" goroutine or "multi" via RunParallel)
func BenchmarkCounter(b *testing.B) {
	b.Run("impl=atomic/op=Add/value=int64/par=single", func(b *testing.B) {
		var c atomic.Int64
		b.ResetTimer()
		for range b.N {
			c.Add(1)
		}
	})

	b.Run("impl=counter/op=Add/value=int64/par=single", func(b *testing.B) {
		var c sync.Counter[int64]
		b.ResetTimer()
		for range b.N {
			c.Add(1)
		}
	})

	b.Run("impl=atomic/op=Add/value=uint64/par=single", func(b *testing.B) {
		var c atomic.Uint64
		b.ResetTimer()
		for range b.N {
			c.Add(1)
		}
	})

	b.Run("impl=counter/op=Add/value=uint64/par=single", func(b *testing.B) {
		var c sync.Counter[uint64]
		b.ResetTimer()
		for range b.N {
			c.Add(1)
		}
	})

	b.Run("impl=atomic/op=Add/value=float64/par=single", func(b *testing.B) {
		var c atomicFloat64
		b.ResetTimer()
		for range b.N {
			c.Add(1)
		}
	})

	b.Run("impl=counter/op=Add/value=float64/par=single", func(b *testing.B) {
		var c sync.Counter[float64]
		b.ResetTimer()
		for range b.N {
			c.Add(1)
		}
	})

	b.Run("impl=atomic/op=Value/value=int64/par=single", func(b *testing.B) {
		var c atomic.Int64
		c.Add(1)
		var sink int64
		b.ResetTimer()
		for range b.N {
			sink = c.Load()
		}
		runtime.KeepAlive(sink)
	})

	b.Run("impl=counter/op=Value/value=int64/par=single", func(b *testing.B) {
		var c sync.Counter[int64]
		c.Add(1)
		var sink int64
		b.ResetTimer()
		for range b.N {
			sink = c.Value()
		}
		runtime.KeepAlive(sink)
	})

	b.Run("impl=atomic/op=Value/value=uint64/par=single", func(b *testing.B) {
		var c atomic.Uint64
		c.Add(1)
		var sink uint64
		b.ResetTimer()
		for range b.N {
			sink = c.Load()
		}
		runtime.KeepAlive(sink)
	})

	b.Run("impl=counter/op=Value/value=uint64/par=single", func(b *testing.B) {
		var c sync.Counter[uint64]
		c.Add(1)
		var sink uint64
		b.ResetTimer()
		for range b.N {
			sink = c.Value()
		}
		runtime.KeepAlive(sink)
	})

	b.Run("impl=atomic/op=Value/value=float64/par=single", func(b *testing.B) {
		var c atomicFloat64
		c.Add(1)
		var sink float64
		b.ResetTimer()
		for range b.N {
			sink = c.Value()
		}
		runtime.KeepAlive(sink)
	})

	b.Run("impl=counter/op=Value/value=float64/par=single", func(b *testing.B) {
		var c sync.Counter[float64]
		c.Add(1)
		var sink float64
		b.ResetTimer()
		for range b.N {
			sink = c.Value()
		}
		runtime.KeepAlive(sink)
	})
}

func BenchmarkCounterParallel(b *testing.B) {
	b.Run("impl=atomic/op=Add/value=int64/par=multi", func(b *testing.B) {
		var c atomic.Int64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				c.Add(1)
			}
		})
	})

	b.Run("impl=counter/op=Add/value=int64/par=multi", func(b *testing.B) {
		var c sync.Counter[int64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				c.Add(1)
			}
		})
	})

	b.Run("impl=atomic/op=Add/value=uint64/par=multi", func(b *testing.B) {
		var c atomic.Uint64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				c.Add(1)
			}
		})
	})

	b.Run("impl=counter/op=Add/value=uint64/par=multi", func(b *testing.B) {
		var c sync.Counter[uint64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				c.Add(1)
			}
		})
	})

	b.Run("impl=atomic/op=Add/value=float64/par=multi", func(b *testing.B) {
		var c atomicFloat64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				c.Add(1)
			}
		})
	})

	b.Run("impl=counter/op=Add/value=float64/par=multi", func(b *testing.B) {
		var c sync.Counter[float64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				c.Add(1)
			}
		})
	})

	b.Run("impl=atomic/op=Value/value=int64/par=multi", func(b *testing.B) {
		var c atomic.Int64
		c.Add(1)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var sink int64
			for pb.Next() {
				sink = c.Load()
			}
			runtime.KeepAlive(sink)
		})
	})

	b.Run("impl=counter/op=Value/value=int64/par=multi", func(b *testing.B) {
		var c sync.Counter[int64]
		c.Add(1)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var sink int64
			for pb.Next() {
				sink = c.Value()
			}
			runtime.KeepAlive(sink)
		})
	})

	b.Run("impl=atomic/op=Value/value=uint64/par=multi", func(b *testing.B) {
		var c atomic.Uint64
		c.Add(1)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var sink uint64
			for pb.Next() {
				sink = c.Load()
			}
			runtime.KeepAlive(sink)
		})
	})

	b.Run("impl=counter/op=Value/value=uint64/par=multi", func(b *testing.B) {
		var c sync.Counter[uint64]
		c.Add(1)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var sink uint64
			for pb.Next() {
				sink = c.Value()
			}
			runtime.KeepAlive(sink)
		})
	})

	b.Run("impl=atomic/op=Value/value=float64/par=multi", func(b *testing.B) {
		var c atomicFloat64
		c.Add(1)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var sink float64
			for pb.Next() {
				sink = c.Value()
			}
			runtime.KeepAlive(sink)
		})
	})

	b.Run("impl=counter/op=Value/value=float64/par=multi", func(b *testing.B) {
		var c sync.Counter[float64]
		c.Add(1)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var sink float64
			for pb.Next() {
				sink = c.Value()
			}
			runtime.KeepAlive(sink)
		})
	})
}

func BenchmarkCounterMixed(b *testing.B) {
	b.Run("impl=atomic/op=value:add=1:99/value=int64/par=multi", func(b *testing.B) {
		var c atomic.Int64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 != 0 {
					c.Add(1)
				} else {
					c.Load()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=1:99/value=int64/par=multi", func(b *testing.B) {
		var c sync.Counter[int64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 != 0 {
					c.Add(1)
				} else {
					c.Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=1:99/value=uint64/par=multi", func(b *testing.B) {
		var c atomic.Uint64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 != 0 {
					c.Add(1)
				} else {
					c.Load()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=1:99/value=uint64/par=multi", func(b *testing.B) {
		var c sync.Counter[uint64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 != 0 {
					c.Add(1)
				} else {
					c.Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=1:99/value=float64/par=multi", func(b *testing.B) {
		var c atomicFloat64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 != 0 {
					c.Add(1)
				} else {
					c.Value()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=1:99/value=float64/par=multi", func(b *testing.B) {
		var c sync.Counter[float64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 != 0 {
					c.Add(1)
				} else {
					c.Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=10:90/value=int64/par=multi", func(b *testing.B) {
		var c atomic.Int64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 < 90 {
					c.Add(1)
				} else {
					c.Load()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=10:90/value=int64/par=multi", func(b *testing.B) {
		var c sync.Counter[int64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 < 90 {
					c.Add(1)
				} else {
					c.Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=10:90/value=uint64/par=multi", func(b *testing.B) {
		var c atomic.Uint64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 < 90 {
					c.Add(1)
				} else {
					c.Load()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=10:90/value=uint64/par=multi", func(b *testing.B) {
		var c sync.Counter[uint64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 < 90 {
					c.Add(1)
				} else {
					c.Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=10:90/value=float64/par=multi", func(b *testing.B) {
		var c atomicFloat64
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 < 90 {
					c.Add(1)
				} else {
					c.Value()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=10:90/value=float64/par=multi", func(b *testing.B) {
		var c sync.Counter[float64]
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32
			for pb.Next() {
				i++
				if i%100 < 90 {
					c.Add(1)
				} else {
					c.Value()
				}
			}
		})
	})
}

func BenchmarkCounterRandom(b *testing.B) {
	const countPerCacheLine = int(max(unsafe.Sizeof(cpu.CacheLinePad{}), uintptr(64)) / unsafe.Sizeof(uint64(0)))

	b.Run("impl=atomic/op=value:add=1:99/value=int64/par=multi", func(b *testing.B) {
		c := make([]atomic.Int64, countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 != 0 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Load()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=1:99/value=int64/par=multi", func(b *testing.B) {
		c := make([]sync.Counter[int64], countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 != 0 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=1:99/value=uint64/par=multi", func(b *testing.B) {
		c := make([]atomic.Uint64, countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 != 0 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Load()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=1:99/value=uint64/par=multi", func(b *testing.B) {
		c := make([]sync.Counter[uint64], countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 != 0 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=1:99/value=float64/par=multi", func(b *testing.B) {
		c := make([]atomicFloat64, countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 != 0 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Value()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=1:99/value=float64/par=multi", func(b *testing.B) {
		c := make([]sync.Counter[float64], countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 != 0 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=10:90/value=int64/par=multi", func(b *testing.B) {
		c := make([]atomic.Int64, countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 < 90 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Load()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=10:90/value=int64/par=multi", func(b *testing.B) {
		c := make([]sync.Counter[int64], countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 < 90 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=10:90/value=uint64/par=multi", func(b *testing.B) {
		c := make([]atomic.Uint64, countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 < 90 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Load()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=10:90/value=uint64/par=multi", func(b *testing.B) {
		c := make([]sync.Counter[uint64], countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 < 90 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Value()
				}
			}
		})
	})

	b.Run("impl=atomic/op=value:add=10:90/value=float64/par=multi", func(b *testing.B) {
		c := make([]atomicFloat64, countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 < 90 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Value()
				}
			}
		})
	})

	b.Run("impl=counter/op=value:add=10:90/value=float64/par=multi", func(b *testing.B) {
		c := make([]sync.Counter[float64], countPerCacheLine)
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			var i uint32 = 1
			for pb.Next() {
				i ^= i << 13
				i ^= i >> 17
				i ^= i << 5
				if i%100 < 90 {
					c[int(i)%countPerCacheLine].Add(1)
				} else {
					c[int(i)%countPerCacheLine].Value()
				}
			}
		})
	})
}

type atomicFloat64 struct {
	val atomic.Uint64
}

func (af *atomicFloat64) Add(delta float64) {
	for {
		old := af.val.Load()
		newVal := math.Float64bits(math.Float64frombits(old) + delta)
		if af.val.CompareAndSwap(old, newVal) {
			break
		}
	}
}

func (af *atomicFloat64) Value() float64 {
	return math.Float64frombits(af.val.Load())
}
