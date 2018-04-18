// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	"math/rand"
	"runtime"
	. "sync/atomic"
	"testing"
)

func TestValue(t *testing.T) {
	var v Value
	if v.Load() != nil {
		t.Fatal("initial Value is not nil")
	}
	v.Store(42)
	x := v.Load()
	if xx, ok := x.(int); !ok || xx != 42 {
		t.Fatalf("wrong value: got %+v, want 42", x)
	}
	v.Store(84)
	x = v.Load()
	if xx, ok := x.(int); !ok || xx != 84 {
		t.Fatalf("wrong value: got %+v, want 84", x)
	}
}

func TestValueLarge(t *testing.T) {
	var v Value
	v.Store("foo")
	x := v.Load()
	if xx, ok := x.(string); !ok || xx != "foo" {
		t.Fatalf("wrong value: got %+v, want foo", x)
	}
	v.Store("barbaz")
	x = v.Load()
	if xx, ok := x.(string); !ok || xx != "barbaz" {
		t.Fatalf("wrong value: got %+v, want barbaz", x)
	}
}

func TestValuePanic(t *testing.T) {
	const nilErr = "sync/atomic: store of nil value into Value"
	const badErr = "sync/atomic: store of inconsistently typed value into Value"
	var v Value
	func() {
		defer func() {
			err := recover()
			if err != nilErr {
				t.Fatalf("inconsistent store panic: got '%v', want '%v'", err, nilErr)
			}
		}()
		v.Store(nil)
	}()
	v.Store(42)
	func() {
		defer func() {
			err := recover()
			if err != badErr {
				t.Fatalf("inconsistent store panic: got '%v', want '%v'", err, badErr)
			}
		}()
		v.Store("foo")
	}()
	func() {
		defer func() {
			err := recover()
			if err != nilErr {
				t.Fatalf("inconsistent store panic: got '%v', want '%v'", err, nilErr)
			}
		}()
		v.Store(nil)
	}()
}

func TestValueConcurrent(t *testing.T) {
	tests := [][]interface{}{
		{uint16(0), ^uint16(0), uint16(1 + 2<<8), uint16(3 + 4<<8)},
		{uint32(0), ^uint32(0), uint32(1 + 2<<16), uint32(3 + 4<<16)},
		{uint64(0), ^uint64(0), uint64(1 + 2<<32), uint64(3 + 4<<32)},
		{complex(0, 0), complex(1, 2), complex(3, 4), complex(5, 6)},
	}
	p := 4 * runtime.GOMAXPROCS(0)
	N := int(1e5)
	if testing.Short() {
		p /= 2
		N = 1e3
	}
	for _, test := range tests {
		var v Value
		done := make(chan bool)
		for i := 0; i < p; i++ {
			go func() {
				r := rand.New(rand.NewSource(rand.Int63()))
			loop:
				for j := 0; j < N; j++ {
					x := test[r.Intn(len(test))]
					v.Store(x)
					x = v.Load()
					for _, x1 := range test {
						if x == x1 {
							continue loop
						}
					}
					t.Logf("loaded unexpected value %+v, want %+v", x, test)
					done <- false
				}
				done <- true
			}()
		}
		for i := 0; i < p; i++ {
			if !<-done {
				t.FailNow()
			}
		}
	}
}

func BenchmarkValueRead(b *testing.B) {
	var v Value
	v.Store(new(int))
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			x := v.Load().(*int)
			if *x != 0 {
				b.Fatalf("wrong value: got %v, want 0", *x)
			}
		}
	})
}
