// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test independent goroutines modifying a comprehensive
// variety of vars during aggressive garbage collection.

// The point is to catch GC regressions like fixedbugs/issue22781.go

package main

import (
	"errors"
	"runtime"
	"runtime/debug"
	"sync"
)

const (
	goroutines = 8
	allocs     = 8
	mods       = 8

	length = 9
)

func main() {
	debug.SetGCPercent(1)
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		for _, t := range types {
			err := t.valid()
			if err != nil {
				panic(err)
			}
			wg.Add(1)
			go func(f modifier) {
				var wg2 sync.WaitGroup
				for j := 0; j < allocs; j++ {
					wg2.Add(1)
					go func() {
						f.t()
						wg2.Done()
					}()
					wg2.Add(1)
					go func() {
						f.pointerT()
						wg2.Done()
					}()
					wg2.Add(1)
					go func() {
						f.arrayT()
						wg2.Done()
					}()
					wg2.Add(1)
					go func() {
						f.sliceT()
						wg2.Done()
					}()
					wg2.Add(1)
					go func() {
						f.mapT()
						wg2.Done()
					}()
					wg2.Add(1)
					go func() {
						f.mapPointerKeyT()
						wg2.Done()
					}()
					wg2.Add(1)
					go func() {
						f.chanT()
						wg2.Done()
					}()
					wg2.Add(1)
					go func() {
						f.interfaceT()
						wg2.Done()
					}()
				}
				wg2.Wait()
				wg.Done()
			}(t)
		}
	}
	wg.Wait()
}

type modifier struct {
	name           string
	t              func()
	pointerT       func()
	arrayT         func()
	sliceT         func()
	mapT           func()
	mapPointerKeyT func()
	chanT          func()
	interfaceT     func()
}

func (a modifier) valid() error {
	switch {
	case a.name == "":
		return errors.New("modifier without name")
	case a.t == nil:
		return errors.New(a.name + " missing t")
	case a.pointerT == nil:
		return errors.New(a.name + " missing pointerT")
	case a.arrayT == nil:
		return errors.New(a.name + " missing arrayT")
	case a.sliceT == nil:
		return errors.New(a.name + " missing sliceT")
	case a.mapT == nil:
		return errors.New(a.name + " missing mapT")
	case a.mapPointerKeyT == nil:
		return errors.New(a.name + " missing mapPointerKeyT")
	case a.chanT == nil:
		return errors.New(a.name + " missing chanT")
	case a.interfaceT == nil:
		return errors.New(a.name + " missing interfaceT")
	default:
		return nil
	}
}

var types = []modifier{
	modifier{
		name: "bool",
		t: func() {
			var a bool
			for i := 0; i < mods; i++ {
				a = !a
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *bool { return new(bool) }()
			for i := 0; i < mods; i++ {
				*a = !*a
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]bool{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] = !a[j]
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]bool, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] = !a[j]
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[bool]bool)
			for i := 0; i < mods; i++ {
				a[false] = !a[false]
				a[true] = !a[true]
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*bool]bool)
			for i := 0; i < length; i++ {
				a[new(bool)] = false
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, v := range a {
					a[k] = !v
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan bool)
			for i := 0; i < mods; i++ {
				go func() { a <- false }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(bool(false))
			for i := 0; i < mods; i++ {
				a = !a.(bool)
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "uint8",
		t: func() {
			var u uint8
			for i := 0; i < mods; i++ {
				u++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *uint8 { return new(uint8) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]uint8{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]uint8, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[uint8]uint8)
			for i := 0; i < length; i++ {
				a[uint8(i)] = uint8(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*uint8]uint8)
			for i := 0; i < length; i++ {
				a[new(uint8)] = uint8(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan uint8)
			for i := 0; i < mods; i++ {
				go func() { a <- uint8(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(uint8(0))
			for i := 0; i < mods; i++ {
				a = a.(uint8) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "uint16",
		t: func() {
			var u uint16
			for i := 0; i < mods; i++ {
				u++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *uint16 { return new(uint16) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]uint16{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]uint16, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[uint16]uint16)
			for i := 0; i < length; i++ {
				a[uint16(i)] = uint16(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*uint16]uint16)
			for i := 0; i < length; i++ {
				a[new(uint16)] = uint16(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan uint16)
			for i := 0; i < mods; i++ {
				go func() { a <- uint16(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(uint16(0))
			for i := 0; i < mods; i++ {
				a = a.(uint16) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "uint32",
		t: func() {
			var u uint32
			for i := 0; i < mods; i++ {
				u++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *uint32 { return new(uint32) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]uint32{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]uint32, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[uint32]uint32)
			for i := 0; i < length; i++ {
				a[uint32(i)] = uint32(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*uint32]uint32)
			for i := 0; i < length; i++ {
				a[new(uint32)] = uint32(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan uint32)
			for i := 0; i < mods; i++ {
				go func() { a <- uint32(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(uint32(0))
			for i := 0; i < mods; i++ {
				a = a.(uint32) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "uint64",
		t: func() {
			var u uint64
			for i := 0; i < mods; i++ {
				u++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *uint64 { return new(uint64) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]uint64{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]uint64, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[uint64]uint64)
			for i := 0; i < length; i++ {
				a[uint64(i)] = uint64(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*uint64]uint64)
			for i := 0; i < length; i++ {
				a[new(uint64)] = uint64(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan uint64)
			for i := 0; i < mods; i++ {
				go func() { a <- uint64(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(uint64(0))
			for i := 0; i < mods; i++ {
				a = a.(uint64) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "int8",
		t: func() {
			var u int8
			for i := 0; i < mods; i++ {
				u++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *int8 { return new(int8) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]int8{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]int8, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[int8]int8)
			for i := 0; i < length; i++ {
				a[int8(i)] = int8(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*int8]int8)
			for i := 0; i < length; i++ {
				a[new(int8)] = int8(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan int8)
			for i := 0; i < mods; i++ {
				go func() { a <- int8(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(int8(0))
			for i := 0; i < mods; i++ {
				a = a.(int8) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "int16",
		t: func() {
			var u int16
			for i := 0; i < mods; i++ {
				u++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *int16 { return new(int16) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]int16{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]int16, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[int16]int16)
			for i := 0; i < length; i++ {
				a[int16(i)] = int16(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*int16]int16)
			for i := 0; i < length; i++ {
				a[new(int16)] = int16(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan int16)
			for i := 0; i < mods; i++ {
				go func() { a <- int16(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(int16(0))
			for i := 0; i < mods; i++ {
				a = a.(int16) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "int32",
		t: func() {
			var u int32
			for i := 0; i < mods; i++ {
				u++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *int32 { return new(int32) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]int32{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]int32, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[int32]int32)
			for i := 0; i < length; i++ {
				a[int32(i)] = int32(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*int32]int32)
			for i := 0; i < length; i++ {
				a[new(int32)] = int32(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan int32)
			for i := 0; i < mods; i++ {
				go func() { a <- int32(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(int32(0))
			for i := 0; i < mods; i++ {
				a = a.(int32) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "int64",
		t: func() {
			var u int64
			for i := 0; i < mods; i++ {
				u++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *int64 { return new(int64) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]int64{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]int64, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[int64]int64)
			for i := 0; i < length; i++ {
				a[int64(i)] = int64(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*int64]int64)
			for i := 0; i < length; i++ {
				a[new(int64)] = int64(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan int64)
			for i := 0; i < mods; i++ {
				go func() { a <- int64(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(int64(0))
			for i := 0; i < mods; i++ {
				a = a.(int64) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "float32",
		t: func() {
			u := float32(1.01)
			for i := 0; i < mods; i++ {
				u *= 1.01
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *float32 { return new(float32) }()
			*a = 1.01
			for i := 0; i < mods; i++ {
				*a *= 1.01
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]float32{}
			for i := 0; i < length; i++ {
				a[i] = float32(1.01)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] *= 1.01
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]float32, length)
			for i := 0; i < length; i++ {
				a[i] = float32(1.01)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] *= 1.01
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[float32]float32)
			for i := 0; i < length; i++ {
				a[float32(i)] = float32(i) + 0.01
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] *= 1.01
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*float32]float32)
			for i := 0; i < length; i++ {
				a[new(float32)] = float32(i) + 0.01
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] *= 1.01
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan float32)
			for i := 0; i < mods; i++ {
				go func() { a <- float32(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(float32(0))
			for i := 0; i < mods; i++ {
				a = a.(float32) * 1.01
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "float64",
		t: func() {
			u := float64(1.01)
			for i := 0; i < mods; i++ {
				u *= 1.01
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *float64 { return new(float64) }()
			*a = 1.01
			for i := 0; i < mods; i++ {
				*a *= 1.01
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]float64{}
			for i := 0; i < length; i++ {
				a[i] = float64(1.01)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] *= 1.01
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]float64, length)
			for i := 0; i < length; i++ {
				a[i] = float64(1.01)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] *= 1.01
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[float64]float64)
			for i := 0; i < length; i++ {
				a[float64(i)] = float64(i) + 0.01
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] *= 1.01
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*float64]float64)
			for i := 0; i < length; i++ {
				a[new(float64)] = float64(i) + 0.01
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] *= 1.01
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan float64)
			for i := 0; i < mods; i++ {
				go func() { a <- float64(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(float64(0))
			for i := 0; i < mods; i++ {
				a = a.(float64) * 1.01
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "complex64",
		t: func() {
			c := complex64(complex(float32(1.01), float32(1.01)))
			for i := 0; i < mods; i++ {
				c = complex(real(c)*1.01, imag(c)*1.01)
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *complex64 { return new(complex64) }()
			*a = complex64(complex(float32(1.01), float32(1.01)))
			for i := 0; i < mods; i++ {
				*a *= complex(real(*a)*1.01, imag(*a)*1.01)
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]complex64{}
			for i := 0; i < length; i++ {
				a[i] = complex64(complex(float32(1.01), float32(1.01)))
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] *= complex(real(a[j])*1.01, imag(a[j])*1.01)
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]complex64, length)
			for i := 0; i < length; i++ {
				a[i] = complex64(complex(float32(1.01), float32(1.01)))
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] *= complex(real(a[j])*1.01, imag(a[j])*1.01)
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[complex64]complex64)
			for i := 0; i < length; i++ {
				a[complex64(complex(float32(i), float32(i)))] = complex64(complex(float32(i), float32(i))) + 0.01
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] *= complex(real(a[k])*1.01, imag(a[k])*1.01)
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*complex64]complex64)
			for i := 0; i < length; i++ {
				a[new(complex64)] = complex64(complex(float32(i), float32(i))) + 0.01
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] *= complex(real(a[k])*1.01, imag(a[k])*1.01)
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan complex64)
			for i := 0; i < mods; i++ {
				go func() { a <- complex64(complex(float32(i), float32(i))) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(complex64(complex(float32(1.01), float32(1.01))))
			for i := 0; i < mods; i++ {
				a = a.(complex64) * complex(real(a.(complex64))*1.01, imag(a.(complex64))*1.01)
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "complex128",
		t: func() {
			c := complex128(complex(float64(1.01), float64(1.01)))
			for i := 0; i < mods; i++ {
				c = complex(real(c)*1.01, imag(c)*1.01)
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *complex128 { return new(complex128) }()
			*a = complex128(complex(float64(1.01), float64(1.01)))
			for i := 0; i < mods; i++ {
				*a *= complex(real(*a)*1.01, imag(*a)*1.01)
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]complex128{}
			for i := 0; i < length; i++ {
				a[i] = complex128(complex(float64(1.01), float64(1.01)))
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] *= complex(real(a[j])*1.01, imag(a[j])*1.01)
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]complex128, length)
			for i := 0; i < length; i++ {
				a[i] = complex128(complex(float64(1.01), float64(1.01)))
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] *= complex(real(a[j])*1.01, imag(a[j])*1.01)
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[complex128]complex128)
			for i := 0; i < length; i++ {
				a[complex128(complex(float64(i), float64(i)))] = complex128(complex(float64(i), float64(i))) + 0.01
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] *= complex(real(a[k])*1.01, imag(a[k])*1.01)
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*complex128]complex128)
			for i := 0; i < length; i++ {
				a[new(complex128)] = complex128(complex(float64(i), float64(i))) + 0.01
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] *= complex(real(a[k])*1.01, imag(a[k])*1.01)
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan complex128)
			for i := 0; i < mods; i++ {
				go func() { a <- complex128(complex(float64(i), float64(i))) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(complex128(complex(float64(1.01), float64(1.01))))
			for i := 0; i < mods; i++ {
				a = a.(complex128) * complex(real(a.(complex128))*1.01, imag(a.(complex128))*1.01)
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "byte",
		t: func() {
			var a byte
			for i := 0; i < mods; i++ {
				a++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *byte { return new(byte) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]byte{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]byte, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[byte]byte)
			for i := 0; i < length; i++ {
				a[byte(i)] = byte(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*byte]byte)
			for i := 0; i < length; i++ {
				a[new(byte)] = byte(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan byte)
			for i := 0; i < mods; i++ {
				go func() { a <- byte(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(byte(0))
			for i := 0; i < mods; i++ {
				a = a.(byte) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "rune",
		t: func() {
			var a rune
			for i := 0; i < mods; i++ {
				a++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *rune { return new(rune) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]rune{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]rune, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[rune]rune)
			for i := 0; i < length; i++ {
				a[rune(i)] = rune(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*rune]rune)
			for i := 0; i < length; i++ {
				a[new(rune)] = rune(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan rune)
			for i := 0; i < mods; i++ {
				go func() { a <- rune(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(rune(0))
			for i := 0; i < mods; i++ {
				a = a.(rune) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "uint",
		t: func() {
			var a uint
			for i := 0; i < mods; i++ {
				a++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *uint { return new(uint) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]uint{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]uint, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[uint]uint)
			for i := 0; i < length; i++ {
				a[uint(i)] = uint(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*uint]uint)
			for i := 0; i < length; i++ {
				a[new(uint)] = uint(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan uint)
			for i := 0; i < mods; i++ {
				go func() { a <- uint(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(uint(0))
			for i := 0; i < mods; i++ {
				a = a.(uint) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "int",
		t: func() {
			var a int
			for i := 0; i < mods; i++ {
				a++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *int { return new(int) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]int{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]int, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[int]int)
			for i := 0; i < length; i++ {
				a[int(i)] = int(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*int]int)
			for i := 0; i < length; i++ {
				a[new(int)] = int(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan int)
			for i := 0; i < mods; i++ {
				go func() { a <- int(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(int(0))
			for i := 0; i < mods; i++ {
				a = a.(int) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "uintptr",
		t: func() {
			var a uintptr
			for i := 0; i < mods; i++ {
				a++
				runtime.Gosched()
			}
		},
		pointerT: func() {
			a := func() *uintptr { return new(uintptr) }()
			for i := 0; i < mods; i++ {
				*a++
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]uintptr{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]uintptr, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j]++
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[uintptr]uintptr)
			for i := 0; i < length; i++ {
				a[uintptr(i)] = uintptr(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*uintptr]uintptr)
			for i := 0; i < length; i++ {
				a[new(uintptr)] = uintptr(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k]++
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan uintptr)
			for i := 0; i < mods; i++ {
				go func() { a <- uintptr(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(uintptr(0))
			for i := 0; i < mods; i++ {
				a = a.(uintptr) + 1
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "string",
		t: func() {
			var s string
			f := func(a string) string { return a }
			for i := 0; i < mods; i++ {
				s = str(i)
				s = f(s)
			}
		},
		pointerT: func() {
			a := func() *string { return new(string) }()
			for i := 0; i < mods; i++ {
				*a = str(i)
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]string{}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] = str(i)
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]string, length)
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j] = str(i)
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[string]string)
			for i := 0; i < length; i++ {
				a[string(i)] = str(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] = str(i)
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*string]string)
			for i := 0; i < length; i++ {
				a[new(string)] = str(i)
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for k, _ := range a {
					a[k] = str(i)
					runtime.Gosched()
				}
			}
		},
		chanT: func() {
			a := make(chan string)
			for i := 0; i < mods; i++ {
				go func() { a <- str(i) }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(str(0))
			f := func(a string) string { return a }
			for i := 0; i < mods; i++ {
				a = str(i)
				a = f(a.(string))
				runtime.Gosched()
			}
		},
	},
	modifier{
		name: "structT",
		t: func() {
			s := newStructT()
			for i := 0; i < mods; i++ {
				s.u8++
				s.u16++
				s.u32++
				s.u64++
				s.i8++
				s.i16++
				s.i32++
				s.i64++
				s.f32 *= 1.01
				s.f64 *= 1.01
				s.c64 = complex(real(s.c64)*1.01, imag(s.c64)*1.01)
				s.c128 = complex(real(s.c128)*1.01, imag(s.c128)*1.01)
				s.b++
				s.r++
				s.u++
				s.in++
				s.uip++
				s.s = str(i)
				runtime.Gosched()
			}
		},
		pointerT: func() {
			s := func() *structT {
				t := newStructT()
				return &t
			}()
			for i := 0; i < mods; i++ {
				s.u8++
				s.u16++
				s.u32++
				s.u64++
				s.i8++
				s.i16++
				s.i32++
				s.i64++
				s.f32 *= 1.01
				s.f64 *= 1.01
				s.c64 = complex(real(s.c64)*1.01, imag(s.c64)*1.01)
				s.c128 = complex(real(s.c128)*1.01, imag(s.c128)*1.01)
				s.b++
				s.r++
				s.u++
				s.in++
				s.uip++
				s.s = str(i)
				runtime.Gosched()
			}
		},
		arrayT: func() {
			a := [length]structT{}
			for i := 0; i < len(a); i++ {
				a[i] = newStructT()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j].u8++
					a[j].u16++
					a[j].u32++
					a[j].u64++
					a[j].i8++
					a[j].i16++
					a[j].i32++
					a[j].i64++
					a[j].f32 *= 1.01
					a[j].f64 *= 1.01
					a[j].c64 = complex(real(a[j].c64)*1.01, imag(a[j].c64)*1.01)
					a[j].c128 = complex(real(a[j].c128)*1.01, imag(a[j].c128)*1.01)
					a[j].b++
					a[j].r++
					a[j].u++
					a[j].in++
					a[j].uip++
					a[j].s = str(i)
					runtime.Gosched()
				}
			}
		},
		sliceT: func() {
			a := make([]structT, length)
			for i := 0; i < len(a); i++ {
				a[i] = newStructT()
			}
			for i := 0; i < mods; i++ {
				for j := 0; j < len(a); j++ {
					a[j].u8++
					a[j].u16++
					a[j].u32++
					a[j].u64++
					a[j].i8++
					a[j].i16++
					a[j].i32++
					a[j].i64++
					a[j].f32 *= 1.01
					a[j].f64 *= 1.01
					a[j].c64 = complex(real(a[j].c64)*1.01, imag(a[j].c64)*1.01)
					a[j].c128 = complex(real(a[j].c128)*1.01, imag(a[j].c128)*1.01)
					a[j].b++
					a[j].r++
					a[j].u++
					a[j].in++
					a[j].uip++
					a[j].s = str(i)
					runtime.Gosched()
				}
			}
		},
		mapT: func() {
			a := make(map[structT]structT)
			for i := 0; i < length; i++ {
				m := newStructT()
				m.in = i
				a[m] = newStructT()
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j, _ := range a {
					m := a[j]
					m.u8++
					m.u16++
					m.u32++
					m.u64++
					m.i8++
					m.i16++
					m.i32++
					m.i64++
					m.f32 *= 1.01
					m.f64 *= 1.01
					m.c64 = complex(real(a[j].c64)*1.01, imag(a[j].c64)*1.01)
					m.c128 = complex(real(a[j].c128)*1.01, imag(a[j].c128)*1.01)
					m.b++
					m.r++
					m.u++
					m.in++
					m.uip++
					m.s = str(i)
					a[j] = m
					runtime.Gosched()
				}
				runtime.Gosched()
			}
		},
		mapPointerKeyT: func() {
			a := make(map[*structT]structT)
			f := func() *structT {
				m := newStructT()
				return &m
			}
			for i := 0; i < length; i++ {
				m := f()
				m.in = i
				a[m] = newStructT()
				runtime.Gosched()
			}
			for i := 0; i < mods; i++ {
				for j, _ := range a {
					m := a[j]
					m.u8++
					m.u16++
					m.u32++
					m.u64++
					m.i8++
					m.i16++
					m.i32++
					m.i64++
					m.f32 *= 1.01
					m.f64 *= 1.01
					m.c64 = complex(real(a[j].c64)*1.01, imag(a[j].c64)*1.01)
					m.c128 = complex(real(a[j].c128)*1.01, imag(a[j].c128)*1.01)
					m.b++
					m.r++
					m.u++
					m.in++
					m.uip++
					m.s = str(i)
					a[j] = m
					runtime.Gosched()
				}
				runtime.Gosched()
			}
		},
		chanT: func() {
			a := make(chan structT)
			for i := 0; i < mods; i++ {
				go func() { a <- newStructT() }()
				<-a
				runtime.Gosched()
			}
		},
		interfaceT: func() {
			a := interface{}(newStructT())
			for i := 0; i < mods; i++ {
				a = a.(structT)
				runtime.Gosched()
			}
		},
	},
}

type structT struct {
	u8   uint8
	u16  uint16
	u32  uint32
	u64  uint64
	i8   int8
	i16  int16
	i32  int32
	i64  int64
	f32  float32
	f64  float64
	c64  complex64
	c128 complex128
	b    byte
	r    rune
	u    uint
	in   int
	uip  uintptr
	s    string
}

func newStructT() structT {
	return structT{
		f32:  1.01,
		f64:  1.01,
		c64:  complex(float32(1.01), float32(1.01)),
		c128: complex(float64(1.01), float64(1.01)),
	}
}

func str(in int) string {
	switch in % 3 {
	case 0:
		return "Hello"
	case 1:
		return "world"
	case 2:
		return "!"
	}
	return "?"
}
