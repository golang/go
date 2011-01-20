// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
)


const memTestN = 1000000


func s(n uint64) string {
	str := fmt.Sprintf("%d", n)
	lens := len(str)
	a := make([]string, (lens+2)/3)
	start := lens
	for i := range a {
		start -= 3
		if start < 0 {
			start = 0
		}
		a[len(a)-i-1] = str[start:lens]
		lens -= 3
	}
	return strings.Join(a, " ")
}


func TestVectorNums(t *testing.T) {
	var v Vector
	c := int(0)
	runtime.GC()
	m0 := runtime.MemStats
	v.Resize(memTestN, memTestN)
	for i := 0; i < memTestN; i++ {
		v.Set(i, c)
	}
	runtime.GC()
	m := runtime.MemStats
	v.Resize(0, 0)
	runtime.GC()
	n := m.Alloc - m0.Alloc
	t.Logf("%T.Push(%#v), n = %s: Alloc/n = %.2f\n", v, c, s(memTestN), float64(n)/memTestN)
}


func TestIntVectorNums(t *testing.T) {
	var v IntVector
	c := int(0)
	runtime.GC()
	m0 := runtime.MemStats
	v.Resize(memTestN, memTestN)
	for i := 0; i < memTestN; i++ {
		v.Set(i, c)
	}
	runtime.GC()
	m := runtime.MemStats
	v.Resize(0, 0)
	runtime.GC()
	n := m.Alloc - m0.Alloc
	t.Logf("%T.Push(%#v), n = %s: Alloc/n = %.2f\n", v, c, s(memTestN), float64(n)/memTestN)
}


func TestStringVectorNums(t *testing.T) {
	var v StringVector
	c := ""
	runtime.GC()
	m0 := runtime.MemStats
	v.Resize(memTestN, memTestN)
	for i := 0; i < memTestN; i++ {
		v.Set(i, c)
	}
	runtime.GC()
	m := runtime.MemStats
	v.Resize(0, 0)
	runtime.GC()
	n := m.Alloc - m0.Alloc
	t.Logf("%T.Push(%#v), n = %s: Alloc/n = %.2f\n", v, c, s(memTestN), float64(n)/memTestN)
}


func BenchmarkVectorNums(b *testing.B) {
	c := int(0)
	var v Vector
	b.StopTimer()
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		v.Push(c)
	}
}


func BenchmarkIntVectorNums(b *testing.B) {
	c := int(0)
	var v IntVector
	b.StopTimer()
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		v.Push(c)
	}
}


func BenchmarkStringVectorNums(b *testing.B) {
	c := ""
	var v StringVector
	b.StopTimer()
	runtime.GC()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		v.Push(c)
	}
}
