// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"testing"
)

func TestRaceSliceRW(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]int, 2)
	go func() {
		a[1] = 1
		ch <- true
	}()
	_ = a[1]
	<-ch
}

func TestNoRaceSliceRW(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]int, 2)
	go func() {
		a[0] = 1
		ch <- true
	}()
	_ = a[1]
	<-ch
}

func TestRaceSliceWW(t *testing.T) {
	a := make([]int, 10)
	ch := make(chan bool, 1)
	go func() {
		a[1] = 1
		ch <- true
	}()
	a[1] = 2
	<-ch
}

func TestNoRaceArrayWW(t *testing.T) {
	var a [5]int
	ch := make(chan bool, 1)
	go func() {
		a[0] = 1
		ch <- true
	}()
	a[1] = 2
	<-ch
}

func TestRaceArrayWW(t *testing.T) {
	var a [5]int
	ch := make(chan bool, 1)
	go func() {
		a[1] = 1
		ch <- true
	}()
	a[1] = 2
	<-ch
}

func TestNoRaceSliceWriteLen(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]bool, 1)
	go func() {
		a[0] = true
		ch <- true
	}()
	_ = len(a)
	<-ch
}

func TestNoRaceSliceWriteCap(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]uint64, 100)
	go func() {
		a[50] = 123
		ch <- true
	}()
	_ = cap(a)
	<-ch
}

func TestRaceSliceCopyRead(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]int, 10)
	b := make([]int, 10)
	go func() {
		_ = a[5]
		ch <- true
	}()
	copy(a, b)
	<-ch
}

func TestNoRaceSliceWriteCopy(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]int, 10)
	b := make([]int, 10)
	go func() {
		a[5] = 1
		ch <- true
	}()
	copy(a[:5], b[:5])
	<-ch
}

func TestRaceSliceCopyWrite2(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]int, 10)
	b := make([]int, 10)
	go func() {
		b[5] = 1
		ch <- true
	}()
	copy(a, b)
	<-ch
}

func TestRaceSliceCopyWrite3(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]byte, 10)
	go func() {
		a[7] = 1
		ch <- true
	}()
	copy(a, "qwertyqwerty")
	<-ch
}

func TestNoRaceSliceCopyRead(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]int, 10)
	b := make([]int, 10)
	go func() {
		_ = b[5]
		ch <- true
	}()
	copy(a, b)
	<-ch
}

func TestNoRaceSliceWriteSlice2(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]float64, 10)
	go func() {
		a[2] = 1.0
		ch <- true
	}()
	_ = a[0:5]
	<-ch
}

func TestRaceSliceWriteSlice(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]float64, 10)
	go func() {
		a[2] = 1.0
		ch <- true
	}()
	a = a[5:10]
	<-ch
}

func TestNoRaceSliceWriteSlice(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]float64, 10)
	go func() {
		a[2] = 1.0
		ch <- true
	}()
	_ = a[5:10]
	<-ch
}

func TestNoRaceSliceLenCap(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]struct{}, 10)
	go func() {
		_ = len(a)
		ch <- true
	}()
	_ = cap(a)
	<-ch
}

func TestNoRaceStructSlicesRangeWrite(t *testing.T) {
	type Str struct {
		a []int
		b []int
	}
	ch := make(chan bool, 1)
	var s Str
	s.a = make([]int, 10)
	s.b = make([]int, 10)
	go func() {
		for _ = range s.a {
		}
		ch <- true
	}()
	s.b[5] = 5
	<-ch
}

func TestRaceSliceDifferent(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	s2 := s
	go func() {
		s[3] = 3
		c <- true
	}()
	// false negative because s2 is PAUTO w/o PHEAP
	// so we do not instrument it
	s2[3] = 3
	<-c
}

func TestRaceSliceRangeWrite(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		s[3] = 3
		c <- true
	}()
	for _, v := range s {
		_ = v
	}
	<-c
}

func TestNoRaceSliceRangeWrite(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		s[3] = 3
		c <- true
	}()
	for _ = range s {
	}
	<-c
}

func TestRaceSliceRangeAppend(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		s = append(s, 3)
		c <- true
	}()
	for _ = range s {
	}
	<-c
}

func TestNoRaceSliceRangeAppend(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		_ = append(s, 3)
		c <- true
	}()
	for _ = range s {
	}
	<-c
}

func TestRaceSliceVarWrite(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		s[3] = 3
		c <- true
	}()
	s = make([]int, 20)
	<-c
}

func TestRaceSliceVarRead(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		_ = s[3]
		c <- true
	}()
	s = make([]int, 20)
	<-c
}

func TestRaceSliceVarRange(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		for _ = range s {
		}
		c <- true
	}()
	s = make([]int, 20)
	<-c
}

func TestRaceSliceVarAppend(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		_ = append(s, 10)
		c <- true
	}()
	s = make([]int, 20)
	<-c
}

func TestRaceSliceVarCopy(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		s2 := make([]int, 10)
		copy(s, s2)
		c <- true
	}()
	s = make([]int, 20)
	<-c
}

func TestRaceSliceVarCopy2(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		s2 := make([]int, 10)
		copy(s2, s)
		c <- true
	}()
	s = make([]int, 20)
	<-c
}

func TestRaceSliceAppend(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10, 20)
	go func() {
		_ = append(s, 1)
		c <- true
	}()
	_ = append(s, 2)
	<-c
}

func TestRaceSliceAppendWrite(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		_ = append(s, 1)
		c <- true
	}()
	s[0] = 42
	<-c
}

func TestRaceSliceAppendSlice(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	go func() {
		s2 := make([]int, 10)
		_ = append(s, s2...)
		c <- true
	}()
	s[0] = 42
	<-c
}

func TestRaceSliceAppendSlice2(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	s2foobar := make([]int, 10)
	go func() {
		_ = append(s, s2foobar...)
		c <- true
	}()
	s2foobar[5] = 42
	<-c
}

func TestRaceSliceAppendString(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]byte, 10)
	go func() {
		_ = append(s, "qwerty"...)
		c <- true
	}()
	s[0] = 42
	<-c
}

func TestNoRaceSliceIndexAccess(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	v := 0
	go func() {
		_ = v
		c <- true
	}()
	s[v] = 1
	<-c
}

func TestNoRaceSliceIndexAccess2(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	v := 0
	go func() {
		_ = v
		c <- true
	}()
	_ = s[v]
	<-c
}

func TestRaceSliceIndexAccess(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	v := 0
	go func() {
		v = 1
		c <- true
	}()
	s[v] = 1
	<-c
}

func TestRaceSliceIndexAccess2(t *testing.T) {
	c := make(chan bool, 1)
	s := make([]int, 10)
	v := 0
	go func() {
		v = 1
		c <- true
	}()
	_ = s[v]
	<-c
}

func TestRaceSliceByteToString(t *testing.T) {
	c := make(chan string)
	s := make([]byte, 10)
	go func() {
		c <- string(s)
	}()
	s[0] = 42
	<-c
}

func TestRaceSliceRuneToString(t *testing.T) {
	c := make(chan string)
	s := make([]rune, 10)
	go func() {
		c <- string(s)
	}()
	s[9] = 42
	<-c
}

func TestRaceConcatString(t *testing.T) {
	s := "hello"
	c := make(chan string, 1)
	go func() {
		c <- s + " world"
	}()
	s = "world"
	<-c
}

func TestRaceCompareString(t *testing.T) {
	s1 := "hello"
	s2 := "world"
	c := make(chan bool, 1)
	go func() {
		c <- s1 == s2
	}()
	s1 = s2
	<-c
}
