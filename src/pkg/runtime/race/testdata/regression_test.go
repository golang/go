// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code patterns that caused problems in the past.

package race_test

import (
	"testing"
)

type LogImpl struct {
	x int
}

func NewLog() (l LogImpl) {
	c := make(chan bool)
	go func() {
		_ = l
		c <- true
	}()
	l = LogImpl{}
	<-c
	return
}

var _ LogImpl = NewLog()

func MakeMap() map[int]int {
	return make(map[int]int)
}

func InstrumentMapLen() {
	_ = len(MakeMap())
}

func InstrumentMapLen2() {
	m := make(map[int]map[int]int)
	_ = len(m[0])
}

func InstrumentMapLen3() {
	m := make(map[int]*map[int]int)
	_ = len(*m[0])
}

func TestRaceUnaddressableMapLen(t *testing.T) {
	m := make(map[int]map[int]int)
	ch := make(chan int, 1)
	m[0] = make(map[int]int)
	go func() {
		_ = len(m[0])
		ch <- 0
	}()
	m[0][0] = 1
	<-ch
}

type Rect struct {
	x, y int
}

type Image struct {
	min, max Rect
}

func NewImage() Image {
	var pleaseDoNotInlineMe stack
	pleaseDoNotInlineMe.push(1)
	_ = pleaseDoNotInlineMe.pop()
	return Image{}
}

func AddrOfTemp() {
	_ = NewImage().min
}

type TypeID int

func (t *TypeID) encodeType(x int) (tt TypeID, err error) {
	switch x {
	case 0:
		return t.encodeType(x * x)
	}
	return 0, nil
}

type stack []int

func (s *stack) push(x int) {
	*s = append(*s, x)
}

func (s *stack) pop() int {
	i := len(*s)
	n := (*s)[i-1]
	*s = (*s)[:i-1]
	return n
}

func TestNoRaceStackPushPop(t *testing.T) {
	var s stack
	go func(s *stack) {}(&s)
	s.push(1)
	x := s.pop()
	_ = x
}

type RpcChan struct {
	c chan bool
}

var makeChanCalls int

func makeChan() *RpcChan {
	var pleaseDoNotInlineMe stack
	pleaseDoNotInlineMe.push(1)
	_ = pleaseDoNotInlineMe.pop()

	makeChanCalls++
	c := &RpcChan{make(chan bool, 1)}
	c.c <- true
	return c
}

func call() bool {
	x := <-makeChan().c
	return x
}

func TestNoRaceRpcChan(t *testing.T) {
	makeChanCalls = 0
	_ = call()
	if makeChanCalls != 1 {
		t.Fatalf("makeChanCalls %d, expected 1\n", makeChanCalls)
	}
}

func divInSlice() {
	v := make([]int64, 10)
	i := 1
	_ = v[(i*4)/3]
}

func TestNoRaceReturn(t *testing.T) {
	c := make(chan int)
	noRaceReturn(c)
	<-c
}

// Return used to do an implicit a = a, causing a read/write race
// with the goroutine. Compiler has an optimization to avoid that now.
// See issue 4014.
func noRaceReturn(c chan int) (a, b int) {
	a = 42
	go func() {
		_ = a
		c <- 1
	}()
	return a, 10
}
