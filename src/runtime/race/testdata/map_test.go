// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"testing"
)

func TestRaceMapRW(t *testing.T) {
	m := make(map[int]int)
	ch := make(chan bool, 1)
	go func() {
		_ = m[1]
		ch <- true
	}()
	m[1] = 1
	<-ch
}

func TestRaceMapRW2(t *testing.T) {
	m := make(map[int]int)
	ch := make(chan bool, 1)
	go func() {
		_, _ = m[1]
		ch <- true
	}()
	m[1] = 1
	<-ch
}

func TestRaceMapRWArray(t *testing.T) {
	// Check instrumentation of unaddressable arrays (issue 4578).
	m := make(map[int][2]int)
	ch := make(chan bool, 1)
	go func() {
		_ = m[1][1]
		ch <- true
	}()
	m[2] = [2]int{1, 2}
	<-ch
}

func TestNoRaceMapRR(t *testing.T) {
	m := make(map[int]int)
	ch := make(chan bool, 1)
	go func() {
		_, _ = m[1]
		ch <- true
	}()
	_ = m[1]
	<-ch
}

func TestRaceMapRange(t *testing.T) {
	m := make(map[int]int)
	ch := make(chan bool, 1)
	go func() {
		for range m {
		}
		ch <- true
	}()
	m[1] = 1
	<-ch
}

func TestRaceMapRange2(t *testing.T) {
	m := make(map[int]int)
	ch := make(chan bool, 1)
	go func() {
		for range m {
		}
		ch <- true
	}()
	m[1] = 1
	<-ch
}

func TestNoRaceMapRangeRange(t *testing.T) {
	m := make(map[int]int)
	// now the map is not empty and range triggers an event
	// should work without this (as in other tests)
	// so it is suspicious if this test passes and others don't
	m[0] = 0
	ch := make(chan bool, 1)
	go func() {
		for range m {
		}
		ch <- true
	}()
	for range m {
	}
	<-ch
}

func TestRaceMapLen(t *testing.T) {
	m := make(map[string]bool)
	ch := make(chan bool, 1)
	go func() {
		_ = len(m)
		ch <- true
	}()
	m[""] = true
	<-ch
}

func TestRaceMapDelete(t *testing.T) {
	m := make(map[string]bool)
	ch := make(chan bool, 1)
	go func() {
		delete(m, "")
		ch <- true
	}()
	m[""] = true
	<-ch
}

func TestRaceMapLenDelete(t *testing.T) {
	m := make(map[string]bool)
	ch := make(chan bool, 1)
	go func() {
		delete(m, "a")
		ch <- true
	}()
	_ = len(m)
	<-ch
}

func TestRaceMapVariable(t *testing.T) {
	ch := make(chan bool, 1)
	m := make(map[int]int)
	_ = m
	go func() {
		m = make(map[int]int)
		ch <- true
	}()
	m = make(map[int]int)
	<-ch
}

func TestRaceMapVariable2(t *testing.T) {
	ch := make(chan bool, 1)
	m := make(map[int]int)
	go func() {
		m[1] = 1
		ch <- true
	}()
	m = make(map[int]int)
	<-ch
}

func TestRaceMapVariable3(t *testing.T) {
	ch := make(chan bool, 1)
	m := make(map[int]int)
	go func() {
		_ = m[1]
		ch <- true
	}()
	m = make(map[int]int)
	<-ch
}

type Big struct {
	x [17]int32
}

func TestRaceMapLookupPartKey(t *testing.T) {
	k := &Big{}
	m := make(map[Big]bool)
	ch := make(chan bool, 1)
	go func() {
		k.x[8] = 1
		ch <- true
	}()
	_ = m[*k]
	<-ch
}

func TestRaceMapLookupPartKey2(t *testing.T) {
	k := &Big{}
	m := make(map[Big]bool)
	ch := make(chan bool, 1)
	go func() {
		k.x[8] = 1
		ch <- true
	}()
	_, _ = m[*k]
	<-ch
}
func TestRaceMapDeletePartKey(t *testing.T) {
	k := &Big{}
	m := make(map[Big]bool)
	ch := make(chan bool, 1)
	go func() {
		k.x[8] = 1
		ch <- true
	}()
	delete(m, *k)
	<-ch
}

func TestRaceMapInsertPartKey(t *testing.T) {
	k := &Big{}
	m := make(map[Big]bool)
	ch := make(chan bool, 1)
	go func() {
		k.x[8] = 1
		ch <- true
	}()
	m[*k] = true
	<-ch
}

func TestRaceMapInsertPartVal(t *testing.T) {
	v := &Big{}
	m := make(map[int]Big)
	ch := make(chan bool, 1)
	go func() {
		v.x[8] = 1
		ch <- true
	}()
	m[1] = *v
	<-ch
}

// Test for issue 7561.
func TestRaceMapAssignMultipleReturn(t *testing.T) {
	connect := func() (int, error) { return 42, nil }
	conns := make(map[int][]int)
	conns[1] = []int{0}
	ch := make(chan bool, 1)
	var err error
	_ = err
	go func() {
		conns[1][0], err = connect()
		ch <- true
	}()
	x := conns[1][0]
	_ = x
	<-ch
}

// BigKey and BigVal must be larger than 256 bytes,
// so that compiler sets KindGCProg for them.
type BigKey [1000]*int

type BigVal struct {
	x int
	y [1000]*int
}

func TestRaceMapBigKeyAccess1(t *testing.T) {
	m := make(map[BigKey]int)
	var k BigKey
	ch := make(chan bool, 1)
	go func() {
		_ = m[k]
		ch <- true
	}()
	k[30] = new(int)
	<-ch
}

func TestRaceMapBigKeyAccess2(t *testing.T) {
	m := make(map[BigKey]int)
	var k BigKey
	ch := make(chan bool, 1)
	go func() {
		_, _ = m[k]
		ch <- true
	}()
	k[30] = new(int)
	<-ch
}

func TestRaceMapBigKeyInsert(t *testing.T) {
	m := make(map[BigKey]int)
	var k BigKey
	ch := make(chan bool, 1)
	go func() {
		m[k] = 1
		ch <- true
	}()
	k[30] = new(int)
	<-ch
}

func TestRaceMapBigKeyDelete(t *testing.T) {
	m := make(map[BigKey]int)
	var k BigKey
	ch := make(chan bool, 1)
	go func() {
		delete(m, k)
		ch <- true
	}()
	k[30] = new(int)
	<-ch
}

func TestRaceMapBigValInsert(t *testing.T) {
	m := make(map[int]BigVal)
	var v BigVal
	ch := make(chan bool, 1)
	go func() {
		m[1] = v
		ch <- true
	}()
	v.y[30] = new(int)
	<-ch
}

func TestRaceMapBigValAccess1(t *testing.T) {
	m := make(map[int]BigVal)
	var v BigVal
	ch := make(chan bool, 1)
	go func() {
		v = m[1]
		ch <- true
	}()
	v.y[30] = new(int)
	<-ch
}

func TestRaceMapBigValAccess2(t *testing.T) {
	m := make(map[int]BigVal)
	var v BigVal
	ch := make(chan bool, 1)
	go func() {
		v, _ = m[1]
		ch <- true
	}()
	v.y[30] = new(int)
	<-ch
}
