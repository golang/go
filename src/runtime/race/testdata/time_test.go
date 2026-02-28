// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"testing"
	"time"
)

func TestNoRaceAfterFunc(_ *testing.T) {
	v := 0
	_ = v
	c := make(chan int)
	f := func() {
		v = 1
		c <- 0
	}
	v = 2
	time.AfterFunc(1, f)
	<-c
	v = 3
}

func TestNoRaceAfterFuncReset(_ *testing.T) {
	v := 0
	_ = v
	c := make(chan int)
	f := func() {
		v = 1
		c <- 0
	}
	t := time.AfterFunc(time.Hour, f)
	t.Stop()
	v = 2
	t.Reset(1)
	<-c
	v = 3
}

func TestNoRaceTimer(_ *testing.T) {
	v := 0
	_ = v
	c := make(chan int)
	f := func() {
		v = 1
		c <- 0
	}
	v = 2
	t := time.NewTimer(1)
	go func() {
		<-t.C
		f()
	}()
	<-c
	v = 3
}

func TestNoRaceTimerReset(_ *testing.T) {
	v := 0
	_ = v
	c := make(chan int)
	f := func() {
		v = 1
		c <- 0
	}
	t := time.NewTimer(time.Hour)
	go func() {
		<-t.C
		f()
	}()
	t.Stop()
	v = 2
	t.Reset(1)
	<-c
	v = 3
}

func TestNoRaceTicker(_ *testing.T) {
	v := 0
	_ = v
	c := make(chan int)
	f := func() {
		v = 1
		c <- 0
	}
	v = 2
	t := time.NewTicker(1)
	go func() {
		<-t.C
		f()
	}()
	<-c
	v = 3
}

func TestNoRaceTickerReset(_ *testing.T) {
	v := 0
	_ = v
	c := make(chan int)
	f := func() {
		v = 1
		c <- 0
	}
	t := time.NewTicker(time.Hour)
	go func() {
		<-t.C
		f()
	}()
	t.Stop()
	v = 2
	t.Reset(1)
	<-c
	v = 3
}
