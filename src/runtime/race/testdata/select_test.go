// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"runtime"
	"testing"
)

func TestNoRaceSelect1(t *testing.T) {
	var x int
	compl := make(chan bool)
	c := make(chan bool)
	c1 := make(chan bool)

	go func() {
		x = 1
		// At least two channels are needed because
		// otherwise the compiler optimizes select out.
		// See comment in runtime/select.go:^func selectgoImpl.
		select {
		case c <- true:
		case c1 <- true:
		}
		compl <- true
	}()
	select {
	case <-c:
	case c1 <- true:
	}
	x = 2
	<-compl
}

func TestNoRaceSelect2(t *testing.T) {
	var x int
	compl := make(chan bool)
	c := make(chan bool)
	c1 := make(chan bool)
	go func() {
		select {
		case <-c:
		case <-c1:
		}
		x = 1
		compl <- true
	}()
	x = 2
	close(c)
	runtime.Gosched()
	<-compl
}

func TestNoRaceSelect3(t *testing.T) {
	var x int
	compl := make(chan bool)
	c := make(chan bool, 10)
	c1 := make(chan bool)
	go func() {
		x = 1
		select {
		case c <- true:
		case <-c1:
		}
		compl <- true
	}()
	<-c
	x = 2
	<-compl
}

func TestNoRaceSelect4(t *testing.T) {
	type Task struct {
		f    func()
		done chan bool
	}

	queue := make(chan Task)
	dummy := make(chan bool)

	go func() {
		for {
			select {
			case t := <-queue:
				t.f()
				t.done <- true
			}
		}
	}()

	doit := func(f func()) {
		done := make(chan bool, 1)
		select {
		case queue <- Task{f, done}:
		case <-dummy:
		}
		select {
		case <-done:
		case <-dummy:
		}
	}

	var x int
	doit(func() {
		x = 1
	})
	_ = x
}

func TestNoRaceSelect5(t *testing.T) {
	test := func(sel, needSched bool) {
		var x int
		ch := make(chan bool)
		c1 := make(chan bool)

		done := make(chan bool, 2)
		go func() {
			if needSched {
				runtime.Gosched()
			}
			// println(1)
			x = 1
			if sel {
				select {
				case ch <- true:
				case <-c1:
				}
			} else {
				ch <- true
			}
			done <- true
		}()

		go func() {
			// println(2)
			if sel {
				select {
				case <-ch:
				case <-c1:
				}
			} else {
				<-ch
			}
			x = 1
			done <- true
		}()
		<-done
		<-done
	}

	test(true, true)
	test(true, false)
	test(false, true)
	test(false, false)
}

func TestRaceSelect1(t *testing.T) {
	var x int
	compl := make(chan bool, 2)
	c := make(chan bool)
	c1 := make(chan bool)

	go func() {
		<-c
		<-c
	}()
	f := func() {
		select {
		case c <- true:
		case c1 <- true:
		}
		x = 1
		compl <- true
	}
	go f()
	go f()
	<-compl
	<-compl
}

func TestRaceSelect2(t *testing.T) {
	var x int
	compl := make(chan bool)
	c := make(chan bool)
	c1 := make(chan bool)
	go func() {
		x = 1
		select {
		case <-c:
		case <-c1:
		}
		compl <- true
	}()
	close(c)
	x = 2
	<-compl
}

func TestRaceSelect3(t *testing.T) {
	var x int
	compl := make(chan bool)
	c := make(chan bool)
	c1 := make(chan bool)
	go func() {
		x = 1
		select {
		case c <- true:
		case c1 <- true:
		}
		compl <- true
	}()
	x = 2
	select {
	case <-c:
	}
	<-compl
}

func TestRaceSelect4(t *testing.T) {
	done := make(chan bool, 1)
	var x int
	go func() {
		select {
		default:
			x = 2
		}
		done <- true
	}()
	_ = x
	<-done
}

// The idea behind this test:
// there are two variables, access to one
// of them is synchronized, access to the other
// is not.
// Select must (unconditionally) choose the non-synchronized variable
// thus causing exactly one race.
// Currently this test doesn't look like it accomplishes
// this goal.
func TestRaceSelect5(t *testing.T) {
	done := make(chan bool, 1)
	c1 := make(chan bool, 1)
	c2 := make(chan bool)
	var x, y int
	go func() {
		select {
		case c1 <- true:
			x = 1
		case c2 <- true:
			y = 1
		}
		done <- true
	}()
	_ = x
	_ = y
	<-done
}

// select statements may introduce
// flakiness: whether this test contains
// a race depends on the scheduling
// (some may argue that the code contains
// this race by definition)
/*
func TestFlakyDefault(t *testing.T) {
	var x int
	c := make(chan bool, 1)
	done := make(chan bool, 1)
	go func() {
		select {
		case <-c:
			x = 2
		default:
			x = 3
		}
		done <- true
	}()
	x = 1
	c <- true
	_ = x
	<-done
}
*/
