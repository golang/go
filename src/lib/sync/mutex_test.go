// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GOMAXPROCS=10 gotest

package sync

import (
	"sync";
	"testing"
)

func HammerSemaphore(s *int32, cdone chan bool) {
	for i := 0; i < 1000; i++ {
		sys.semacquire(s);
		sys.semrelease(s);
	}
	cdone <- true;
}

export func TestSemaphore(t *testing.T) {
	s := new(*int32);
	*s = 1;
	c := new(chan bool);
	for i := 0; i < 10; i++ {
		go HammerSemaphore(s, c);
	}
	for i := 0; i < 10; i++ {
		<-c;
	}
}


func HammerMutex(m *Mutex, cdone chan bool) {
	for i := 0; i < 1000; i++ {
		m.Lock();
		m.Unlock();
	}
	cdone <- true;
}

export func TestMutex(t *testing.T) {
	m := new(*Mutex);
	c := new(chan bool);
	for i := 0; i < 10; i++ {
		go HammerMutex(m, c);
	}
	for i := 0; i < 10; i++ {
		<-c;
	}
}

