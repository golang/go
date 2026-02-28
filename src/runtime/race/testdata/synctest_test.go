// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"internal/synctest"
	"testing"
	"time"
)

func TestRaceSynctestGoroutinesExit(t *testing.T) {
	synctest.Run(func() {
		x := 0
		_ = x
		f := func() {
			x = 1
		}
		go f()
		go f()
	})
}

func TestNoRaceSynctestGoroutinesExit(t *testing.T) {
	synctest.Run(func() {
		x := 0
		_ = x
		f := func() {
			x = 1
		}
		go f()
		synctest.Wait()
		go f()
	})
}

func TestRaceSynctestGoroutinesRecv(t *testing.T) {
	synctest.Run(func() {
		x := 0
		_ = x
		ch := make(chan struct{})
		f := func() {
			x = 1
			<-ch
		}
		go f()
		go f()
		close(ch)
	})
}

func TestRaceSynctestGoroutinesUnblocked(t *testing.T) {
	synctest.Run(func() {
		x := 0
		_ = x
		ch := make(chan struct{})
		f := func() {
			<-ch
			x = 1
		}
		go f()
		go f()
		close(ch)
	})
}

func TestRaceSynctestGoroutinesSleep(t *testing.T) {
	synctest.Run(func() {
		x := 0
		_ = x
		go func() {
			time.Sleep(1 * time.Second)
			x = 1
			time.Sleep(2 * time.Second)
		}()
		go func() {
			time.Sleep(2 * time.Second)
			x = 1
			time.Sleep(1 * time.Second)
		}()
		time.Sleep(5 * time.Second)
	})
}

func TestRaceSynctestTimers(t *testing.T) {
	synctest.Run(func() {
		x := 0
		_ = x
		f := func() {
			x = 1
		}
		time.AfterFunc(1*time.Second, f)
		time.AfterFunc(2*time.Second, f)
		time.Sleep(5 * time.Second)
	})
}
