// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql

import (
	"testing"
	"testing/synctest"
)

func TestClosingMutex(t *testing.T) {
	start := func(t *testing.T, f func()) func() bool {
		done := false
		go func() {
			f()
			done = true
		}()
		return func() bool {
			synctest.Wait()
			return done
		}
	}

	synctest.Test(t, func(t *testing.T) {
		var m closingMutex

		// RLock does not block RLock.
		m.RLock()
		m.RLock()
		m.RUnlock()
		m.RUnlock()

		// RLock blocks Lock.
		m.RLock()
		lock1Done := start(t, m.Lock)
		if lock1Done() {
			t.Fatalf("m.Lock(): succeeded on RLocked mutex")
		}
		m.RLock()
		m.RUnlock()
		if lock1Done() {
			t.Fatalf("m.Lock(): succeeded after one RUnlock, one RLock remains")
		}
		m.RUnlock()
		if !lock1Done() {
			t.Fatalf("m.Lock(): still blocking after all RUnlocks")
		}
		m.Unlock()

		// Lock blocks RLock.
		m.Lock()
		rlock1Done := start(t, m.RLock)
		rlock2Done := start(t, m.RLock)
		if rlock1Done() || rlock2Done() {
			t.Fatalf("m.RLock(): succeeded on Locked mutex")
		}
		m.Unlock()
		if !rlock1Done() || !rlock2Done() {
			t.Fatalf("m.RLock(): succeeded on Locked mutex")
		}
		m.RUnlock()
		m.RUnlock()

		// Lock blocks Lock.
		m.Lock()
		lock2Done := start(t, m.Lock)
		if lock2Done() {
			t.Fatalf("m.Lock(): succeeded on Locked mutex")
		}
		m.Unlock()
		if !lock2Done() {
			t.Fatalf("m.Lock(): still blocking after Unlock")
		}
		m.Unlock()

		// Lock on RLocked mutex does not block RLock.
		m.RLock()
		lock3Done := start(t, m.Lock)
		if lock3Done() {
			t.Fatalf("m.Lock(): succeeded on RLocked mutex")
		}
		m.RLock()
		m.RUnlock()
		m.RUnlock()
		if !lock3Done() {
			t.Fatalf("m.Lock(): still blocking after RUnlock")
		}
		m.Unlock()

	})
}

func TestClosingMutexPanics(t *testing.T) {
	for _, test := range []struct {
		name string
		f    func()
	}{{
		name: "double RUnlock",
		f: func() {
			var m closingMutex
			m.RLock()
			m.RUnlock()
			m.RUnlock()
		},
	}, {
		name: "double Unlock",
		f: func() {
			var m closingMutex
			m.Lock()
			m.Unlock()
			m.Unlock()
		},
	}} {
		var got any
		func() {
			defer func() {
				got = recover()
			}()
			test.f()
		}()
		if got == nil {
			t.Errorf("no panic, want one")
		}
	}
}
