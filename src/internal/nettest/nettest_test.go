// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest_test

import (
	"errors"
	"internal/nettest"
	"net"
	"os"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

var (
	_ net.Conn       = (*nettest.Conn)(nil)
	_ net.Listener   = (*nettest.Listener)(nil)
	_ net.PacketConn = (*nettest.PacketConn)(nil)
)

func synctestSubtest(t *testing.T, name string, f func(t *testing.T)) {
	t.Run(name, func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			f(t)
		})
	})
}

// A deadlineTest describes an operation which blocks until a deadline,
// and a separate operation which unblocks it.
type deadlineTest struct {
	what        string                // name of the blocking op; e.g., "Read"
	block       func() error          // blocking op; e.g., reading from a conn
	unblock     func()                // unblocking op; e.g. writing to the other side
	setDeadline func(d time.Duration) // deadline func; e.g., SetReadDeadline
}

// testDeadline tests a variety of scenarios involving deadlines.
func testDeadline(t *testing.T, setup func() deadlineTest) {
	synctestSubtest(t, "no deadline", func(t *testing.T) {
		test := setup()
		test.unblock()
		synctest.Wait()
		if err := test.block(); errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("%v: %v, want not deadline exceeded", test.what, err)
		}
	})
	synctestSubtest(t, "unblock before setdeadline", func(t *testing.T) {
		test := setup()
		test.unblock()
		synctest.Wait()
		test.setDeadline(5 * time.Second)
		if err := test.block(); errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("%v: %v, want not deadline exceeded", test.what, err)
		}
	})
	synctestSubtest(t, "unblock after blocking", func(t *testing.T) {
		test := setup()
		test.setDeadline(5 * time.Second)
		var done bool
		go func() {
			if err := test.block(); errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("%v: %v, want not deadline exceeded", test.what, err)
			}
			done = true
		}()
		synctest.Wait()
		if done {
			t.Fatalf("%v: unexpectedly returned before unblocking", test.what)
		}
		test.unblock()
		synctest.Wait()
		if !done {
			t.Fatalf("%v: did not return after unblocking", test.what)
		}
	})
	synctestSubtest(t, "deadline expires", func(t *testing.T) {
		test := setup()
		start := time.Now()
		const delay = 5 * time.Second
		test.setDeadline(delay)
		var done atomic.Bool
		go func() {
			if err := test.block(); !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("%v: %v, want os.ErrDeadlineExceeded", test.what, err)
			}
			if got, want := time.Since(start), delay; got != want {
				t.Errorf("%v: returned after %v, want %v", test.what, got, want)
			}
			done.Store(true)
		}()
		synctest.Wait()
		if done.Load() {
			t.Fatalf("%v: unexpectedly returned before unblocking", test.what)
		}
		time.Sleep(delay)
		synctest.Wait()
		if !done.Load() {
			t.Fatalf("%v: did not return after deadline", test.what)
		}
	})
	synctestSubtest(t, "deadline already expired", func(t *testing.T) {
		test := setup()
		test.setDeadline(-1 * time.Second)
		test.unblock()
		synctest.Wait()
		if err := test.block(); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("%v: %v, want os.ErrDeadlineExceeded", test.what, err)
		}
	})
	synctestSubtest(t, "reduce deadline after blocking", func(t *testing.T) {
		test := setup()
		test.setDeadline(5 * time.Second)
		var done bool
		go func() {
			if err := test.block(); !errors.Is(err, os.ErrDeadlineExceeded) {
				t.Errorf("%v: %v, want os.ErrDeadlineExceeded", test.what, err)
			}
			done = true
		}()
		synctest.Wait()
		if done {
			t.Fatalf("%v: unexpectedly returned before reducing deadline", test.what)
		}
		test.setDeadline(-1 * time.Second)
		synctest.Wait()
		if !done {
			t.Fatalf("%v: did not return after deadline", test.what)
		}
	})
}
