// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package net

import (
	"context"
	"errors"
	"syscall"
	"testing"
	"time"
)

func init() {
	isEADDRINUSE = func(err error) bool {
		return errors.Is(err, syscall.EADDRINUSE)
	}
}

// Issue 16523
func TestDialContextCancelRace(t *testing.T) {
	oldConnectFunc := connectFunc
	oldGetsockoptIntFunc := getsockoptIntFunc
	oldTestHookCanceledDial := testHookCanceledDial
	defer func() {
		connectFunc = oldConnectFunc
		getsockoptIntFunc = oldGetsockoptIntFunc
		testHookCanceledDial = oldTestHookCanceledDial
	}()

	ln := newLocalListener(t, "tcp")
	listenerDone := make(chan struct{})
	go func() {
		defer close(listenerDone)
		c, err := ln.Accept()
		if err == nil {
			c.Close()
		}
	}()
	defer func() { <-listenerDone }()
	defer ln.Close()

	sawCancel := make(chan bool, 1)
	testHookCanceledDial = func() {
		sawCancel <- true
	}

	ctx, cancelCtx := context.WithCancel(context.Background())

	connectFunc = func(fd int, addr syscall.Sockaddr) error {
		err := oldConnectFunc(fd, addr)
		t.Logf("connect(%d, addr) = %v", fd, err)
		if err == nil {
			// On some operating systems, localhost
			// connects _sometimes_ succeed immediately.
			// Prevent that, so we exercise the code path
			// we're interested in testing. This seems
			// harmless. It makes FreeBSD 10.10 work when
			// run with many iterations. It failed about
			// half the time previously.
			return syscall.EINPROGRESS
		}
		return err
	}

	getsockoptIntFunc = func(fd, level, opt int) (val int, err error) {
		val, err = oldGetsockoptIntFunc(fd, level, opt)
		t.Logf("getsockoptIntFunc(%d, %d, %d) = (%v, %v)", fd, level, opt, val, err)
		if level == syscall.SOL_SOCKET && opt == syscall.SO_ERROR && err == nil && val == 0 {
			t.Logf("canceling context")

			// Cancel the context at just the moment which
			// caused the race in issue 16523.
			cancelCtx()

			// And wait for the "interrupter" goroutine to
			// cancel the dial by messing with its write
			// timeout before returning.
			select {
			case <-sawCancel:
				t.Logf("saw cancel")
			case <-time.After(5 * time.Second):
				t.Errorf("didn't see cancel after 5 seconds")
			}
		}
		return
	}

	var d Dialer
	c, err := d.DialContext(ctx, "tcp", ln.Addr().String())
	if err == nil {
		c.Close()
		t.Fatal("unexpected successful dial; want context canceled error")
	}

	select {
	case <-ctx.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("expected context to be canceled")
	}

	oe, ok := err.(*OpError)
	if !ok || oe.Op != "dial" {
		t.Fatalf("Dial error = %#v; want dial *OpError", err)
	}

	if oe.Err != errCanceled {
		t.Errorf("DialContext = (%v, %v); want OpError with error %v", c, err, errCanceled)
	}
}
