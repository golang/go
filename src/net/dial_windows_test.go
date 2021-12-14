// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"errors"
	"net/internal/socktest"
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
	oldTestHookCanceledDial := testHookCanceledDial
	defer func() {
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
	sw.Set(socktest.FilterConnect, func(*socktest.Status) (socktest.AfterFilter, error) {
		return func(*socktest.Status) error {
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
			return context.Canceled
		}, nil
	})
	defer sw.Set(socktest.FilterConnect, nil)

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
