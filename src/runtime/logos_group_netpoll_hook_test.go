// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1 || windows

package runtime_test

import (
	"context"
	"errors"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"logos/rt"
	"runtime"
)

func TestLogosGroupNetpollCancelWinsRegisterParkRace(t *testing.T) {
	_, server := logosConnPair(t)
	registered := make(chan struct{})
	release := make(chan struct{})
	var once atomic.Bool
	runtime.SetLogosWaitAfterRegisterHookForTest(func() {
		if once.CompareAndSwap(false, true) {
			close(registered)
		}
		<-release
	})
	defer runtime.SetLogosWaitAfterRegisterHookForTest(nil)
	g, gctx := rt.NewGroup(t.Context(), 0)
	readDone := make(chan bool, 1)
	g.Go(func(ctx context.Context) error {
		buf := make([]byte, 1)
		_, err := server.Read(buf)
		readDone <- os.IsTimeout(err)
		return nil
	})
	g.Go(func(ctx context.Context) error {
		select {
		case <-registered:
			return errLogosSibling
		case <-time.After(5 * time.Second):
			return errors.New("waiter never reached park seam")
		}
	})
	select {
	case <-registered:
	case <-time.After(5 * time.Second):
		t.Fatal("waiter never reached park seam")
	}
	select {
	case <-gctx.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("group context never cancelled")
	}
	time.Sleep(100 * time.Millisecond)
	close(release)
	if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want sibling failure", err)
	}
	select {
	case timedOut := <-readDone:
		if !timedOut {
			t.Fatal("parked Read did not return a timeout after cancel won the race")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("parked Read still blocked after cancel won the race")
	}
}

func TestLogosGroupNetpollCancelRaceStress(t *testing.T) {
	for range 50 {
		_, server := logosConnPair(t)
		g, _ := rt.NewGroup(t.Context(), 0)
		var timedOut atomic.Bool
		g.Go(func(ctx context.Context) error {
			buf := make([]byte, 1)
			_, err := server.Read(buf)
			timedOut.Store(os.IsTimeout(err))
			return nil
		})
		g.Go(func(ctx context.Context) error {
			time.Sleep(time.Millisecond)
			return errLogosSibling
		})
		if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
			t.Fatalf("Wait = %v, want sibling failure", err)
		}
		if !timedOut.Load() {
			t.Fatal("blocked Read did not time out after cancel")
		}
	}
}
