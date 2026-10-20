// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"context"
	"errors"
	"net"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"logos/rt"
)

// M8-S4 attacks in Go mode (SPEC Sec 2.6, T-206). A child of a logos
// group blocked in conn.Read with no data wakes within 100 ms after a
// sibling fails, with a timeout error, and the descriptor stays usable.
// A goroutine outside the group blocked on the same kind of read stays
// blocked: this proof needs no change to the unsupervised_goroutine lint.

var errLogosSibling = errors.New("logos sibling failed")

// logosConnPair returns a connected client and server side on loopback.
func logosConnPair(t *testing.T) (client, server net.Conn) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	client, err = net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	server, err = ln.Accept()
	if err != nil {
		client.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		client.Close()
		server.Close()
	})
	return client, server
}

// logosGroupWait joins g with a bound: a child that does not wake fails
// the test instead of hanging it.
func logosGroupWait(t *testing.T, g *rt.Group, bound time.Duration) error {
	t.Helper()
	res := make(chan error, 1)
	go func() { res <- g.Wait() }()
	select {
	case err := <-res:
		return err
	case <-time.After(bound):
		t.Fatalf("Wait still blocked after %v: a blocked child did not wake", bound)
		return nil
	}
}

func TestLogosGroupNetpollWakesChild(t *testing.T) {
	client, server := logosConnPair(t)
	g, _ := rt.NewGroup(t.Context(), 0)
	var failedAt, wokeAt atomic.Int64
	var timedOut atomic.Bool
	g.Go(func(ctx context.Context) error {
		buf := make([]byte, 1)
		_, err := server.Read(buf)
		wokeAt.Store(time.Now().UnixNano())
		timedOut.Store(os.IsTimeout(err))
		return nil
	})
	g.Go(func(ctx context.Context) error {
		time.Sleep(50 * time.Millisecond)
		failedAt.Store(time.Now().UnixNano())
		return errLogosSibling
	})
	if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
	}
	if !timedOut.Load() {
		t.Fatal("blocked Read did not return a timeout error")
	}
	if d := time.Duration(wokeAt.Load() - failedAt.Load()); d > 100*time.Millisecond {
		t.Fatalf("child woke %v after the sibling failed, want <= 100ms", d)
	}

	// The descriptor stays usable: a later read gets data normally.
	if _, err := client.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 1)
	if n, err := server.Read(buf); err != nil || n != 1 || buf[0] != 'x' {
		t.Fatalf("read after cancel = %d, %v, %q; want 1, nil, x", n, err, buf[:n])
	}
}

func TestLogosGroupNetpollOutsiderStaysBlocked(t *testing.T) {
	client, server := logosConnPair(t)
	outClient, outServer := logosConnPair(t)
	_ = client

	outDone := make(chan error, 1)
	go func() {
		buf := make([]byte, 1)
		_, err := outServer.Read(buf)
		outDone <- err
	}()

	g, _ := rt.NewGroup(t.Context(), 0)
	g.Go(func(ctx context.Context) error {
		buf := make([]byte, 1)
		_, err := server.Read(buf)
		return err
	})
	g.Go(func(ctx context.Context) error {
		time.Sleep(50 * time.Millisecond)
		return errLogosSibling
	})
	if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
	}

	select {
	case err := <-outDone:
		t.Fatalf("outsider Read returned %v after the group failed; want still blocked", err)
	case <-time.After(200 * time.Millisecond):
	}
	if _, err := outClient.Write([]byte("y")); err != nil {
		t.Fatal(err)
	}
	if err := <-outDone; err != nil {
		t.Fatalf("outsider Read = %v after data arrived, want nil", err)
	}
}

// A child of a different group blocked on the same kind of read stays
// blocked when another group fails: the cancel visits only the waiters
// of the failing group.
func TestLogosGroupNetpollOtherGroupStaysBlocked(t *testing.T) {
	_, server := logosConnPair(t)
	otherClient, otherServer := logosConnPair(t)

	other, _ := rt.NewGroup(t.Context(), 0)
	otherDone := make(chan error, 1)
	other.Go(func(ctx context.Context) error {
		buf := make([]byte, 1)
		_, err := otherServer.Read(buf)
		otherDone <- err
		return nil
	})

	g, _ := rt.NewGroup(t.Context(), 0)
	g.Go(func(ctx context.Context) error {
		buf := make([]byte, 1)
		_, err := server.Read(buf)
		return err
	})
	g.Go(func(ctx context.Context) error {
		time.Sleep(50 * time.Millisecond)
		return errLogosSibling
	})
	if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
	}

	select {
	case err := <-otherDone:
		t.Fatalf("other group's Read returned %v after a different group failed; want still blocked", err)
	case <-time.After(200 * time.Millisecond):
	}
	if _, err := otherClient.Write([]byte("z")); err != nil {
		t.Fatal(err)
	}
	if err := logosGroupWait(t, other, 5*time.Second); err != nil {
		t.Fatalf("other group Wait = %v, want nil", err)
	}
	if err := <-otherDone; err != nil {
		t.Fatalf("other group's Read = %v after data arrived, want nil", err)
	}
}

// A waiter's late release must not erase a cancellation that a new
// waiter received on the reused descriptor: the purge on reuse drops
// the stale entry, and a release without an entry clears nothing.
func TestLogosGroupNetpollStaleReleaseKeepsCancel(t *testing.T) {
	if !runtime.LogosStaleReleaseKeepsCancel() {
		t.Fatal("a stale waiter's release erased the cancel bit of a reused descriptor")
	}
}

// A descriptor reused while no group waiter is registered anywhere still
// starts without a stale cancel bit, and the published waiter count
// returns to zero once the registry empties: the idle fast path of the
// purge must neither skip a stale bit nor lag behind the registry.
func TestLogosGroupNetpollIdlePurgeAndCount(t *testing.T) {
	if !runtime.LogosPurgeClearsBit() {
		t.Fatal("a stale cancel bit survived the purge of a reused descriptor")
	}
	if !runtime.LogosStaleReleaseKeepsCancel() {
		t.Fatal("a stale waiter's release erased the cancel bit of a reused descriptor")
	}
	if !runtime.LogosRegistryIdle() {
		t.Fatal("the registry count stayed positive after every waiter left")
	}
}

// Cancel visits only the registered waiters of its own group.
func TestLogosGroupNetpollCancelFiltersByGroup(t *testing.T) {
	otherBit, ownBit := runtime.LogosCancelFiltersByGroup()
	if otherBit {
		t.Fatal("cancelling a different group set the cancel bit")
	}
	if !ownBit {
		t.Fatal("cancelling the waiter's own group did not set the cancel bit")
	}
}
