// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1 || windows

package runtime_test

import (
	"context"
	"errors"
	"net"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"logos/rt"
)

var errLogosSibling = errors.New("logos sibling failed")

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

func TestLogosGroupNetpollCancelBeforeRegister(t *testing.T) {
	_, server := logosConnPair(t)
	g, _ := rt.NewGroup(t.Context(), 0)
	var timedOut atomic.Bool
	g.Go(func(ctx context.Context) error {
		<-ctx.Done()
		time.Sleep(200 * time.Millisecond)
		buf := make([]byte, 1)
		_, err := server.Read(buf)
		timedOut.Store(os.IsTimeout(err))
		return nil
	})
	g.Go(func(ctx context.Context) error {
		return errLogosSibling
	})
	if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
	}
	if !timedOut.Load() {
		t.Fatal("Read after the sibling's cancel did not return a timeout error")
	}
}

func TestLogosGroupNetpollCancelRacingDataUnblocksNextRead(t *testing.T) {
	client, server := logosConnPair(t)
	g, _ := rt.NewGroup(t.Context(), 0)
	var firstN atomic.Int32
	var firstOK atomic.Bool
	var secondTimedOut atomic.Bool
	g.Go(func(ctx context.Context) error {
		<-ctx.Done()
		time.Sleep(200 * time.Millisecond)
		buf := make([]byte, 1)
		n, err := server.Read(buf)
		firstN.Store(int32(n))
		firstOK.Store(err == nil)
		_, err = server.Read(buf)
		secondTimedOut.Store(os.IsTimeout(err))
		return nil
	})
	g.Go(func(ctx context.Context) error {
		if _, err := client.Write([]byte("x")); err != nil {
			return err
		}
		return errLogosSibling
	})
	if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
	}
	if !firstOK.Load() || firstN.Load() != 1 {
		t.Fatalf("first Read = %d bytes, ok = %v; want 1 byte of arriving data", firstN.Load(), firstOK.Load())
	}
	if !secondTimedOut.Load() {
		t.Fatal("second Read after the cancel did not return a timeout error")
	}
}

func TestLogosGroupNetpollSecondReadAfterCancelStillTimesOut(t *testing.T) {
	_, server := logosConnPair(t)
	g, _ := rt.NewGroup(t.Context(), 0)
	var firstTimedOut, secondTimedOut atomic.Bool
	g.Go(func(ctx context.Context) error {
		<-ctx.Done()
		time.Sleep(200 * time.Millisecond)
		buf := make([]byte, 1)
		_, err := server.Read(buf)
		firstTimedOut.Store(os.IsTimeout(err))
		_, err = server.Read(buf)
		secondTimedOut.Store(os.IsTimeout(err))
		return nil
	})
	g.Go(func(ctx context.Context) error {
		return errLogosSibling
	})
	if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
	}
	if !firstTimedOut.Load() {
		t.Fatal("first Read after the sibling's cancel did not return a timeout error")
	}
	if !secondTimedOut.Load() {
		t.Fatal("second Read after the cancel did not return a timeout error; the first waiter's release must not clear it")
	}
}

func TestLogosGroupNetpollReusedDescriptorStartsClean(t *testing.T) {
	client, server := logosConnPair(t)
	g, _ := rt.NewGroup(t.Context(), 0)
	var timedOut atomic.Bool
	g.Go(func(ctx context.Context) error {
		buf := make([]byte, 1)
		_, err := server.Read(buf)
		timedOut.Store(os.IsTimeout(err))
		return nil
	})
	g.Go(func(ctx context.Context) error {
		time.Sleep(50 * time.Millisecond)
		return errLogosSibling
	})
	if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
	}
	if !timedOut.Load() {
		t.Fatal("blocked Read did not return a timeout error")
	}
	client.Close()
	server.Close()

	client, server = logosConnPair(t)
	next, _ := rt.NewGroup(t.Context(), 0)
	started := make(chan struct{})
	readDone := make(chan error, 1)
	next.Go(func(ctx context.Context) error {
		close(started)
		buf := make([]byte, 1)
		_, err := server.Read(buf)
		readDone <- err
		return nil
	})
	<-started
	time.Sleep(100 * time.Millisecond)
	select {
	case err := <-readDone:
		t.Fatalf("new group's Read returned %v without data; want still blocked", err)
	case <-time.After(200 * time.Millisecond):
	}
	if _, err := client.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-readDone:
		if err != nil {
			t.Fatalf("new group's Read = %v after data arrived, want nil", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("new group's Read still blocked after data arrived")
	}
	if err := logosGroupWait(t, next, 5*time.Second); err != nil {
		t.Fatalf("next group Wait = %v, want nil", err)
	}
}
