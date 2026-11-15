// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !logosejected

package rt

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestPR3GroupParentCancelWakesNetpoll(t *testing.T) {
	cli, srv := pr3DialPair(t)
	defer cli.Close()
	defer srv.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	g, _ := NewGroup(ctx, 0)
	started := make(chan struct{})
	g.Go(func(context.Context) error {
		close(started)
		_, err := srv.Read(make([]byte, 1))
		return err
	})
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		cli.Close()
		srv.Close()
		t.Fatalf("child did not reach Read")
	}
	cancel()
	done := make(chan error, 1)
	go func() { done <- g.Wait() }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		cli.Close()
		srv.Close()
		t.Fatalf("Wait did not return within 2s after parent cancel")
	}
}

func TestPR3GroupParentDeadlineWakesNetpoll(t *testing.T) {
	cli, srv := pr3DialPair(t)
	defer cli.Close()
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	g, _ := NewGroup(ctx, 0)
	started := make(chan struct{})
	g.Go(func(context.Context) error {
		close(started)
		_, err := srv.Read(make([]byte, 1))
		return err
	})
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		cli.Close()
		srv.Close()
		t.Fatalf("child did not reach Read")
	}
	done := make(chan error, 1)
	go func() { done <- g.Wait() }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		cli.Close()
		srv.Close()
		t.Fatalf("Wait did not return within 2s after parent deadline")
	}
}

func TestPR3GroupSlotReuseNotSpurious(t *testing.T) {
	cli1, srv1 := pr3DialPair(t)
	ctx1, cancel1 := context.WithCancel(context.Background())
	defer cancel1()
	g1, _ := NewGroup(ctx1, 0)
	started1 := make(chan struct{})
	g1.Go(func(context.Context) error {
		close(started1)
		_, err := srv1.Read(make([]byte, 1))
		return err
	})
	select {
	case <-started1:
	case <-time.After(2 * time.Second):
		cli1.Close()
		srv1.Close()
		t.Fatalf("first child did not reach Read")
	}
	cancel1()
	done1 := make(chan error, 1)
	go func() { done1 <- g1.Wait() }()
	select {
	case <-done1:
	case <-time.After(2 * time.Second):
		cli1.Close()
		srv1.Close()
		t.Fatalf("first Wait did not return within 2s after parent cancel")
	}
	cli1.Close()
	srv1.Close()

	cli2, srv2 := pr3DialPair(t)
	defer cli2.Close()
	defer srv2.Close()
	g2, _ := NewGroup(context.Background(), 0)
	started2 := make(chan struct{})
	gotN := make(chan int, 1)
	gotByte := make(chan byte, 1)
	g2.Go(func(context.Context) error {
		close(started2)
		buf := make([]byte, 1)
		n, err := srv2.Read(buf)
		if err == nil {
			gotByte <- buf[0]
			gotN <- n
		} else {
			gotN <- -1
		}
		return err
	})
	select {
	case <-started2:
	case <-time.After(2 * time.Second):
		cli2.Close()
		srv2.Close()
		t.Fatalf("second child did not reach Read")
	}
	if _, err := cli2.Write([]byte{0x7A}); err != nil {
		cli2.Close()
		srv2.Close()
		t.Fatalf("controlled write failed: %v", err)
	}
	done2 := make(chan error, 1)
	go func() { done2 <- g2.Wait() }()
	select {
	case <-done2:
	case <-time.After(2 * time.Second):
		cli2.Close()
		srv2.Close()
		t.Fatalf("second Wait did not return within 2s")
	}
	select {
	case n := <-gotN:
		if n != 1 {
			t.Fatalf("reused slot waiter woke spuriously: n=%d, want 1", n)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("reader result missing")
	}
	select {
	case b := <-gotByte:
		if b != 0x7A {
			t.Fatalf("reused slot byte = %#x, want 0x7a", b)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("reader byte missing")
	}
}

func TestPR3GroupFailureLeavesOutsiderNetpollBlocked(t *testing.T) {
	outCli, outSrv := pr3DialPair(t)
	defer outCli.Close()
	defer outSrv.Close()
	cli, srv := pr3DialPair(t)
	defer cli.Close()
	defer srv.Close()
	outsider := make(chan error, 1)
	reading := make(chan uint64, 1)
	go func() {
		reading <- goroutineID()
		_, err := outSrv.Read(make([]byte, 1))
		outsider <- err
	}()
	id := <-reading
	deadline := time.Now().Add(2 * time.Second)
	for !strings.Contains(goroutineStack(id), "[IO wait]") {
		select {
		case err := <-outsider:
			t.Fatalf("outsider Read returned before group failure: %v", err)
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal("outsider did not block in netpoll")
		}
		runtime.Gosched()
	}
	errChild := errors.New("child failed")
	g, _ := NewGroup(context.Background(), 0)
	started := make(chan struct{})
	g.Go(func(context.Context) error {
		close(started)
		_, err := srv.Read(make([]byte, 1))
		return err
	})
	<-started
	g.Go(func(context.Context) error { return errChild })
	if err := g.Wait(); !errors.Is(err, errChild) {
		t.Fatalf("Wait = %v, want %v", err, errChild)
	}
	select {
	case err := <-outsider:
		t.Fatalf("outsider Read returned %v after group failure", err)
	default:
	}
	if _, err := outCli.Write([]byte{1}); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-outsider:
		if err != nil {
			t.Fatalf("outsider Read = %v, want the written byte", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("outsider Read did not complete after the write")
	}
}
