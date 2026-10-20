// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime_test

import (
	"context"
	"errors"
	"os"
	"syscall"
	"testing"
	"time"

	"logos/rt"
)

// M8-S4 documented limit (SPEC Sec 2.6): only netpoll-backed descriptors
// wake. An os.File over a blocking descriptor is not registered with the
// poller (the same class as regular files and cgo calls), so a child
// blocked in its Read stays blocked after a sibling fails, and the group
// joins once the read completes.
func TestLogosGroupNetpollSkipsBlockingDescriptor(t *testing.T) {
	var fds [2]int
	if err := syscall.Pipe(fds[:]); err != nil {
		t.Fatal(err)
	}
	reader := os.NewFile(uintptr(fds[0]), "blocking-read")
	writer := os.NewFile(uintptr(fds[1]), "blocking-write")
	t.Cleanup(func() {
		reader.Close()
		writer.Close()
	})

	readDone := make(chan error, 1)
	g, _ := rt.NewGroup(t.Context(), 0)
	g.Go(func(ctx context.Context) error {
		buf := make([]byte, 1)
		_, err := reader.Read(buf)
		readDone <- err
		return nil
	})
	g.Go(func(ctx context.Context) error {
		time.Sleep(50 * time.Millisecond)
		return errLogosSibling
	})
	waitDone := make(chan error, 1)
	go func() { waitDone <- g.Wait() }()

	select {
	case err := <-readDone:
		t.Fatalf("blocking Read returned %v after the group failed; want still blocked", err)
	case err := <-waitDone:
		t.Fatalf("Wait returned %v while a child was still blocked in a non-netpoll read", err)
	case <-time.After(300 * time.Millisecond):
	}
	if _, err := writer.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-waitDone:
		if !errors.Is(err, errLogosSibling) {
			t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Wait did not return after the blocking read completed")
	}
	if err := <-readDone; err != nil {
		t.Fatalf("blocking Read = %v after data arrived, want nil", err)
	}
}
