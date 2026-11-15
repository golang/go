// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1 || windows

package runtime_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"logos/rt"
)

func TestLogosGroupNetpollSlotHeldUntilJoin(t *testing.T) {
	joinClient, joinServer := logosConnPair(t)
	joining, _ := rt.NewGroup(t.Context(), 0)
	started := make(chan struct{})
	joiningRead := make(chan error, 1)
	joining.Go(func(ctx context.Context) error {
		close(started)
		buf := make([]byte, 1)
		_, err := joinServer.Read(buf)
		joiningRead <- err
		return nil
	})
	joinDone := make(chan error, 1)
	go func() { joinDone <- joining.Wait() }()
	<-started
	time.Sleep(100 * time.Millisecond)

	const failing = 8
	var groups [failing]*rt.Group
	for i := range groups {
		groups[i], _ = rt.NewGroup(t.Context(), 0)
		groups[i].Go(func(ctx context.Context) error { return errLogosSibling })
	}
	for _, g := range groups {
		if err := logosGroupWait(t, g, 5*time.Second); !errors.Is(err, errLogosSibling) {
			t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
		}
	}

	select {
	case err := <-joiningRead:
		t.Fatalf("joining group's Read returned %v after another group failed; want still blocked", err)
	case <-time.After(200 * time.Millisecond):
	}
	if _, err := joinClient.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	if err := <-joinDone; err != nil {
		t.Fatalf("joining group Wait = %v, want nil", err)
	}
	if err := <-joiningRead; err != nil {
		t.Fatalf("joining group's Read = %v after data arrived, want nil", err)
	}
}

func TestLogosGroupSlotReusedAfterJoinStartsClean(t *testing.T) {
	failed, _ := rt.NewGroup(t.Context(), 0)
	failed.Go(func(ctx context.Context) error { return errLogosSibling })
	if err := logosGroupWait(t, failed, 5*time.Second); !errors.Is(err, errLogosSibling) {
		t.Fatalf("Wait = %v, want %v", err, errLogosSibling)
	}

	client, server := logosConnPair(t)
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
		t.Fatalf("reused slot's Read returned %v without data; want still blocked", err)
	case <-time.After(200 * time.Millisecond):
	}
	if _, err := client.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-readDone:
		if err != nil {
			t.Fatalf("reused slot's Read = %v after data arrived, want nil", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("reused slot's Read still blocked after data arrived")
	}
	if err := logosGroupWait(t, next, 5*time.Second); err != nil {
		t.Fatalf("next group Wait = %v, want nil", err)
	}
}
