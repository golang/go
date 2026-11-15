// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build logosejected

package rt

import (
	"context"
	"testing"
	"time"
)

const nativeWakeWindow = 300 * time.Millisecond

func TestEjectedGroupCancelLeavesNetworkReadToComplete(t *testing.T) {
	cli, srv := pr3DialPair(t)
	defer cli.Close()
	defer srv.Close()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	g, _ := NewGroup(ctx, 0)
	started := make(chan struct{})
	received := make(chan byte, 1)
	g.Go(func(context.Context) error {
		close(started)
		buf := make([]byte, 1)
		if _, err := srv.Read(buf); err != nil {
			return err
		}
		received <- buf[0]
		return nil
	})
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatalf("child did not reach Read")
	}
	cancel()
	waited := make(chan error, 1)
	go func() { waited <- g.Wait() }()
	select {
	case <-waited:
		t.Fatalf("Wait returned while the child was still blocked in Read")
	case <-time.After(nativeWakeWindow):
	}
	if _, err := cli.Write([]byte{0x7A}); err != nil {
		t.Fatalf("write to the blocked reader failed: %v", err)
	}
	select {
	case <-waited:
	case <-time.After(2 * time.Second):
		t.Fatalf("Wait did not return within 2s after the read completed")
	}
	select {
	case b := <-received:
		if b != 0x7A {
			t.Fatalf("read byte = %#x, want 0x7a", b)
		}
	default:
		t.Fatalf("cancellation interrupted the read; want it to complete with the written byte")
	}
}
