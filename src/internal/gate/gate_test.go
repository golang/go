// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gate_test

import (
	"context"
	"internal/gate"
	"testing"
	"time"
)

func TestGateLockAndUnlock(t *testing.T) {
	g := gate.New(false)
	if set := g.Lock(); set {
		t.Errorf("g.Lock of never-locked gate: true, want false")
	}
	unlockedc := make(chan struct{})
	donec := make(chan struct{})
	go func() {
		defer close(donec)
		if set := g.Lock(); !set {
			t.Errorf("g.Lock of set gate: false, want true")
		}
		select {
		case <-unlockedc:
		default:
			t.Errorf("g.Lock succeeded while gate was held")
		}
		g.Unlock(false)
	}()
	time.Sleep(1 * time.Millisecond)
	close(unlockedc)
	g.Unlock(true)
	<-donec
	if set := g.Lock(); set {
		t.Errorf("g.Lock of unset gate: true, want false")
	}
}

func TestGateWaitAndLock(t *testing.T) {
	g := gate.New(false)
	// WaitAndLock is canceled.
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()
	if err := g.WaitAndLock(ctx); err != context.DeadlineExceeded {
		t.Fatalf("g.WaitAndLock = %v, want context.DeadlineExceeded", err)
	}
	// WaitAndLock succeeds.
	set := false
	go func() {
		time.Sleep(1 * time.Millisecond)
		g.Lock()
		set = true
		g.Unlock(true)
	}()
	if err := g.WaitAndLock(context.Background()); err != nil {
		t.Fatalf("g.WaitAndLock = %v, want nil", err)
	}
	if !set {
		t.Fatalf("g.WaitAndLock returned before gate was set")
	}
	g.Unlock(true)
	// WaitAndLock succeeds when the gate is set and the context is canceled.
	if err := g.WaitAndLock(ctx); err != nil {
		t.Fatalf("g.WaitAndLock = %v, want nil", err)
	}
}

func TestGateLockIfSet(t *testing.T) {
	g := gate.New(false)
	if locked := g.LockIfSet(); locked {
		t.Fatalf("g.LockIfSet of unset gate = %v, want false", locked)
	}
	g.Lock()
	if locked := g.LockIfSet(); locked {
		t.Fatalf("g.LockIfSet of locked gate = %v, want false", locked)
	}
	g.Unlock(true)
	if locked := g.LockIfSet(); !locked {
		t.Fatalf("g.LockIfSet of set gate = %v, want true", locked)
	}
}
