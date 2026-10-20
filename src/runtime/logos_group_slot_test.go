// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"context"
	"errors"
	"maps"
	"runtime"
	"slices"
	"testing"
	"time"

	"logos/rt"
)

// The group tag of a goroutine is a slot index the runtime lends to one
// logos/rt.Group at a time (SPEC Sec 2.6, T-206). These tests pin the
// allocator (lowest free slot, reuse only of released slots) and the
// isolation it buys: a cancel for a slot never reaches a goroutine of a
// different live group, and a group's slot returns only after its join.

// Slots reuse only what was released, lowest first, and a take after
// that finds a slot nobody holds.
func TestLogosGroupSlotsReuseOnlyReleased(t *testing.T) {
	const n = 200
	slots := runtime.LogosSlotsAcquire(n)
	held := make(map[uint32]bool, n)
	for _, slot := range slots {
		if slot == 0 {
			t.Fatal("slot 0 lent: it is the tag of a goroutine outside a group")
		}
		if held[slot] {
			t.Fatalf("slot %d lent twice while held", slot)
		}
		held[slot] = true
	}

	var freed []uint32
	for i, slot := range slots {
		if i%3 == 0 {
			freed = append(freed, slot)
			delete(held, slot)
		}
	}
	runtime.LogosSlotsRelease(freed)
	again := runtime.LogosSlotsAcquire(len(freed))
	for _, slot := range again {
		held[slot] = true
	}
	want := slices.Sorted(slices.Values(freed))
	if !slices.Equal(again, want) {
		t.Fatalf("reacquired slots %v, want exactly the released slots, lowest first: %v", again, want)
	}

	fresh := runtime.LogosSlotsAcquire(1)
	if held[fresh[0]] {
		t.Fatalf("slot %d lent while held", fresh[0])
	}
	runtime.LogosSlotsRelease(fresh)
	runtime.LogosSlotsRelease(slices.Collect(maps.Keys(held)))
}

// Cancelling the reused slot of a released group leaves the waiter of a
// different live group alone.
func TestLogosGroupReusedSlotCancelSparesLiveGroup(t *testing.T) {
	live, reused, cancelled := runtime.LogosReusedSlotCancel()
	if live == reused {
		t.Fatalf("slot %d lent to two live groups", live)
	}
	if cancelled {
		t.Fatal("cancelling a reused slot set the cancel bit of another live group's waiter")
	}
}

// logosAwaitWaiter blocks until the registry holds more waiters than
// before: the group child under test is parked in a netpoll wait. It
// fails the test when none registers within bound.
func logosAwaitWaiter(t *testing.T, before int, bound time.Duration) {
	t.Helper()
	deadline := time.Now().Add(bound)
	for runtime.LogosWaiterCount() <= before {
		if time.Now().After(deadline) {
			t.Fatalf("no group child registered as a netpoll waiter within %v", bound)
		}
		time.Sleep(time.Millisecond)
	}
}

// A group whose Wait is still joining keeps its slot: groups created
// meanwhile get other slots, so their failures never wake the joining
// group's blocked child.
func TestLogosGroupNetpollSlotHeldUntilJoin(t *testing.T) {
	joinClient, joinServer := logosConnPair(t)
	registered := runtime.LogosWaiterCount()
	joining, _ := rt.NewGroup(t.Context(), 0)
	joiningRead := make(chan error, 1)
	joining.Go(func(ctx context.Context) error {
		buf := make([]byte, 1)
		_, err := joinServer.Read(buf)
		joiningRead <- err
		return nil
	})
	joinDone := make(chan error, 1)
	go func() { joinDone <- joining.Wait() }()
	logosAwaitWaiter(t, registered, 5*time.Second)
	time.Sleep(50 * time.Millisecond)

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
