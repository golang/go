// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"runtime"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

var (
	errGroupFirst  = errors.New("group first")
	errGroupSecond = errors.New("group second")
)

// M8-S1 attack: the first failure in completion order wins. childA fails
// at once; childB sleeps a millisecond (deterministic under synctest)
// then fails. Wait must report the first error.
func TestGroupFirstErrorWins(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		g.Go(func(ctx context.Context) error { return errGroupFirst })
		g.Go(func(ctx context.Context) error {
			synctest.Wait()
			return errGroupSecond
		})
		if err := g.Wait(); !errors.Is(err, errGroupFirst) {
			t.Fatalf("Wait = %v, want %v", err, errGroupFirst)
		}
	})
}

// M8-S1 attack: cancellation reaches a sibling blocked on the child
// context. childB waits on Done; childA fails. Wait must report childA's
// error and childB must have observed Done.
func TestGroupCancelReachesSibling(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		sawDone := make(chan struct{})
		g.Go(func(ctx context.Context) error { return errGroupFirst })
		g.Go(func(ctx context.Context) error {
			<-ctx.Done()
			close(sawDone)
			return ctx.Err()
		})
		if err := g.Wait(); !errors.Is(err, errGroupFirst) {
			t.Fatalf("Wait = %v, want %v", err, errGroupFirst)
		}
		select {
		case <-sawDone:
		default:
			t.Fatal("sibling did not observe cancellation")
		}
	})
}

// M18-S4 attack: group cancellation flows through the group context
// alone, with no g.cancelChan field. The first child fails at once;
// the second blocks in the lowered channel form (a select on the child
// context) and must observe the cancellation. Wait reports the first
// error, never the sibling's context error.
func TestGroupCancelThroughGroupContext(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		ch := make(chan int)
		cancelled := make(chan struct{})
		g.Go(func(ctx context.Context) error { return errGroupFirst })
		g.Go(func(ctx context.Context) error {
			select {
			case <-ch:
				return nil
			case <-ctx.Done():
				close(cancelled)
				return ctx.Err()
			}
		})
		if err := g.Wait(); !errors.Is(err, errGroupFirst) {
			t.Fatalf("Wait = %v, want %v", err, errGroupFirst)
		}
		select {
		case <-cancelled:
		default:
			t.Fatal("sibling blocked on channel did not observe group cancellation")
		}
	})
}

// M8-S1 attack: limit = 2 never runs 3 children at once. Five children
// record the peak concurrency; the peak must stay at or below 2.
func TestGroupLimitBoundsConcurrency(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 2)
		var cur, peak atomic.Int32
		for range 5 {
			g.Go(func(ctx context.Context) error {
				n := cur.Add(1)
				for {
					p := peak.Load()
					if n <= p || peak.CompareAndSwap(p, n) {
						break
					}
				}
				// Sleep holds both token holders inside at once: the
				// first holder is still asleep when the second starts,
				// so the peak reaches 2 deterministically under the
				// synctest scheduler.
				time.Sleep(time.Millisecond)
				cur.Add(-1)
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			t.Fatalf("Wait = %v, want nil", err)
		}
		if p := peak.Load(); p > 2 {
			t.Fatalf("peak concurrency = %d, want <= 2", p)
		}
		if p := peak.Load(); p < 2 {
			t.Fatalf("peak concurrency = %d, want 2 (limit not saturated)", p)
		}
	})
}

// M8-S1 attack: a child panic becomes *PanicError carrying the value and
// a stack, and the panic cancels the group.
func TestGroupPanicBecomesPanicError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		g.Go(func(ctx context.Context) error { panic("boom") })
		g.Go(func(ctx context.Context) error {
			<-ctx.Done()
			return ctx.Err()
		})
		err := g.Wait()
		pe, ok := errors.AsType[*PanicError](err)
		if !ok {
			t.Fatalf("Wait = %v (%T), want *PanicError", err, err)
		}
		if pe.Value != "boom" {
			t.Fatalf("PanicError.Value = %v, want boom", pe.Value)
		}
		if len(pe.Stack) == 0 {
			t.Fatal("PanicError.Stack is empty")
		}
	})
}

// A Go after Wait panics instead of starting a child: the group's slot
// is back with the runtime, and a late child would carry it into the
// next group that takes it. The panic leaves the join counter balanced,
// so no child ever runs.
func TestGroupGoAfterWaitPanics(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		if err := g.Wait(); err != nil {
			t.Fatalf("Wait = %v, want nil", err)
		}
		var ran atomic.Bool
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("Go after Wait did not panic")
				}
			}()
			g.Go(func(ctx context.Context) error {
				ran.Store(true)
				return nil
			})
		}()
		synctest.Wait()
		if ran.Load() {
			t.Fatal("child ran after Wait")
		}
	})
}

// M8-S1 attack: Wait returns only after all children exit. Two children
// block on release; Wait must stay pending until release closes.
func TestGroupWaitJoinsChildren(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		release := make(chan struct{})
		for range 2 {
			g.Go(func(ctx context.Context) error {
				select {
				case <-release:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			})
		}
		done := make(chan error, 1)
		go func() { done <- g.Wait() }()
		synctest.Wait()
		select {
		case err := <-done:
			t.Fatalf("Wait returned %v before children exited", err)
		default:
		}
		close(release)
		synctest.Wait()
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("Wait = %v, want nil", err)
			}
		default:
			t.Fatal("Wait did not return after children exited")
		}
	})
}

// M8-S1 attack: the child context is cancelled at join, even on success.
func TestGroupContextCancelledAtJoin(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, gctx := NewGroup(t.Context(), 0)
		g.Go(func(ctx context.Context) error { return nil })
		if err := g.Wait(); err != nil {
			t.Fatalf("Wait = %v, want nil", err)
		}
		select {
		case <-gctx.Done():
		default:
			t.Fatal("child context not cancelled at join")
		}
	})
}

// holdSlot records the current and peak concurrency around a one
// millisecond hold, which is deterministic under synctest.
func holdSlot(cur, peak *atomic.Int32) {
	n := cur.Add(1)
	for {
		p := peak.Load()
		if n <= p || peak.CompareAndSwap(p, n) {
			break
		}
	}
	time.Sleep(time.Millisecond)
	cur.Add(-1)
}

// M8-S1 attack: limit = 1 serializes the children. The bound must hold
// at its smallest value, where a `limit > 1` slip leaves the group
// unbounded.
func TestGroupLimitOneSerializes(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 1)
		var cur, peak atomic.Int32
		for range 3 {
			g.Go(func(ctx context.Context) error {
				holdSlot(&cur, &peak)
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			t.Fatalf("Wait = %v, want nil", err)
		}
		if p := peak.Load(); p != 1 {
			t.Fatalf("peak concurrency = %d, want 1", p)
		}
	})
}

// M8-S1 attack: a limit of zero leaves the group unbounded, so all
// children run at once.
func TestGroupLimitZeroIsUnbounded(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		var cur, peak atomic.Int32
		for range 3 {
			g.Go(func(ctx context.Context) error {
				holdSlot(&cur, &peak)
				return nil
			})
		}
		if err := g.Wait(); err != nil {
			t.Fatalf("Wait = %v, want nil", err)
		}
		if p := peak.Load(); p != 3 {
			t.Fatalf("peak concurrency = %d, want 3", p)
		}
	})
}

// M8-S1 attack: the PanicError text names the group child and carries
// the panic value.
func TestGroupPanicErrorMessage(t *testing.T) {
	pe := &PanicError{Value: "boom"}
	if got, want := pe.Error(), "logos: group child panicked: boom"; got != want {
		t.Fatalf("Error() = %q, want %q", got, want)
	}
}

// M8-S1 attack: a child that exits through runtime.Goexit still
// releases the join counter, so Wait returns instead of hanging.
func TestGroupGoexitChildJoins(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 1)
		g.Go(func(ctx context.Context) error {
			runtime.Goexit()
			return nil
		})
		g.Go(func(ctx context.Context) error { return nil })
		if err := g.Wait(); err != nil {
			t.Fatalf("Wait = %v, want nil", err)
		}
	})
}

type (
	panicNamedString string
	panicNamedInt    int
	panicNamedBool   bool
	panicNamedFloat  float32
	panicValueErr    struct{ msg string }
	panicPtrErr      struct{ msg string }
	panicStringer    struct{}
	panicBoth        struct{}
	panicIntStringer int
	panicPlainStruct struct{ A, B int }
)

func (e panicValueErr) Error() string   { return e.msg }
func (e *panicPtrErr) Error() string    { return e.msg }
func (panicStringer) String() string    { return "stringer-text" }
func (panicBoth) Error() string         { return "as-error" }
func (panicBoth) String() string        { return "as-string" }
func (panicIntStringer) String() string { return "int-stringer" }

// M8-S1 attack: panicValueText spells every kind a panic usually carries
// exactly as fmt's %v does, so PanicError.Error needs no fmt import
// (go/build deps keep logos/rt to context, errors, reflect, unsafe).
func TestPanicValueTextMatchesFmt(t *testing.T) {
	values := []any{
		nil,
		"boom", "", panicNamedString("named"),
		errors.New("plain error"), panicValueErr{"value error"}, &panicPtrErr{"pointer error"},
		panicStringer{}, panicBoth{}, panicIntStringer(3),
		true, false, panicNamedBool(true),
		int(-5), int8(-128), int16(300), int32(-1 << 31), int64(1 << 62), panicNamedInt(42),
		uint(7), uint8(255), uint16(65535), uint32(1<<32 - 1), uint64(1<<64 - 1), uintptr(0xdeadbeef),
		float32(0.1), float32(1e21), float32(16777216), float32(3.4028235e38), float32(1e-7),
		float64(math.Pi), 1e21, 1e-7, 100000.0, 1000000.0, 123456789.0,
		math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1),
		math.SmallestNonzeroFloat64, math.MaxFloat64, panicNamedFloat(2.5),
	}
	for _, v := range values {
		if got, want := panicValueText(v), fmt.Sprint(v); got != want {
			t.Errorf("panicValueText(%T %#v) = %q, want %q (fmt %%v)", v, v, got, want)
		}
	}
}

// M8-S1 attack: a value with both Error and String methods spells as its
// Error text, the order fmt applies, and a value of an integer kind with
// a String method spells as that text, not as the integer.
func TestPanicValueTextMethodsBeforeKinds(t *testing.T) {
	if got := panicValueText(panicBoth{}); got != "as-error" {
		t.Fatalf("error and Stringer value = %q, want as-error", got)
	}
	if got := panicValueText(panicIntStringer(3)); got != "int-stringer" {
		t.Fatalf("Stringer of integer kind = %q, want int-stringer", got)
	}
}

// M8-S1 attack: every other kind spells as `(T) 0xADDR`, with the real
// dynamic type name, and a pointer spells with its own address.
func TestPanicValueTextOpaqueKinds(t *testing.T) {
	n := 7
	opaque := regexp.MustCompile(`^\((.+)\) 0x[0-9a-f]+$`)
	values := []any{
		panicPlainStruct{1, 2}, struct{}{}, []int{1}, map[string]int{"a": 1},
		func() {}, make(chan int), [2]int{1, 2}, complex(1, 2), complex64(1),
	}
	for _, v := range values {
		got := panicValueText(v)
		m := opaque.FindStringSubmatch(got)
		if m == nil {
			t.Errorf("panicValueText(%T) = %q, want `(T) 0xADDR`", v, got)
			continue
		}
		if want := fmt.Sprintf("%T", v); m[1] != want {
			t.Errorf("panicValueText(%T) names type %q, want %q", v, m[1], want)
		}
	}
	if got, want := panicValueText(&n), fmt.Sprintf("(*int) %p", &n); got != want {
		t.Errorf("panicValueText(*int) = %q, want %q", got, want)
	}
}

// M8-S1 attack: PanicError.Error joins the fixed prefix and the spelled
// value, for a nil value too.
func TestPanicErrorMessageSpellsValue(t *testing.T) {
	cases := []struct {
		value any
		want  string
	}{
		{7, "logos: group child panicked: 7"},
		{"boom", "logos: group child panicked: boom"},
		{errors.New("inner"), "logos: group child panicked: inner"},
		{nil, "logos: group child panicked: <nil>"},
		{2.5, "logos: group child panicked: 2.5"},
	}
	for _, c := range cases {
		if got := (&PanicError{Value: c.value}).Error(); got != c.want {
			t.Errorf("PanicError{%#v}.Error() = %q, want %q", c.value, got, c.want)
		}
	}
}

func deepPanic(n int) {
	if n == 0 {
		panic("deep")
	}
	deepPanic(n - 1)
}

// M8-S1 attack: the panic stack is complete even when it exceeds the
// first 1024-byte buffer: it grows until the whole trace fits, so the
// trace still ends with the frame that created the goroutine.
func TestGroupPanicStackGrows(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		g.Go(func(ctx context.Context) error {
			deepPanic(40)
			return nil
		})
		pe, ok := errors.AsType[*PanicError](g.Wait())
		if !ok {
			t.Fatal("Wait did not return *PanicError")
		}
		if len(pe.Stack) <= 1024 {
			t.Fatalf("stack is %d bytes, want a deep trace over 1024", len(pe.Stack))
		}
		for _, frame := range []string{"logos/rt.deepPanic", "created by logos/rt.(*Group).Go"} {
			if !bytes.Contains(pe.Stack, []byte(frame)) {
				t.Errorf("stack lacks %q:\n%s", frame, pe.Stack)
			}
		}
	})
}
