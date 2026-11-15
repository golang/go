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
	"reflect"
	"regexp"
	"runtime"
	"strings"
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

// M8-S1 attack: the first failure in completion order wins. childA is
// spawned first but blocks on the child context; childB is spawned
// second and fails at once. Wait must report childB error even though
// childA has the lower spawn index.
func TestGroupFirstCompletionWins(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		g.Go(func(ctx context.Context) error {
			<-ctx.Done()
			return errGroupFirst
		})
		g.Go(func(ctx context.Context) error { return errGroupSecond })
		if err := g.Wait(); !errors.Is(err, errGroupSecond) {
			t.Fatalf("Wait = %v, want %v", err, errGroupSecond)
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

func TestGroupLimitBoundsConcurrency(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		if got := inFlightWhenQuiet(t, 2, 5); got != 2 {
			t.Fatalf("children in flight = %d, want 2 (limit saturated, not exceeded)", got)
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

func TestGroupOutsideGoroutineStaysBlocked(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		g.Go(func(ctx context.Context) error { return errGroupFirst })
		siblingWoke := make(chan struct{})
		g.Go(func(ctx context.Context) error {
			<-ctx.Done()
			close(siblingWoke)
			return ctx.Err()
		})
		outsiderDone := make(chan struct{})
		release := make(chan struct{})
		exited := make(chan struct{})
		go func() {
			defer close(exited)
			select {
			case <-t.Context().Done():
				close(outsiderDone)
			case <-release:
			}
		}()
		if err := g.Wait(); !errors.Is(err, errGroupFirst) {
			t.Fatalf("Wait = %v, want %v", err, errGroupFirst)
		}
		synctest.Wait()
		select {
		case <-siblingWoke:
		default:
			t.Fatal("sibling did not observe cancellation")
		}
		select {
		case <-outsiderDone:
			t.Fatal("outside goroutine woke on group failure")
		default:
		}
		close(release)
		<-exited
	})
}

func inFlightWhenQuiet(t *testing.T, limit, children int) int32 {
	t.Helper()
	g, _ := NewGroup(t.Context(), limit)
	var inFlight, ran atomic.Int32
	release := make(chan struct{})
	spawned := make(chan struct{})
	go func() {
		defer close(spawned)
		for range children {
			g.Go(func(ctx context.Context) error {
				inFlight.Add(1)
				<-release
				ran.Add(1)
				return nil
			})
		}
	}()
	synctest.Wait()
	got := inFlight.Load()
	close(release)
	<-spawned
	if err := g.Wait(); err != nil {
		t.Fatalf("Wait = %v, want nil", err)
	}
	if n := ran.Load(); n != int32(children) {
		t.Fatalf("children that ran = %d, want %d", n, children)
	}
	return got
}

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

func TestGroupLimitOneSerializes(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		if got := inFlightWhenQuiet(t, 1, 3); got != 1 {
			t.Fatalf("children in flight = %d, want 1", got)
		}
	})
}

func TestGroupGoNestedUnderLimitOneJoins(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 1)
		var order []string
		g.Go(func(ctx context.Context) error {
			g.GoNested(func(ctx context.Context) error {
				order = append(order, "inner")
				return nil
			})
			order = append(order, "outer")
			return nil
		})
		if err := g.Wait(); err != nil {
			t.Fatalf("Wait = %v, want nil", err)
		}
		if got := strings.Join(order, ","); got != "outer,inner" {
			t.Fatalf("order = %s, want outer,inner", got)
		}
	})
}

func TestGroupGoNestedKeepsTheLimit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 1)
		var cur, peak atomic.Int32
		g.Go(func(ctx context.Context) error {
			for range 3 {
				g.GoNested(func(ctx context.Context) error {
					holdSlot(&cur, &peak)
					return nil
				})
			}
			holdSlot(&cur, &peak)
			return nil
		})
		if err := g.Wait(); err != nil {
			t.Fatalf("Wait = %v, want nil", err)
		}
		if p := peak.Load(); p != 1 {
			t.Fatalf("peak concurrency = %d, want 1", p)
		}
	})
}

func TestGroupGoNestedErrorFailsTheGroup(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 1)
		g.Go(func(ctx context.Context) error {
			g.GoNested(func(ctx context.Context) error { return errGroupFirst })
			return nil
		})
		if err := g.Wait(); !errors.Is(err, errGroupFirst) {
			t.Fatalf("Wait = %v, want %v", err, errGroupFirst)
		}
	})
}

// M8-S1 attack: a limit of zero leaves the group unbounded, so all
// children run at once.
func TestGroupLimitZeroIsUnbounded(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		if got := inFlightWhenQuiet(t, 0, 3); got != 3 {
			t.Fatalf("children in flight = %d, want 3", got)
		}
	})
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

func TestPanicErrorMatchesFmt(t *testing.T) {
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
		got := (&PanicError{Value: v}).Error()
		want := fmt.Sprint(v)
		if !strings.Contains(got, want) {
			t.Errorf("PanicError(%T %#v).Error() = %q, lacks %q", v, v, got, want)
		}
	}
}

func TestPanicErrorMethodsBeforeKinds(t *testing.T) {
	if got := (&PanicError{Value: panicBoth{}}).Error(); !strings.Contains(got, "as-error") {
		t.Fatalf("Error and Stringer value = %q, lacks as-error", got)
	}
	if got := (&PanicError{Value: panicIntStringer(3)}).Error(); !strings.Contains(got, "int-stringer") {
		t.Fatalf("Stringer of integer kind = %q, lacks int-stringer", got)
	}
}
func TestPanicErrorOpaqueKinds(t *testing.T) {
	n := 7
	opaque := regexp.MustCompile(`\((.+)\) 0x[0-9a-f]+`)
	values := []any{
		panicPlainStruct{1, 2}, struct{}{}, []int{1}, map[string]int{"a": 1},
		func() {}, make(chan int), [2]int{1, 2}, complex(1, 2), complex64(1),
	}
	for _, v := range values {
		pe := &PanicError{Value: v}
		if reflect.TypeOf(pe.Value) != reflect.TypeOf(v) {
			t.Errorf("PanicError(%T).Value type = %T, want %T", v, pe.Value, v)
		}
		if _, isFunc := v.(func()); isFunc {
			if reflect.ValueOf(pe.Value).Pointer() != reflect.ValueOf(v).Pointer() {
				t.Errorf("PanicError func value changed: %p vs %p", pe.Value, v)
			}
		} else if !reflect.DeepEqual(pe.Value, v) {
			t.Errorf("PanicError(%T).Value = %#v, want %#v", v, pe.Value, v)
		}
		got := pe.Error()
		m := opaque.FindStringSubmatch(got)
		if m == nil {
			t.Errorf("PanicError(%T).Error() = %q, lacks `(T) 0xADDR`", v, got)
			continue
		}
		if want := fmt.Sprintf("%T", v); m[1] != want {
			t.Errorf("PanicError(%T) names type %q, want %q", v, m[1], want)
		}
	}
	if got := (&PanicError{Value: &n}).Error(); !strings.Contains(got, fmt.Sprintf("(*int) %p", &n)) {
		t.Errorf("PanicError(*int).Error() = %q, lacks pointer spelling", got)
	}
}
func TestPanicErrorCarriesValue(t *testing.T) {
	values := []any{7, "boom", errors.New("inner"), nil, 2.5}
	for _, v := range values {
		var err error = &PanicError{Value: v}
		pe, ok := errors.AsType[*PanicError](err)
		if !ok {
			t.Fatalf("AsType(%T) ok=%v, want *PanicError", err, ok)
		}
		if !reflect.DeepEqual(pe.Value, v) {
			t.Errorf("PanicError(%#v).Value = %#v, want %#v", v, pe.Value, v)
		}
		if want := fmt.Sprint(v); !strings.Contains(pe.Error(), want) {
			t.Errorf("PanicError(%#v).Error() = %q, lacks %q", v, pe.Error(), want)
		}
	}
}

func TestPanicErrorValueRecoverable(t *testing.T) {
	inner := errors.New("inner")
	values := []any{7, "boom", inner, nil, 2.5}
	for _, v := range values {
		var err error = &PanicError{Value: v}
		pe, ok := errors.AsType[*PanicError](err)
		if !ok {
			t.Fatalf("AsType(%T) ok=%v, want *PanicError", err, ok)
		}
		if !reflect.DeepEqual(pe.Value, v) {
			t.Errorf("PanicError value = %#v, want %#v", pe.Value, v)
		}
		if pe.Error() == fmt.Sprint(v) {
			t.Errorf("PanicError(%#v).Error() = %q, names no origin beyond the value", v, pe.Error())
		}
	}
	synctest.Test(t, func(t *testing.T) {
		g, _ := NewGroup(t.Context(), 0)
		g.Go(func(ctx context.Context) error { panic(inner) })
		pe, ok := errors.AsType[*PanicError](g.Wait())
		if !ok {
			t.Fatal("Wait did not return *PanicError")
		}
		if !reflect.DeepEqual(pe.Value, any(inner)) {
			t.Errorf("recovered value = %#v, want %#v", pe.Value, any(inner))
		}
	})
}

func deepPanic(n int) {
	if n == 0 {
		panic("deep")
	}
	deepPanic(n - 1)
}

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
		if n := bytes.Count(pe.Stack, []byte("logos/rt.deepPanic")); n < 41 {
			t.Errorf("stack holds %d deepPanic frames, want at least 41:\n%s", n, pe.Stack)
		}
	})
}
