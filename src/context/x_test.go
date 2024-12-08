// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context_test

import (
	. "context"
	"errors"
	"fmt"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// Each XTestFoo in context_test.go must be called from a TestFoo here to run.
func TestParentFinishesChild(t *testing.T) {
	XTestParentFinishesChild(t) // uses unexported context types
}
func TestChildFinishesFirst(t *testing.T) {
	XTestChildFinishesFirst(t) // uses unexported context types
}
func TestCancelRemoves(t *testing.T) {
	XTestCancelRemoves(t) // uses unexported context types
}
func TestCustomContextGoroutines(t *testing.T) {
	XTestCustomContextGoroutines(t) // reads the context.goroutines counter
}

// The following are regular tests in package context_test.

// otherContext is a Context that's not one of the types defined in context.go.
// This lets us test code paths that differ based on the underlying type of the
// Context.
type otherContext struct {
	Context
}

const (
	shortDuration    = 1 * time.Millisecond // a reasonable duration to block in a test
	veryLongDuration = 1000 * time.Hour     // an arbitrary upper bound on the test's running time
)

// quiescent returns an arbitrary duration by which the program should have
// completed any remaining work and reached a steady (idle) state.
func quiescent(t *testing.T) time.Duration {
	deadline, ok := t.Deadline()
	if !ok {
		return 5 * time.Second
	}

	const arbitraryCleanupMargin = 1 * time.Second
	return time.Until(deadline) - arbitraryCleanupMargin
}
func TestBackground(t *testing.T) {
	c := Background()
	if c == nil {
		t.Fatalf("Background returned nil")
	}
	select {
	case x := <-c.Done():
		t.Errorf("<-c.Done() == %v want nothing (it should block)", x)
	default:
	}
	if got, want := fmt.Sprint(c), "context.Background"; got != want {
		t.Errorf("Background().String() = %q want %q", got, want)
	}
}

func TestTODO(t *testing.T) {
	c := TODO()
	if c == nil {
		t.Fatalf("TODO returned nil")
	}
	select {
	case x := <-c.Done():
		t.Errorf("<-c.Done() == %v want nothing (it should block)", x)
	default:
	}
	if got, want := fmt.Sprint(c), "context.TODO"; got != want {
		t.Errorf("TODO().String() = %q want %q", got, want)
	}
}

func TestWithCancel(t *testing.T) {
	c1, cancel := WithCancel(Background())

	if got, want := fmt.Sprint(c1), "context.Background.WithCancel"; got != want {
		t.Errorf("c1.String() = %q want %q", got, want)
	}

	o := otherContext{c1}
	c2, _ := WithCancel(o)
	contexts := []Context{c1, o, c2}

	for i, c := range contexts {
		if d := c.Done(); d == nil {
			t.Errorf("c[%d].Done() == %v want non-nil", i, d)
		}
		if e := c.Err(); e != nil {
			t.Errorf("c[%d].Err() == %v want nil", i, e)
		}

		select {
		case x := <-c.Done():
			t.Errorf("<-c.Done() == %v want nothing (it should block)", x)
		default:
		}
	}

	cancel() // Should propagate synchronously.
	for i, c := range contexts {
		select {
		case <-c.Done():
		default:
			t.Errorf("<-c[%d].Done() blocked, but shouldn't have", i)
		}
		if e := c.Err(); e != Canceled {
			t.Errorf("c[%d].Err() == %v want %v", i, e, Canceled)
		}
	}
}

func testDeadline(c Context, name string, t *testing.T) {
	t.Helper()
	d := quiescent(t)
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		t.Fatalf("%s: context not timed out after %v", name, d)
	case <-c.Done():
	}
	if e := c.Err(); e != DeadlineExceeded {
		t.Errorf("%s: c.Err() == %v; want %v", name, e, DeadlineExceeded)
	}
}

func TestDeadline(t *testing.T) {
	t.Parallel()

	c, _ := WithDeadline(Background(), time.Now().Add(shortDuration))
	if got, prefix := fmt.Sprint(c), "context.Background.WithDeadline("; !strings.HasPrefix(got, prefix) {
		t.Errorf("c.String() = %q want prefix %q", got, prefix)
	}
	testDeadline(c, "WithDeadline", t)

	c, _ = WithDeadline(Background(), time.Now().Add(shortDuration))
	o := otherContext{c}
	testDeadline(o, "WithDeadline+otherContext", t)

	c, _ = WithDeadline(Background(), time.Now().Add(shortDuration))
	o = otherContext{c}
	c, _ = WithDeadline(o, time.Now().Add(veryLongDuration))
	testDeadline(c, "WithDeadline+otherContext+WithDeadline", t)

	c, _ = WithDeadline(Background(), time.Now().Add(-shortDuration))
	testDeadline(c, "WithDeadline+inthepast", t)

	c, _ = WithDeadline(Background(), time.Now())
	testDeadline(c, "WithDeadline+now", t)
}

func TestTimeout(t *testing.T) {
	t.Parallel()

	c, _ := WithTimeout(Background(), shortDuration)
	if got, prefix := fmt.Sprint(c), "context.Background.WithDeadline("; !strings.HasPrefix(got, prefix) {
		t.Errorf("c.String() = %q want prefix %q", got, prefix)
	}
	testDeadline(c, "WithTimeout", t)

	c, _ = WithTimeout(Background(), shortDuration)
	o := otherContext{c}
	testDeadline(o, "WithTimeout+otherContext", t)

	c, _ = WithTimeout(Background(), shortDuration)
	o = otherContext{c}
	c, _ = WithTimeout(o, veryLongDuration)
	testDeadline(c, "WithTimeout+otherContext+WithTimeout", t)
}

func TestCanceledTimeout(t *testing.T) {
	c, _ := WithTimeout(Background(), time.Second)
	o := otherContext{c}
	c, cancel := WithTimeout(o, veryLongDuration)
	cancel() // Should propagate synchronously.
	select {
	case <-c.Done():
	default:
		t.Errorf("<-c.Done() blocked, but shouldn't have")
	}
	if e := c.Err(); e != Canceled {
		t.Errorf("c.Err() == %v want %v", e, Canceled)
	}
}

type key1 int
type key2 int

func (k key2) String() string { return fmt.Sprintf("%[1]T(%[1]d)", k) }

var k1 = key1(1)
var k2 = key2(1) // same int as k1, different type
var k3 = key2(3) // same type as k2, different int

func TestValues(t *testing.T) {
	check := func(c Context, nm, v1, v2, v3 string) {
		if v, ok := c.Value(k1).(string); ok == (len(v1) == 0) || v != v1 {
			t.Errorf(`%s.Value(k1).(string) = %q, %t want %q, %t`, nm, v, ok, v1, len(v1) != 0)
		}
		if v, ok := c.Value(k2).(string); ok == (len(v2) == 0) || v != v2 {
			t.Errorf(`%s.Value(k2).(string) = %q, %t want %q, %t`, nm, v, ok, v2, len(v2) != 0)
		}
		if v, ok := c.Value(k3).(string); ok == (len(v3) == 0) || v != v3 {
			t.Errorf(`%s.Value(k3).(string) = %q, %t want %q, %t`, nm, v, ok, v3, len(v3) != 0)
		}
	}

	c0 := Background()
	check(c0, "c0", "", "", "")

	c1 := WithValue(Background(), k1, "c1k1")
	check(c1, "c1", "c1k1", "", "")

	if got, want := fmt.Sprint(c1), `context.Background.WithValue(context_test.key1, c1k1)`; got != want {
		t.Errorf("c.String() = %q want %q", got, want)
	}

	c2 := WithValue(c1, k2, "c2k2")
	check(c2, "c2", "c1k1", "c2k2", "")

	if got, want := fmt.Sprint(c2), `context.Background.WithValue(context_test.key1, c1k1).WithValue(context_test.key2(1), c2k2)`; got != want {
		t.Errorf("c.String() = %q want %q", got, want)
	}

	c3 := WithValue(c2, k3, "c3k3")
	check(c3, "c2", "c1k1", "c2k2", "c3k3")

	c4 := WithValue(c3, k1, nil)
	check(c4, "c4", "", "c2k2", "c3k3")

	if got, want := fmt.Sprint(c4), `context.Background.WithValue(context_test.key1, c1k1).WithValue(context_test.key2(1), c2k2).WithValue(context_test.key2(3), c3k3).WithValue(context_test.key1, <nil>)`; got != want {
		t.Errorf("c.String() = %q want %q", got, want)
	}

	o0 := otherContext{Background()}
	check(o0, "o0", "", "", "")

	o1 := otherContext{WithValue(Background(), k1, "c1k1")}
	check(o1, "o1", "c1k1", "", "")

	o2 := WithValue(o1, k2, "o2k2")
	check(o2, "o2", "c1k1", "o2k2", "")

	o3 := otherContext{c4}
	check(o3, "o3", "", "c2k2", "c3k3")

	o4 := WithValue(o3, k3, nil)
	check(o4, "o4", "", "c2k2", "")
}

func TestAllocs(t *testing.T) {
	bg := Background()
	for _, test := range []struct {
		desc       string
		f          func()
		limit      float64
		gccgoLimit float64
	}{
		{
			desc:       "Background()",
			f:          func() { Background() },
			limit:      0,
			gccgoLimit: 0,
		},
		{
			desc: fmt.Sprintf("WithValue(bg, %v, nil)", k1),
			f: func() {
				c := WithValue(bg, k1, nil)
				c.Value(k1)
			},
			limit:      3,
			gccgoLimit: 3,
		},
		{
			desc: "WithTimeout(bg, 1*time.Nanosecond)",
			f: func() {
				c, _ := WithTimeout(bg, 1*time.Nanosecond)
				<-c.Done()
			},
			limit:      12,
			gccgoLimit: 15,
		},
		{
			desc: "WithCancel(bg)",
			f: func() {
				c, cancel := WithCancel(bg)
				cancel()
				<-c.Done()
			},
			limit:      5,
			gccgoLimit: 8,
		},
		{
			desc: "WithTimeout(bg, 5*time.Millisecond)",
			f: func() {
				c, cancel := WithTimeout(bg, 5*time.Millisecond)
				cancel()
				<-c.Done()
			},
			limit:      8,
			gccgoLimit: 25,
		},
	} {
		limit := test.limit
		if runtime.Compiler == "gccgo" {
			// gccgo does not yet do escape analysis.
			// TODO(iant): Remove this when gccgo does do escape analysis.
			limit = test.gccgoLimit
		}
		numRuns := 100
		if testing.Short() {
			numRuns = 10
		}
		if n := testing.AllocsPerRun(numRuns, test.f); n > limit {
			t.Errorf("%s allocs = %f want %d", test.desc, n, int(limit))
		}
	}
}

func TestSimultaneousCancels(t *testing.T) {
	root, cancel := WithCancel(Background())
	m := map[Context]CancelFunc{root: cancel}
	q := []Context{root}
	// Create a tree of contexts.
	for len(q) != 0 && len(m) < 100 {
		parent := q[0]
		q = q[1:]
		for i := 0; i < 4; i++ {
			ctx, cancel := WithCancel(parent)
			m[ctx] = cancel
			q = append(q, ctx)
		}
	}
	// Start all the cancels in a random order.
	var wg sync.WaitGroup
	wg.Add(len(m))
	for _, cancel := range m {
		go func(cancel CancelFunc) {
			cancel()
			wg.Done()
		}(cancel)
	}

	d := quiescent(t)
	stuck := make(chan struct{})
	timer := time.AfterFunc(d, func() { close(stuck) })
	defer timer.Stop()

	// Wait on all the contexts in a random order.
	for ctx := range m {
		select {
		case <-ctx.Done():
		case <-stuck:
			buf := make([]byte, 10<<10)
			n := runtime.Stack(buf, true)
			t.Fatalf("timed out after %v waiting for <-ctx.Done(); stacks:\n%s", d, buf[:n])
		}
	}
	// Wait for all the cancel functions to return.
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-stuck:
		buf := make([]byte, 10<<10)
		n := runtime.Stack(buf, true)
		t.Fatalf("timed out after %v waiting for cancel functions; stacks:\n%s", d, buf[:n])
	}
}

func TestInterlockedCancels(t *testing.T) {
	parent, cancelParent := WithCancel(Background())
	child, cancelChild := WithCancel(parent)
	go func() {
		<-parent.Done()
		cancelChild()
	}()
	cancelParent()
	d := quiescent(t)
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-child.Done():
	case <-timer.C:
		buf := make([]byte, 10<<10)
		n := runtime.Stack(buf, true)
		t.Fatalf("timed out after %v waiting for child.Done(); stacks:\n%s", d, buf[:n])
	}
}

func TestLayersCancel(t *testing.T) {
	testLayers(t, time.Now().UnixNano(), false)
}

func TestLayersTimeout(t *testing.T) {
	testLayers(t, time.Now().UnixNano(), true)
}

func testLayers(t *testing.T, seed int64, testTimeout bool) {
	t.Parallel()

	r := rand.New(rand.NewSource(seed))
	prefix := fmt.Sprintf("seed=%d", seed)
	errorf := func(format string, a ...any) {
		t.Errorf(prefix+format, a...)
	}
	const (
		minLayers = 30
	)
	type value int
	var (
		vals      []*value
		cancels   []CancelFunc
		numTimers int
		ctx       = Background()
	)
	for i := 0; i < minLayers || numTimers == 0 || len(cancels) == 0 || len(vals) == 0; i++ {
		switch r.Intn(3) {
		case 0:
			v := new(value)
			ctx = WithValue(ctx, v, v)
			vals = append(vals, v)
		case 1:
			var cancel CancelFunc
			ctx, cancel = WithCancel(ctx)
			cancels = append(cancels, cancel)
		case 2:
			var cancel CancelFunc
			d := veryLongDuration
			if testTimeout {
				d = shortDuration
			}
			ctx, cancel = WithTimeout(ctx, d)
			cancels = append(cancels, cancel)
			numTimers++
		}
	}
	checkValues := func(when string) {
		for _, key := range vals {
			if val := ctx.Value(key).(*value); key != val {
				errorf("%s: ctx.Value(%p) = %p want %p", when, key, val, key)
			}
		}
	}
	if !testTimeout {
		select {
		case <-ctx.Done():
			errorf("ctx should not be canceled yet")
		default:
		}
	}
	if s, prefix := fmt.Sprint(ctx), "context.Background."; !strings.HasPrefix(s, prefix) {
		t.Errorf("ctx.String() = %q want prefix %q", s, prefix)
	}
	t.Log(ctx)
	checkValues("before cancel")
	if testTimeout {
		d := quiescent(t)
		timer := time.NewTimer(d)
		defer timer.Stop()
		select {
		case <-ctx.Done():
		case <-timer.C:
			errorf("ctx should have timed out after %v", d)
		}
		checkValues("after timeout")
	} else {
		cancel := cancels[r.Intn(len(cancels))]
		cancel()
		select {
		case <-ctx.Done():
		default:
			errorf("ctx should be canceled")
		}
		checkValues("after cancel")
	}
}

func TestWithCancelCanceledParent(t *testing.T) {
	parent, pcancel := WithCancelCause(Background())
	cause := fmt.Errorf("Because!")
	pcancel(cause)

	c, _ := WithCancel(parent)
	select {
	case <-c.Done():
	default:
		t.Errorf("child not done immediately upon construction")
	}
	if got, want := c.Err(), Canceled; got != want {
		t.Errorf("child not canceled; got = %v, want = %v", got, want)
	}
	if got, want := Cause(c), cause; got != want {
		t.Errorf("child has wrong cause; got = %v, want = %v", got, want)
	}
}

func TestWithCancelSimultaneouslyCanceledParent(t *testing.T) {
	// Cancel the parent goroutine concurrently with creating a child.
	for i := 0; i < 100; i++ {
		parent, pcancel := WithCancelCause(Background())
		cause := fmt.Errorf("Because!")
		go pcancel(cause)

		c, _ := WithCancel(parent)
		<-c.Done()
		if got, want := c.Err(), Canceled; got != want {
			t.Errorf("child not canceled; got = %v, want = %v", got, want)
		}
		if got, want := Cause(c), cause; got != want {
			t.Errorf("child has wrong cause; got = %v, want = %v", got, want)
		}
	}
}

func TestWithValueChecksKey(t *testing.T) {
	panicVal := recoveredValue(func() { _ = WithValue(Background(), []byte("foo"), "bar") })
	if panicVal == nil {
		t.Error("expected panic")
	}
	panicVal = recoveredValue(func() { _ = WithValue(Background(), nil, "bar") })
	if got, want := fmt.Sprint(panicVal), "nil key"; got != want {
		t.Errorf("panic = %q; want %q", got, want)
	}
}

func TestInvalidDerivedFail(t *testing.T) {
	panicVal := recoveredValue(func() { _, _ = WithCancel(nil) })
	if panicVal == nil {
		t.Error("expected panic")
	}
	panicVal = recoveredValue(func() { _, _ = WithDeadline(nil, time.Now().Add(shortDuration)) })
	if panicVal == nil {
		t.Error("expected panic")
	}
	panicVal = recoveredValue(func() { _ = WithValue(nil, "foo", "bar") })
	if panicVal == nil {
		t.Error("expected panic")
	}
}

func recoveredValue(fn func()) (v any) {
	defer func() { v = recover() }()
	fn()
	return
}

func TestDeadlineExceededSupportsTimeout(t *testing.T) {
	i, ok := DeadlineExceeded.(interface {
		Timeout() bool
	})
	if !ok {
		t.Fatal("DeadlineExceeded does not support Timeout interface")
	}
	if !i.Timeout() {
		t.Fatal("wrong value for timeout")
	}
}
func TestCause(t *testing.T) {
	var (
		forever       = 1e6 * time.Second
		parentCause   = fmt.Errorf("parentCause")
		childCause    = fmt.Errorf("childCause")
		tooSlow       = fmt.Errorf("tooSlow")
		finishedEarly = fmt.Errorf("finishedEarly")
	)
	for _, test := range []struct {
		name  string
		ctx   func() Context
		err   error
		cause error
	}{
		{
			name:  "Background",
			ctx:   Background,
			err:   nil,
			cause: nil,
		},
		{
			name:  "TODO",
			ctx:   TODO,
			err:   nil,
			cause: nil,
		},
		{
			name: "WithCancel",
			ctx: func() Context {
				ctx, cancel := WithCancel(Background())
				cancel()
				return ctx
			},
			err:   Canceled,
			cause: Canceled,
		},
		{
			name: "WithCancelCause",
			ctx: func() Context {
				ctx, cancel := WithCancelCause(Background())
				cancel(parentCause)
				return ctx
			},
			err:   Canceled,
			cause: parentCause,
		},
		{
			name: "WithCancelCause nil",
			ctx: func() Context {
				ctx, cancel := WithCancelCause(Background())
				cancel(nil)
				return ctx
			},
			err:   Canceled,
			cause: Canceled,
		},
		{
			name: "WithCancelCause: parent cause before child",
			ctx: func() Context {
				ctx, cancelParent := WithCancelCause(Background())
				ctx, cancelChild := WithCancelCause(ctx)
				cancelParent(parentCause)
				cancelChild(childCause)
				return ctx
			},
			err:   Canceled,
			cause: parentCause,
		},
		{
			name: "WithCancelCause: parent cause after child",
			ctx: func() Context {
				ctx, cancelParent := WithCancelCause(Background())
				ctx, cancelChild := WithCancelCause(ctx)
				cancelChild(childCause)
				cancelParent(parentCause)
				return ctx
			},
			err:   Canceled,
			cause: childCause,
		},
		{
			name: "WithCancelCause: parent cause before nil",
			ctx: func() Context {
				ctx, cancelParent := WithCancelCause(Background())
				ctx, cancelChild := WithCancel(ctx)
				cancelParent(parentCause)
				cancelChild()
				return ctx
			},
			err:   Canceled,
			cause: parentCause,
		},
		{
			name: "WithCancelCause: parent cause after nil",
			ctx: func() Context {
				ctx, cancelParent := WithCancelCause(Background())
				ctx, cancelChild := WithCancel(ctx)
				cancelChild()
				cancelParent(parentCause)
				return ctx
			},
			err:   Canceled,
			cause: Canceled,
		},
		{
			name: "WithCancelCause: child cause after nil",
			ctx: func() Context {
				ctx, cancelParent := WithCancel(Background())
				ctx, cancelChild := WithCancelCause(ctx)
				cancelParent()
				cancelChild(childCause)
				return ctx
			},
			err:   Canceled,
			cause: Canceled,
		},
		{
			name: "WithCancelCause: child cause before nil",
			ctx: func() Context {
				ctx, cancelParent := WithCancel(Background())
				ctx, cancelChild := WithCancelCause(ctx)
				cancelChild(childCause)
				cancelParent()
				return ctx
			},
			err:   Canceled,
			cause: childCause,
		},
		{
			name: "WithTimeout",
			ctx: func() Context {
				ctx, cancel := WithTimeout(Background(), 0)
				cancel()
				return ctx
			},
			err:   DeadlineExceeded,
			cause: DeadlineExceeded,
		},
		{
			name: "WithTimeout canceled",
			ctx: func() Context {
				ctx, cancel := WithTimeout(Background(), forever)
				cancel()
				return ctx
			},
			err:   Canceled,
			cause: Canceled,
		},
		{
			name: "WithTimeoutCause",
			ctx: func() Context {
				ctx, cancel := WithTimeoutCause(Background(), 0, tooSlow)
				cancel()
				return ctx
			},
			err:   DeadlineExceeded,
			cause: tooSlow,
		},
		{
			name: "WithTimeoutCause canceled",
			ctx: func() Context {
				ctx, cancel := WithTimeoutCause(Background(), forever, tooSlow)
				cancel()
				return ctx
			},
			err:   Canceled,
			cause: Canceled,
		},
		{
			name: "WithTimeoutCause stacked",
			ctx: func() Context {
				ctx, cancel := WithCancelCause(Background())
				ctx, _ = WithTimeoutCause(ctx, 0, tooSlow)
				cancel(finishedEarly)
				return ctx
			},
			err:   DeadlineExceeded,
			cause: tooSlow,
		},
		{
			name: "WithTimeoutCause stacked canceled",
			ctx: func() Context {
				ctx, cancel := WithCancelCause(Background())
				ctx, _ = WithTimeoutCause(ctx, forever, tooSlow)
				cancel(finishedEarly)
				return ctx
			},
			err:   Canceled,
			cause: finishedEarly,
		},
		{
			name: "WithoutCancel",
			ctx: func() Context {
				return WithoutCancel(Background())
			},
			err:   nil,
			cause: nil,
		},
		{
			name: "WithoutCancel canceled",
			ctx: func() Context {
				ctx, cancel := WithCancelCause(Background())
				ctx = WithoutCancel(ctx)
				cancel(finishedEarly)
				return ctx
			},
			err:   nil,
			cause: nil,
		},
		{
			name: "WithoutCancel timeout",
			ctx: func() Context {
				ctx, cancel := WithTimeoutCause(Background(), 0, tooSlow)
				ctx = WithoutCancel(ctx)
				cancel()
				return ctx
			},
			err:   nil,
			cause: nil,
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			ctx := test.ctx()
			if got, want := ctx.Err(), test.err; want != got {
				t.Errorf("ctx.Err() = %v want %v", got, want)
			}
			if got, want := Cause(ctx), test.cause; want != got {
				t.Errorf("Cause(ctx) = %v want %v", got, want)
			}
		})
	}
}

func TestCauseRace(t *testing.T) {
	cause := errors.New("TestCauseRace")
	ctx, cancel := WithCancelCause(Background())
	go func() {
		cancel(cause)
	}()
	for {
		// Poll Cause, rather than waiting for Done, to test that
		// access to the underlying cause is synchronized properly.
		if err := Cause(ctx); err != nil {
			if err != cause {
				t.Errorf("Cause returned %v, want %v", err, cause)
			}
			break
		}
		runtime.Gosched()
	}
}

func TestWithoutCancel(t *testing.T) {
	key, value := "key", "value"
	ctx := WithValue(Background(), key, value)
	ctx = WithoutCancel(ctx)
	if d, ok := ctx.Deadline(); !d.IsZero() || ok != false {
		t.Errorf("ctx.Deadline() = %v, %v want zero, false", d, ok)
	}
	if done := ctx.Done(); done != nil {
		t.Errorf("ctx.Deadline() = %v want nil", done)
	}
	if err := ctx.Err(); err != nil {
		t.Errorf("ctx.Err() = %v want nil", err)
	}
	if v := ctx.Value(key); v != value {
		t.Errorf("ctx.Value(%q) = %q want %q", key, v, value)
	}
}

type customDoneContext struct {
	Context
	donec chan struct{}
}

func (c *customDoneContext) Done() <-chan struct{} {
	return c.donec
}

func TestCustomContextPropagation(t *testing.T) {
	cause := errors.New("TestCustomContextPropagation")
	donec := make(chan struct{})
	ctx1, cancel1 := WithCancelCause(Background())
	ctx2 := &customDoneContext{
		Context: ctx1,
		donec:   donec,
	}
	ctx3, cancel3 := WithCancel(ctx2)
	defer cancel3()

	cancel1(cause)
	close(donec)

	<-ctx3.Done()
	if got, want := ctx3.Err(), Canceled; got != want {
		t.Errorf("child not canceled; got = %v, want = %v", got, want)
	}
	if got, want := Cause(ctx3), cause; got != want {
		t.Errorf("child has wrong cause; got = %v, want = %v", got, want)
	}
}

// customCauseContext is a custom Context used to test context.Cause.
type customCauseContext struct {
	mu   sync.Mutex
	done chan struct{}
	err  error

	cancelChild CancelFunc
}

func (ccc *customCauseContext) Deadline() (deadline time.Time, ok bool) {
	return
}

func (ccc *customCauseContext) Done() <-chan struct{} {
	ccc.mu.Lock()
	defer ccc.mu.Unlock()
	return ccc.done
}

func (ccc *customCauseContext) Err() error {
	ccc.mu.Lock()
	defer ccc.mu.Unlock()
	return ccc.err
}

func (ccc *customCauseContext) Value(key any) any {
	return nil
}

func (ccc *customCauseContext) cancel() {
	ccc.mu.Lock()
	ccc.err = Canceled
	close(ccc.done)
	cancelChild := ccc.cancelChild
	ccc.mu.Unlock()

	if cancelChild != nil {
		cancelChild()
	}
}

func (ccc *customCauseContext) setCancelChild(cancelChild CancelFunc) {
	ccc.cancelChild = cancelChild
}

func TestCustomContextCause(t *testing.T) {
	// Test if we cancel a custom context, Err and Cause return Canceled.
	ccc := &customCauseContext{
		done: make(chan struct{}),
	}
	ccc.cancel()
	if got := ccc.Err(); got != Canceled {
		t.Errorf("ccc.Err() = %v, want %v", got, Canceled)
	}
	if got := Cause(ccc); got != Canceled {
		t.Errorf("Cause(ccc) = %v, want %v", got, Canceled)
	}

	// Test that if we pass a custom context to WithCancelCause,
	// and then cancel that child context with a cause,
	// that the cause of the child canceled context is correct
	// but that the parent custom context is not canceled.
	ccc = &customCauseContext{
		done: make(chan struct{}),
	}
	ctx, causeFunc := WithCancelCause(ccc)
	cause := errors.New("TestCustomContextCause")
	causeFunc(cause)
	if got := ctx.Err(); got != Canceled {
		t.Errorf("after CancelCauseFunc ctx.Err() = %v, want %v", got, Canceled)
	}
	if got := Cause(ctx); got != cause {
		t.Errorf("after CancelCauseFunc Cause(ctx) = %v, want %v", got, cause)
	}
	if got := ccc.Err(); got != nil {
		t.Errorf("after CancelCauseFunc ccc.Err() = %v, want %v", got, nil)
	}
	if got := Cause(ccc); got != nil {
		t.Errorf("after CancelCauseFunc Cause(ccc) = %v, want %v", got, nil)
	}

	// Test that if we now cancel the parent custom context,
	// the cause of the child canceled context is still correct,
	// and the parent custom context is canceled without a cause.
	ccc.cancel()
	if got := ctx.Err(); got != Canceled {
		t.Errorf("after CancelCauseFunc ctx.Err() = %v, want %v", got, Canceled)
	}
	if got := Cause(ctx); got != cause {
		t.Errorf("after CancelCauseFunc Cause(ctx) = %v, want %v", got, cause)
	}
	if got := ccc.Err(); got != Canceled {
		t.Errorf("after CancelCauseFunc ccc.Err() = %v, want %v", got, Canceled)
	}
	if got := Cause(ccc); got != Canceled {
		t.Errorf("after CancelCauseFunc Cause(ccc) = %v, want %v", got, Canceled)
	}

	// Test that if we associate a custom context with a child,
	// then canceling the custom context cancels the child.
	ccc = &customCauseContext{
		done: make(chan struct{}),
	}
	ctx, cancelFunc := WithCancel(ccc)
	ccc.setCancelChild(cancelFunc)
	ccc.cancel()
	if got := ctx.Err(); got != Canceled {
		t.Errorf("after CancelCauseFunc ctx.Err() = %v, want %v", got, Canceled)
	}
	if got := Cause(ctx); got != Canceled {
		t.Errorf("after CancelCauseFunc Cause(ctx) = %v, want %v", got, Canceled)
	}
	if got := ccc.Err(); got != Canceled {
		t.Errorf("after CancelCauseFunc ccc.Err() = %v, want %v", got, Canceled)
	}
	if got := Cause(ccc); got != Canceled {
		t.Errorf("after CancelCauseFunc Cause(ccc) = %v, want %v", got, Canceled)
	}
}

func TestAfterFuncCalledAfterCancel(t *testing.T) {
	ctx, cancel := WithCancel(Background())
	donec := make(chan struct{})
	stop := AfterFunc(ctx, func() {
		close(donec)
	})
	select {
	case <-donec:
		t.Fatalf("AfterFunc called before context is done")
	case <-time.After(shortDuration):
	}
	cancel()
	select {
	case <-donec:
	case <-time.After(veryLongDuration):
		t.Fatalf("AfterFunc not called after context is canceled")
	}
	if stop() {
		t.Fatalf("stop() = true, want false")
	}
}

func TestAfterFuncCalledAfterTimeout(t *testing.T) {
	ctx, cancel := WithTimeout(Background(), shortDuration)
	defer cancel()
	donec := make(chan struct{})
	AfterFunc(ctx, func() {
		close(donec)
	})
	select {
	case <-donec:
	case <-time.After(veryLongDuration):
		t.Fatalf("AfterFunc not called after context is canceled")
	}
}

func TestAfterFuncCalledImmediately(t *testing.T) {
	ctx, cancel := WithCancel(Background())
	cancel()
	donec := make(chan struct{})
	AfterFunc(ctx, func() {
		close(donec)
	})
	select {
	case <-donec:
	case <-time.After(veryLongDuration):
		t.Fatalf("AfterFunc not called for already-canceled context")
	}
}

func TestAfterFuncNotCalledAfterStop(t *testing.T) {
	ctx, cancel := WithCancel(Background())
	donec := make(chan struct{})
	stop := AfterFunc(ctx, func() {
		close(donec)
	})
	if !stop() {
		t.Fatalf("stop() = false, want true")
	}
	cancel()
	select {
	case <-donec:
		t.Fatalf("AfterFunc called for already-canceled context")
	case <-time.After(shortDuration):
	}
	if stop() {
		t.Fatalf("stop() = true, want false")
	}
}

// This test verifies that canceling a context does not block waiting for AfterFuncs to finish.
func TestAfterFuncCalledAsynchronously(t *testing.T) {
	ctx, cancel := WithCancel(Background())
	donec := make(chan struct{})
	stop := AfterFunc(ctx, func() {
		// The channel send blocks until donec is read from.
		donec <- struct{}{}
	})
	defer stop()
	cancel()
	// After cancel returns, read from donec and unblock the AfterFunc.
	select {
	case <-donec:
	case <-time.After(veryLongDuration):
		t.Fatalf("AfterFunc not called after context is canceled")
	}
}
