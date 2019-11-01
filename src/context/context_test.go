// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context

import (
	"fmt"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type testingT interface {
	Error(args ...interface{})
	Errorf(format string, args ...interface{})
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...interface{})
	Fatalf(format string, args ...interface{})
	Helper()
	Log(args ...interface{})
	Logf(format string, args ...interface{})
	Name() string
	Skip(args ...interface{})
	SkipNow()
	Skipf(format string, args ...interface{})
	Skipped() bool
}

// otherContext is a Context that's not one of the types defined in context.go.
// This lets us test code paths that differ based on the underlying type of the
// Context.
type otherContext struct {
	Context
}

func XTestBackground(t testingT) {
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

func XTestTODO(t testingT) {
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

func XTestWithCancel(t testingT) {
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

	cancel()
	time.Sleep(100 * time.Millisecond) // let cancellation propagate

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

func contains(m map[canceler]struct{}, key canceler) bool {
	_, ret := m[key]
	return ret
}

func XTestParentFinishesChild(t testingT) {
	// Context tree:
	// parent -> cancelChild
	// parent -> valueChild -> timerChild
	parent, cancel := WithCancel(Background())
	cancelChild, stop := WithCancel(parent)
	defer stop()
	valueChild := WithValue(parent, "key", "value")
	timerChild, stop := WithTimeout(valueChild, 10000*time.Hour)
	defer stop()

	select {
	case x := <-parent.Done():
		t.Errorf("<-parent.Done() == %v want nothing (it should block)", x)
	case x := <-cancelChild.Done():
		t.Errorf("<-cancelChild.Done() == %v want nothing (it should block)", x)
	case x := <-timerChild.Done():
		t.Errorf("<-timerChild.Done() == %v want nothing (it should block)", x)
	case x := <-valueChild.Done():
		t.Errorf("<-valueChild.Done() == %v want nothing (it should block)", x)
	default:
	}

	// The parent's children should contain the two cancelable children.
	pc := parent.(*cancelCtx)
	cc := cancelChild.(*cancelCtx)
	tc := timerChild.(*timerCtx)
	pc.mu.Lock()
	if len(pc.children) != 2 || !contains(pc.children, cc) || !contains(pc.children, tc) {
		t.Errorf("bad linkage: pc.children = %v, want %v and %v",
			pc.children, cc, tc)
	}
	pc.mu.Unlock()

	if p, ok := parentCancelCtx(cc.Context); !ok || p != pc {
		t.Errorf("bad linkage: parentCancelCtx(cancelChild.Context) = %v, %v want %v, true", p, ok, pc)
	}
	if p, ok := parentCancelCtx(tc.Context); !ok || p != pc {
		t.Errorf("bad linkage: parentCancelCtx(timerChild.Context) = %v, %v want %v, true", p, ok, pc)
	}

	cancel()

	pc.mu.Lock()
	if len(pc.children) != 0 {
		t.Errorf("pc.cancel didn't clear pc.children = %v", pc.children)
	}
	pc.mu.Unlock()

	// parent and children should all be finished.
	check := func(ctx Context, name string) {
		select {
		case <-ctx.Done():
		default:
			t.Errorf("<-%s.Done() blocked, but shouldn't have", name)
		}
		if e := ctx.Err(); e != Canceled {
			t.Errorf("%s.Err() == %v want %v", name, e, Canceled)
		}
	}
	check(parent, "parent")
	check(cancelChild, "cancelChild")
	check(valueChild, "valueChild")
	check(timerChild, "timerChild")

	// WithCancel should return a canceled context on a canceled parent.
	precanceledChild := WithValue(parent, "key", "value")
	select {
	case <-precanceledChild.Done():
	default:
		t.Errorf("<-precanceledChild.Done() blocked, but shouldn't have")
	}
	if e := precanceledChild.Err(); e != Canceled {
		t.Errorf("precanceledChild.Err() == %v want %v", e, Canceled)
	}
}

func XTestChildFinishesFirst(t testingT) {
	cancelable, stop := WithCancel(Background())
	defer stop()
	for _, parent := range []Context{Background(), cancelable} {
		child, cancel := WithCancel(parent)

		select {
		case x := <-parent.Done():
			t.Errorf("<-parent.Done() == %v want nothing (it should block)", x)
		case x := <-child.Done():
			t.Errorf("<-child.Done() == %v want nothing (it should block)", x)
		default:
		}

		cc := child.(*cancelCtx)
		pc, pcok := parent.(*cancelCtx) // pcok == false when parent == Background()
		if p, ok := parentCancelCtx(cc.Context); ok != pcok || (ok && pc != p) {
			t.Errorf("bad linkage: parentCancelCtx(cc.Context) = %v, %v want %v, %v", p, ok, pc, pcok)
		}

		if pcok {
			pc.mu.Lock()
			if len(pc.children) != 1 || !contains(pc.children, cc) {
				t.Errorf("bad linkage: pc.children = %v, cc = %v", pc.children, cc)
			}
			pc.mu.Unlock()
		}

		cancel()

		if pcok {
			pc.mu.Lock()
			if len(pc.children) != 0 {
				t.Errorf("child's cancel didn't remove self from pc.children = %v", pc.children)
			}
			pc.mu.Unlock()
		}

		// child should be finished.
		select {
		case <-child.Done():
		default:
			t.Errorf("<-child.Done() blocked, but shouldn't have")
		}
		if e := child.Err(); e != Canceled {
			t.Errorf("child.Err() == %v want %v", e, Canceled)
		}

		// parent should not be finished.
		select {
		case x := <-parent.Done():
			t.Errorf("<-parent.Done() == %v want nothing (it should block)", x)
		default:
		}
		if e := parent.Err(); e != nil {
			t.Errorf("parent.Err() == %v want nil", e)
		}
	}
}

func testDeadline(c Context, name string, failAfter time.Duration, t testingT) {
	t.Helper()
	select {
	case <-time.After(failAfter):
		t.Fatalf("%s: context should have timed out", name)
	case <-c.Done():
	}
	if e := c.Err(); e != DeadlineExceeded {
		t.Errorf("%s: c.Err() == %v; want %v", name, e, DeadlineExceeded)
	}
}

func XTestDeadline(t testingT) {
	c, _ := WithDeadline(Background(), time.Now().Add(50*time.Millisecond))
	if got, prefix := fmt.Sprint(c), "context.Background.WithDeadline("; !strings.HasPrefix(got, prefix) {
		t.Errorf("c.String() = %q want prefix %q", got, prefix)
	}
	testDeadline(c, "WithDeadline", time.Second, t)

	c, _ = WithDeadline(Background(), time.Now().Add(50*time.Millisecond))
	o := otherContext{c}
	testDeadline(o, "WithDeadline+otherContext", time.Second, t)

	c, _ = WithDeadline(Background(), time.Now().Add(50*time.Millisecond))
	o = otherContext{c}
	c, _ = WithDeadline(o, time.Now().Add(4*time.Second))
	testDeadline(c, "WithDeadline+otherContext+WithDeadline", 2*time.Second, t)

	c, _ = WithDeadline(Background(), time.Now().Add(-time.Millisecond))
	testDeadline(c, "WithDeadline+inthepast", time.Second, t)

	c, _ = WithDeadline(Background(), time.Now())
	testDeadline(c, "WithDeadline+now", time.Second, t)
}

func XTestTimeout(t testingT) {
	c, _ := WithTimeout(Background(), 50*time.Millisecond)
	if got, prefix := fmt.Sprint(c), "context.Background.WithDeadline("; !strings.HasPrefix(got, prefix) {
		t.Errorf("c.String() = %q want prefix %q", got, prefix)
	}
	testDeadline(c, "WithTimeout", time.Second, t)

	c, _ = WithTimeout(Background(), 50*time.Millisecond)
	o := otherContext{c}
	testDeadline(o, "WithTimeout+otherContext", time.Second, t)

	c, _ = WithTimeout(Background(), 50*time.Millisecond)
	o = otherContext{c}
	c, _ = WithTimeout(o, 3*time.Second)
	testDeadline(c, "WithTimeout+otherContext+WithTimeout", 2*time.Second, t)
}

func XTestCanceledTimeout(t testingT) {
	c, _ := WithTimeout(Background(), time.Second)
	o := otherContext{c}
	c, cancel := WithTimeout(o, 2*time.Second)
	cancel()
	time.Sleep(100 * time.Millisecond) // let cancellation propagate
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

var k1 = key1(1)
var k2 = key2(1) // same int as k1, different type
var k3 = key2(3) // same type as k2, different int

func XTestValues(t testingT) {
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

	if got, want := fmt.Sprint(c1), `context.Background.WithValue(type context.key1, val c1k1)`; got != want {
		t.Errorf("c.String() = %q want %q", got, want)
	}

	c2 := WithValue(c1, k2, "c2k2")
	check(c2, "c2", "c1k1", "c2k2", "")

	c3 := WithValue(c2, k3, "c3k3")
	check(c3, "c2", "c1k1", "c2k2", "c3k3")

	c4 := WithValue(c3, k1, nil)
	check(c4, "c4", "", "c2k2", "c3k3")

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

func XTestAllocs(t testingT, testingShort func() bool, testingAllocsPerRun func(int, func()) float64) {
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
			desc: "WithTimeout(bg, 15*time.Millisecond)",
			f: func() {
				c, _ := WithTimeout(bg, 15*time.Millisecond)
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
		if testingShort() {
			numRuns = 10
		}
		if n := testingAllocsPerRun(numRuns, test.f); n > limit {
			t.Errorf("%s allocs = %f want %d", test.desc, n, int(limit))
		}
	}
}

func XTestSimultaneousCancels(t testingT) {
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
	// Wait on all the contexts in a random order.
	for ctx := range m {
		select {
		case <-ctx.Done():
		case <-time.After(1 * time.Second):
			buf := make([]byte, 10<<10)
			n := runtime.Stack(buf, true)
			t.Fatalf("timed out waiting for <-ctx.Done(); stacks:\n%s", buf[:n])
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
	case <-time.After(1 * time.Second):
		buf := make([]byte, 10<<10)
		n := runtime.Stack(buf, true)
		t.Fatalf("timed out waiting for cancel functions; stacks:\n%s", buf[:n])
	}
}

func XTestInterlockedCancels(t testingT) {
	parent, cancelParent := WithCancel(Background())
	child, cancelChild := WithCancel(parent)
	go func() {
		parent.Done()
		cancelChild()
	}()
	cancelParent()
	select {
	case <-child.Done():
	case <-time.After(1 * time.Second):
		buf := make([]byte, 10<<10)
		n := runtime.Stack(buf, true)
		t.Fatalf("timed out waiting for child.Done(); stacks:\n%s", buf[:n])
	}
}

func XTestLayersCancel(t testingT) {
	testLayers(t, time.Now().UnixNano(), false)
}

func XTestLayersTimeout(t testingT) {
	testLayers(t, time.Now().UnixNano(), true)
}

func testLayers(t testingT, seed int64, testTimeout bool) {
	rand.Seed(seed)
	errorf := func(format string, a ...interface{}) {
		t.Errorf(fmt.Sprintf("seed=%d: %s", seed, format), a...)
	}
	const (
		timeout   = 200 * time.Millisecond
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
		switch rand.Intn(3) {
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
			ctx, cancel = WithTimeout(ctx, timeout)
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
	select {
	case <-ctx.Done():
		errorf("ctx should not be canceled yet")
	default:
	}
	if s, prefix := fmt.Sprint(ctx), "context.Background."; !strings.HasPrefix(s, prefix) {
		t.Errorf("ctx.String() = %q want prefix %q", s, prefix)
	}
	t.Log(ctx)
	checkValues("before cancel")
	if testTimeout {
		select {
		case <-ctx.Done():
		case <-time.After(timeout + time.Second):
			errorf("ctx should have timed out")
		}
		checkValues("after timeout")
	} else {
		cancel := cancels[rand.Intn(len(cancels))]
		cancel()
		select {
		case <-ctx.Done():
		default:
			errorf("ctx should be canceled")
		}
		checkValues("after cancel")
	}
}

func XTestCancelRemoves(t testingT) {
	checkChildren := func(when string, ctx Context, want int) {
		if got := len(ctx.(*cancelCtx).children); got != want {
			t.Errorf("%s: context has %d children, want %d", when, got, want)
		}
	}

	ctx, _ := WithCancel(Background())
	checkChildren("after creation", ctx, 0)
	_, cancel := WithCancel(ctx)
	checkChildren("with WithCancel child ", ctx, 1)
	cancel()
	checkChildren("after canceling WithCancel child", ctx, 0)

	ctx, _ = WithCancel(Background())
	checkChildren("after creation", ctx, 0)
	_, cancel = WithTimeout(ctx, 60*time.Minute)
	checkChildren("with WithTimeout child ", ctx, 1)
	cancel()
	checkChildren("after canceling WithTimeout child", ctx, 0)
}

func XTestWithCancelCanceledParent(t testingT) {
	parent, pcancel := WithCancel(Background())
	pcancel()

	c, _ := WithCancel(parent)
	select {
	case <-c.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for Done")
	}
	if got, want := c.Err(), Canceled; got != want {
		t.Errorf("child not cancelled; got = %v, want = %v", got, want)
	}
}

func XTestWithValueChecksKey(t testingT) {
	panicVal := recoveredValue(func() { WithValue(Background(), []byte("foo"), "bar") })
	if panicVal == nil {
		t.Error("expected panic")
	}
	panicVal = recoveredValue(func() { WithValue(Background(), nil, "bar") })
	if got, want := fmt.Sprint(panicVal), "nil key"; got != want {
		t.Errorf("panic = %q; want %q", got, want)
	}
}

func recoveredValue(fn func()) (v interface{}) {
	defer func() { v = recover() }()
	fn()
	return
}

func XTestDeadlineExceededSupportsTimeout(t testingT) {
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

type myCtx struct {
	Context
}

type myDoneCtx struct {
	Context
}

func (d *myDoneCtx) Done() <-chan struct{} {
	c := make(chan struct{})
	return c
}

func XTestCustomContextGoroutines(t testingT) {
	g := atomic.LoadInt32(&goroutines)
	checkNoGoroutine := func() {
		t.Helper()
		now := atomic.LoadInt32(&goroutines)
		if now != g {
			t.Fatalf("%d goroutines created", now-g)
		}
	}
	checkCreatedGoroutine := func() {
		t.Helper()
		now := atomic.LoadInt32(&goroutines)
		if now != g+1 {
			t.Fatalf("%d goroutines created, want 1", now-g)
		}
		g = now
	}

	_, cancel0 := WithCancel(&myDoneCtx{Background()})
	cancel0()
	checkCreatedGoroutine()

	_, cancel0 = WithTimeout(&myDoneCtx{Background()}, 1*time.Hour)
	cancel0()
	checkCreatedGoroutine()

	checkNoGoroutine()
	defer checkNoGoroutine()

	ctx1, cancel1 := WithCancel(Background())
	defer cancel1()
	checkNoGoroutine()

	ctx2 := &myCtx{ctx1}
	ctx3, cancel3 := WithCancel(ctx2)
	defer cancel3()
	checkNoGoroutine()

	_, cancel3b := WithCancel(&myDoneCtx{ctx2})
	defer cancel3b()
	checkCreatedGoroutine() // ctx1 is not providing Done, must not be used

	ctx4, cancel4 := WithTimeout(ctx3, 1*time.Hour)
	defer cancel4()
	checkNoGoroutine()

	ctx5, cancel5 := WithCancel(ctx4)
	defer cancel5()
	checkNoGoroutine()

	cancel5()
	checkNoGoroutine()

	_, cancel6 := WithTimeout(ctx5, 1*time.Hour)
	defer cancel6()
	checkNoGoroutine()

	// Check applied to cancelled context.
	cancel6()
	cancel1()
	_, cancel7 := WithCancel(ctx5)
	defer cancel7()
	checkNoGoroutine()
}
