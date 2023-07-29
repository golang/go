// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context

// Tests in package context cannot depend directly on package testing due to an import cycle.
// If your test does requires access to unexported members of the context package,
// add your test below as `func XTestFoo(t testingT)` and add a `TestFoo` to x_test.go
// that calls it. Otherwise, write a regular test in a test.go file in package context_test.

import (
	"time"
)

type testingT interface {
	Deadline() (time.Time, bool)
	Error(args ...any)
	Errorf(format string, args ...any)
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Helper()
	Log(args ...any)
	Logf(format string, args ...any)
	Name() string
	Parallel()
	Skip(args ...any)
	SkipNow()
	Skipf(format string, args ...any)
	Skipped() bool
}

const veryLongDuration = 1000 * time.Hour // an arbitrary upper bound on the test's running time

func contains(m map[canceler]struct{}, key canceler) bool {
	_, ret := m[key]
	return ret
}

func XTestParentFinishesChild(t testingT) {
	// Context tree:
	// parent -> cancelChild
	// parent -> valueChild -> timerChild
	// parent -> afterChild
	parent, cancel := WithCancel(Background())
	cancelChild, stop := WithCancel(parent)
	defer stop()
	valueChild := WithValue(parent, "key", "value")
	timerChild, stop := WithTimeout(valueChild, veryLongDuration)
	defer stop()
	afterStop := AfterFunc(parent, func() {})
	defer afterStop()

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

	// The parent's children should contain the three cancelable children.
	pc := parent.(*cancelCtx)
	cc := cancelChild.(*cancelCtx)
	tc := timerChild.(*timerCtx)
	pc.mu.Lock()
	var ac *afterFuncCtx
	for c := range pc.children {
		if a, ok := c.(*afterFuncCtx); ok {
			ac = a
			break
		}
	}
	if len(pc.children) != 3 || !contains(pc.children, cc) || !contains(pc.children, tc) || ac == nil {
		t.Errorf("bad linkage: pc.children = %v, want %v, %v, and an afterFunc",
			pc.children, cc, tc)
	}
	pc.mu.Unlock()

	if p, ok := parentCancelCtx(cc.Context); !ok || p != pc {
		t.Errorf("bad linkage: parentCancelCtx(cancelChild.Context) = %v, %v want %v, true", p, ok, pc)
	}
	if p, ok := parentCancelCtx(tc.Context); !ok || p != pc {
		t.Errorf("bad linkage: parentCancelCtx(timerChild.Context) = %v, %v want %v, true", p, ok, pc)
	}
	if p, ok := parentCancelCtx(ac.Context); !ok || p != pc {
		t.Errorf("bad linkage: parentCancelCtx(afterChild.Context) = %v, %v want %v, true", p, ok, pc)
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

	ctx, _ = WithCancel(Background())
	checkChildren("after creation", ctx, 0)
	stop := AfterFunc(ctx, func() {})
	checkChildren("with AfterFunc child ", ctx, 1)
	stop()
	checkChildren("after stopping AfterFunc child ", ctx, 0)
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
	g := goroutines.Load()
	checkNoGoroutine := func() {
		t.Helper()
		now := goroutines.Load()
		if now != g {
			t.Fatalf("%d goroutines created", now-g)
		}
	}
	checkCreatedGoroutine := func() {
		t.Helper()
		now := goroutines.Load()
		if now != g+1 {
			t.Fatalf("%d goroutines created, want 1", now-g)
		}
		g = now
	}

	_, cancel0 := WithCancel(&myDoneCtx{Background()})
	cancel0()
	checkCreatedGoroutine()

	_, cancel0 = WithTimeout(&myDoneCtx{Background()}, veryLongDuration)
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

	ctx4, cancel4 := WithTimeout(ctx3, veryLongDuration)
	defer cancel4()
	checkNoGoroutine()

	ctx5, cancel5 := WithCancel(ctx4)
	defer cancel5()
	checkNoGoroutine()

	cancel5()
	checkNoGoroutine()

	_, cancel6 := WithTimeout(ctx5, veryLongDuration)
	defer cancel6()
	checkNoGoroutine()

	// Check applied to canceled context.
	cancel6()
	cancel1()
	_, cancel7 := WithCancel(ctx5)
	defer cancel7()
	checkNoGoroutine()
}
