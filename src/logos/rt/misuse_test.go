// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"logos/rt"
)

func logosPanicMessage(t *testing.T, f func()) string {
	t.Helper()
	var got string
	func() {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("call succeeded, want panic")
			}
			s, ok := r.(string)
			if !ok {
				t.Fatalf("panic value is %T, want string", r)
			}
			got = s
		}()
		f()
	}()
	return got
}

func TestMustNonNilNilSitePanicsDefined(t *testing.T) {
	var p *int
	msg := logosPanicMessage(t, func() { rt.MustNonNil(p, nil) })
	if !strings.HasPrefix(msg, "logos:") || strings.Contains(msg, "runtime error") {
		t.Fatalf("MustNonNil(nil site) panic = %q, want defined logos: message", msg)
	}
}

func TestMustNonNilDocumentedSite(t *testing.T) {
	var p *int
	site := rt.Site{File: "use.go", Line: 9}
	msg := logosPanicMessage(t, func() { rt.MustNonNil(p, &site) })
	if msg != "logos: nil from Go value stored into non-nil *int at use.go:9" {
		t.Fatalf("MustNonNil panic = %q, want documented message", msg)
	}
	if got := rt.MustNonNil(3, nil); got != 3 {
		t.Fatalf("MustNonNil(3, nil) = %d, want passthrough", got)
	}
}

func TestDeferTryNilTargetPanicsDefined(t *testing.T) {
	msg := logosPanicMessage(t, func() { rt.DeferTry(nil, errors.New("boom")) })
	if !strings.HasPrefix(msg, "logos:") || strings.Contains(msg, "runtime error") {
		t.Fatalf("DeferTry(nil) panic = %q, want defined logos: message", msg)
	}
	var target error
	rt.DeferTry(&target, nil)
	if target != nil {
		t.Fatalf("DeferTry(nil result) set target to %v, want nil", target)
	}
	boom := errors.New("boom")
	rt.DeferTry(&target, boom)
	if target != boom {
		t.Fatalf("DeferTry into nil target = %v, want the result", target)
	}
}

func TestChanNilContextRanges(t *testing.T) {
	collect := func(seq func(func(int) bool)) []int {
		var out []int
		for v := range seq {
			out = append(out, v)
		}
		return out
	}
	mk := func() <-chan int {
		ch := make(chan int, 2)
		ch <- 1
		ch <- 2
		close(ch)
		return ch
	}
	var nilCtx context.Context
	got := collect(rt.Chan(nilCtx, mk()))
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("Chan(nil) collected %v, want [1 2]", got)
	}
	want := collect(rt.Chan(context.Background(), mk()))
	if len(want) != 2 || want[0] != 1 || want[1] != 2 {
		t.Fatalf("Chan(background) collected %v, want [1 2]", want)
	}
}

func TestConstTableZeroValueMisses(t *testing.T) {
	var tab rt.ConstTable[string, int]
	v, ok := tab.Get(12345, "missing")
	if ok || v != 0 {
		t.Fatalf("zero ConstTable.Get = (%v, %v), want (0, false)", v, ok)
	}
}
