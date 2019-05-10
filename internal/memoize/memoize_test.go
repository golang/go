// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memoize_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"runtime"
	"strings"
	"testing"
	"time"

	"golang.org/x/tools/internal/memoize"
)

func TestStore(t *testing.T) {
	pinned := []string{"b", "_1", "_3"}
	unpinned := []string{"a", "c", "d", "_2", "_4"}
	ctx := context.Background()
	s := &memoize.Store{}
	logBuffer := &bytes.Buffer{}
	s.Bind("logger", func(context.Context) interface{} { return logBuffer }).Get(ctx)
	verifyBuffer := func(name, expect string) {
		got := logBuffer.String()
		if got != expect {
			t.Errorf("at %q expected:\n%v\ngot:\n%s", name, expect, got)
		}
		logBuffer.Reset()
	}
	verifyBuffer("nothing", ``)
	s.Bind("_1", generate(s, "_1")).Get(ctx)
	verifyBuffer("get 1", `
start @1
simple a = A
simple b = B
simple c = C
end @1 =  A B C
`[1:])
	s.Bind("_1", generate(s, "_1")).Get(ctx)
	verifyBuffer("redo 1", ``)
	s.Bind("_2", generate(s, "_2")).Get(ctx)
	verifyBuffer("get 2", `
start @2
simple d = D
simple e = E
simple f = F
end @2 =  D E F
`[1:])
	s.Bind("_2", generate(s, "_2")).Get(ctx)
	verifyBuffer("redo 2", ``)
	s.Bind("_3", generate(s, "_3")).Get(ctx)
	verifyBuffer("get 3", `
start @3
end @3 =  @1[ A B C] @2[ D E F]
`[1:])
	s.Bind("_4", generate(s, "_4")).Get(ctx)
	verifyBuffer("get 4", `
start @3
simple g = G
error ERR = fail
simple h = H
end @3 =  G !fail H
`[1:])

	var pins []*memoize.Handle
	for _, key := range pinned {
		h := s.Bind(key, generate(s, key))
		h.Get(ctx)
		pins = append(pins, h)
	}

	runAllFinalizers(t)

	for _, k := range pinned {
		if v := s.Cached(k); v == nil {
			t.Errorf("Pinned value %q was nil", k)
		}
	}
	for _, k := range unpinned {
		if v := s.Cached(k); v != nil {
			t.Errorf("Unpinned value %q was %q", k, v)
		}
	}
	runtime.KeepAlive(pins)
}

func runAllFinalizers(t *testing.T) {
	// the following is very tricky, be very careful changing it
	// it relies on behavior of finalizers that is not guaranteed
	// first run the GC to queue the finalizers
	runtime.GC()
	// wait is used to signal that the finalizers are all done
	wait := make(chan struct{})
	// register a finalizer against an immediately collectible object
	runtime.SetFinalizer(&struct{ s string }{"obj"}, func(_ interface{}) { close(wait) })
	// now run the GC again to pick up the tracker
	runtime.GC()
	// now wait for the finalizers to run
	select {
	case <-wait:
	case <-time.Tick(time.Second):
		t.Fatalf("Finalizers had not run after a second")
	}
}

type stringOrError struct {
	memoize.NoCopy
	value string
	err   error
}

func (v *stringOrError) String() string {
	if v.err != nil {
		return v.err.Error()
	}
	return v.value
}

func asValue(v interface{}) *stringOrError {
	if v == nil {
		return nil
	}
	return v.(*stringOrError)
}

func generate(s *memoize.Store, key interface{}) memoize.Function {
	return func(ctx context.Context) interface{} {
		name := key.(string)
		switch name {
		case "err":
			return logGenerator(ctx, s, "ERR", "", fmt.Errorf("fail"))
		case "_1":
			return joinValues(ctx, s, "@1", "a", "b", "c")
		case "_2":
			return joinValues(ctx, s, "@2", "d", "e", "f")
		case "_3":
			return joinValues(ctx, s, "@3", "_1", "_2")
		case "_4":
			return joinValues(ctx, s, "@3", "g", "err", "h")
		default:
			return logGenerator(ctx, s, name, strings.ToUpper(name), nil)
		}
	}
}

func logGenerator(ctx context.Context, s *memoize.Store, name string, v string, err error) *stringOrError {
	w := s.Cached("logger").(io.Writer)
	if err != nil {
		fmt.Fprintf(w, "error %v = %v\n", name, err)
	} else {
		fmt.Fprintf(w, "simple %v = %v\n", name, v)
	}
	return &stringOrError{value: v, err: err}
}

func joinValues(ctx context.Context, s *memoize.Store, name string, keys ...string) *stringOrError {
	w := s.Cached("logger").(io.Writer)
	fmt.Fprintf(w, "start %v\n", name)
	value := ""
	for _, key := range keys {
		v := asValue(s.Bind(key, generate(s, key)).Get(ctx))
		if v == nil {
			value = value + " <nil>"
		} else if v.err != nil {
			value = value + " !" + v.err.Error()
		} else {
			value = value + " " + v.value
		}
	}
	fmt.Fprintf(w, "end %v = %v\n", name, value)
	return &stringOrError{value: fmt.Sprintf("%s[%v]", name, value)}
}
