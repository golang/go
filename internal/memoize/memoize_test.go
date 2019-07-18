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
	ctx := context.Background()
	s := &memoize.Store{}
	logBuffer := &bytes.Buffer{}
	ctx = context.WithValue(ctx, "logger", logBuffer)

	// These tests check the behavior of the Bind and Get functions.
	// They confirm that the functions only ever run once for a given value.
	for _, test := range []struct {
		name, key, want string
	}{
		{
			name: "nothing",
		},
		{
			name: "get 1",
			key:  "_1",
			want: `
start @1
simple a = A
simple b = B
simple c = C
end @1 =  A B C
`[1:],
		},
		{
			name: "redo 1",
			key:  "_1",
			want: ``,
		},
		{
			name: "get 2",
			key:  "_2",
			want: `
start @2
simple d = D
simple e = E
simple f = F
end @2 =  D E F
`[1:],
		},
		{
			name: "redo 2",
			key:  "_2",
			want: ``,
		},
		{
			name: "get 3",
			key:  "_3",
			want: `
start @3
end @3 =  @1[ A B C] @2[ D E F]
`[1:],
		},
		{
			name: "get 4",
			key:  "_4",
			want: `
start @3
simple g = G
error ERR = fail
simple h = H
end @3 =  G !fail H
`[1:],
		},
	} {
		s.Bind(test.key, generate(s, test.key)).Get(ctx)
		got := logBuffer.String()
		if got != test.want {
			t.Errorf("at %q expected:\n%v\ngot:\n%s", test.name, test.want, got)
		}
		logBuffer.Reset()
	}

	// This test checks that values are garbage collected and removed from the
	// store when they are no longer used.

	pinned := []string{"b", "_1", "_3"}             // keys to pin in memory
	unpinned := []string{"a", "c", "d", "_2", "_4"} // keys to garbage collect

	// Handles maintain a strong reference to their values.
	// By generating handles for the pinned keys and keeping the pins alive in memory,
	// we ensure these keys stay cached.
	var pins []*memoize.Handle
	for _, key := range pinned {
		h := s.Bind(key, generate(s, key))
		h.Get(ctx)
		pins = append(pins, h)
	}

	// Force the garbage collector to run.
	runAllFinalizers(t)

	// Confirm our expectation that pinned values should remain cached,
	// and unpinned values should be garbage collected.
	for _, k := range pinned {
		if v := s.Find(k); v == nil {
			t.Errorf("pinned value %q was nil", k)
		}
	}
	for _, k := range unpinned {
		if v := s.Find(k); v != nil {
			t.Errorf("unpinned value %q should be nil, was %v", k, v)
		}
	}

	// This forces the pins to stay alive until this point in the function.
	runtime.KeepAlive(pins)
}

func runAllFinalizers(t *testing.T) {
	// The following is very tricky, so be very when careful changing it.
	// It relies on behavior of finalizers that is not guaranteed.

	// First, run the GC to queue the finalizers.
	runtime.GC()

	// wait is used to signal that the finalizers are all done.
	wait := make(chan struct{})

	// Register a finalizer against an immediately collectible object.
	//
	// The finalizer will signal on the wait channel once it executes,
	// and it was the most recently registered finalizer,
	// so the wait channel will be closed when all of the finalizers have run.
	runtime.SetFinalizer(&struct{ s string }{"obj"}, func(_ interface{}) { close(wait) })

	// Now, run the GC again to pick up the tracker object above.
	runtime.GC()

	// Wait for the finalizers to run or a timeout.
	select {
	case <-wait:
	case <-time.Tick(time.Second):
		t.Fatalf("finalizers had not run after 1 second")
	}
}

type stringOrError struct {
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
		case "":
			return nil
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

// logGenerator generates a *stringOrError value, while logging to the store's logger.
func logGenerator(ctx context.Context, s *memoize.Store, name string, v string, err error) *stringOrError {
	// Get the logger from the context.
	w := ctx.Value("logger").(io.Writer)

	if err != nil {
		fmt.Fprintf(w, "error %v = %v\n", name, err)
	} else {
		fmt.Fprintf(w, "simple %v = %v\n", name, v)
	}
	return &stringOrError{value: v, err: err}
}

// joinValues binds a list of keys to their values, while logging to the store's logger.
func joinValues(ctx context.Context, s *memoize.Store, name string, keys ...string) *stringOrError {
	// Get the logger from the context.
	w := ctx.Value("logger").(io.Writer)

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
