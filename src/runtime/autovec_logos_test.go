// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"runtime/debug"
	"testing"
)

func TestLogosAutovecPanicOnFault(t *testing.T) {
	original := debug.SetPanicOnFault(false)
	defer debug.SetPanicOnFault(original)
	initial := runtime.LogosAutovecPanicOnFault()
	first := make(chan [2]bool)
	second := make(chan bool)
	next := make(chan struct{})
	done := make(chan struct{})
	go func() {
		old := debug.SetPanicOnFault(true)
		first <- [2]bool{old, runtime.LogosAutovecPanicOnFault()}
		<-next
		debug.SetPanicOnFault(false)
		second <- runtime.LogosAutovecPanicOnFault()
		<-next
		debug.SetPanicOnFault(old)
		close(done)
	}()
	child := <-first
	parentWhileChildEnabled := runtime.LogosAutovecPanicOnFault()
	debug.SetPanicOnFault(true)
	next <- struct{}{}
	childWhileParentEnabled := <-second
	parentWhileChildDisabled := runtime.LogosAutovecPanicOnFault()
	next <- struct{}{}
	<-done
	previous := debug.SetPanicOnFault(false)
	restored := runtime.LogosAutovecPanicOnFault()

	for _, test := range []struct {
		name string
		got  bool
		want bool
	}{
		{"initial disabled", initial, false},
		{"child default", child[0], false},
		{"child enabled", child[1], true},
		{"parent stays disabled", parentWhileChildEnabled, false},
		{"child stays disabled", childWhileParentEnabled, false},
		{"parent stays enabled", parentWhileChildDisabled, true},
		{"previous enabled", previous, true},
		{"restored disabled", restored, false},
	} {
		if test.got != test.want {
			t.Errorf("%s: got %t want %t", test.name, test.got, test.want)
		}
	}
}
