// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"strconv"
	"testing"
)

func BenchmarkTraceStack(b *testing.B) {
	for _, stackDepth := range []int{1, 10, 100} {
		b.Run("stackDepth="+strconv.Itoa(stackDepth), func(b *testing.B) {
			benchmarkTraceStack(b, stackDepth)
		})
	}
}

func benchmarkTraceStack(b *testing.B, stackDepth int) {
	var tab runtime.TraceStackTable
	defer tab.Reset()

	wait := make(chan struct{})
	ready := make(chan struct{})
	done := make(chan struct{})
	var gp *runtime.G
	go func() {
		gp = runtime.Getg()
		useStackAndCall(stackDepth, func() {
			ready <- struct{}{}
			<-wait
		})
		done <- struct{}{}
	}()
	<-ready

	for b.Loop() {
		runtime.TraceStack(gp, &tab)
	}

	// Clean up.
	wait <- struct{}{}
	<-done
}
