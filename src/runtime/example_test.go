// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"runtime"
)

func ExampleFrames() {
	c := func() {
		pc := make([]uintptr, 5)
		n := runtime.Callers(0, pc)
		if n == 0 {
			return
		}

		frames := runtime.CallersFrames(pc[:n])
		var frame runtime.Frame
		for i, more := 0, true; more && i < 5; i++ {
			frame, more = frames.Next()
			fmt.Printf("- %s\n", frame.Function)
		}
	}

	b := func() { c() }
	a := func() { b() }

	a()
	// Output:
	// - runtime.Callers
	// - runtime_test.ExampleFrames.func1
	// - runtime_test.ExampleFrames.func2
	// - runtime_test.ExampleFrames.func3
	// - runtime_test.ExampleFrames
}
