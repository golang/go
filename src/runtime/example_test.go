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
		more := true
		for more {
			frame, more = frames.Next()
			fmt.Printf("- more:%v | %s\n", more, frame.Function)
		}
	}

	b := func() { c() }
	a := func() { b() }

	a()
	// Output:
	// - more:true | runtime.Callers
	// - more:true | runtime_test.ExampleFrames.func1
	// - more:true | runtime_test.ExampleFrames.func2
	// - more:true | runtime_test.ExampleFrames.func3
	// - more:false | runtime_test.ExampleFrames
}
