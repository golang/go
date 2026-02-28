// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loopclosure defines an Analyzer that checks for references to
// enclosing loop variables from within nested functions.
//
// # Analyzer loopclosure
//
// loopclosure: check references to loop variables from within nested functions
//
// This analyzer reports places where a function literal references the
// iteration variable of an enclosing loop, and the loop calls the function
// in such a way (e.g. with go or defer) that it may outlive the loop
// iteration and possibly observe the wrong value of the variable.
//
// Note: An iteration variable can only outlive a loop iteration in Go versions <=1.21.
// In Go 1.22 and later, the loop variable lifetimes changed to create a new
// iteration variable per loop iteration. (See go.dev/issue/60078.)
//
// In this example, all the deferred functions run after the loop has
// completed, so all observe the final value of v [<go1.22].
//
//	for _, v := range list {
//	    defer func() {
//	        use(v) // incorrect
//	    }()
//	}
//
// One fix is to create a new variable for each iteration of the loop:
//
//	for _, v := range list {
//	    v := v // new var per iteration
//	    defer func() {
//	        use(v) // ok
//	    }()
//	}
//
// After Go version 1.22, the previous two for loops are equivalent
// and both are correct.
//
// The next example uses a go statement and has a similar problem [<go1.22].
// In addition, it has a data race because the loop updates v
// concurrent with the goroutines accessing it.
//
//	for _, v := range elem {
//	    go func() {
//	        use(v)  // incorrect, and a data race
//	    }()
//	}
//
// A fix is the same as before. The checker also reports problems
// in goroutines started by golang.org/x/sync/errgroup.Group.
// A hard-to-spot variant of this form is common in parallel tests:
//
//	func Test(t *testing.T) {
//	    for _, test := range tests {
//	        t.Run(test.name, func(t *testing.T) {
//	            t.Parallel()
//	            use(test) // incorrect, and a data race
//	        })
//	    }
//	}
//
// The t.Parallel() call causes the rest of the function to execute
// concurrent with the loop [<go1.22].
//
// The analyzer reports references only in the last statement,
// as it is not deep enough to understand the effects of subsequent
// statements that might render the reference benign.
// ("Last statement" is defined recursively in compound
// statements such as if, switch, and select.)
//
// See: https://golang.org/doc/go_faq.html#closures_and_goroutines
package loopclosure
