// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"fmt"
	"time"
)

func Since() (t time.Duration) {
	return
}

func x(time.Duration) {}
func x2(float64)      {}

func good() {
	// The following are OK because func is not evaluated in defer invocation.
	now := time.Now()
	defer func() {
		fmt.Println(time.Since(now)) // OK because time.Since is not evaluated in defer
	}()
	evalBefore := time.Since(now)
	defer fmt.Println(evalBefore)
	do := func(f func()) {}
	defer do(func() { time.Since(now) })
	defer fmt.Println(Since())       // OK because Since function is not in module time
	defer copy([]int(nil), []int{1}) // check that a builtin doesn't cause a panic
}

type y struct{}

func (y) A(float64)        {}
func (*y) B(float64)       {}
func (y) C(time.Duration)  {}
func (*y) D(time.Duration) {}

func bad() {
	var zero time.Time
	now := time.Now()
	defer time.Since(zero)                    // want "call to time.Since is not deferred"
	defer time.Since(now)                     // want "call to time.Since is not deferred"
	defer fmt.Println(time.Since(now))        // want "call to time.Since is not deferred"
	defer fmt.Println(time.Since(time.Now())) // want "call to time.Since is not deferred"
	defer x(time.Since(now))                  // want "call to time.Since is not deferred"
	defer x2(time.Since(now).Seconds())       // want "call to time.Since is not deferred"
	defer y{}.A(time.Since(now).Seconds())    // want "call to time.Since is not deferred"
	defer (&y{}).B(time.Since(now).Seconds()) // want "call to time.Since is not deferred"
	defer y{}.C(time.Since(now))              // want "call to time.Since is not deferred"
	defer (&y{}).D(time.Since(now))           // want "call to time.Since is not deferred"
}

func ugly() {
	// The following is ok even though time.Since is evaluated. We don't
	// walk into function literals or check what function definitions are doing.
	defer x((func() time.Duration { return time.Since(time.Now()) })())
}
