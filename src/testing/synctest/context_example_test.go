// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.synctest

package synctest_test

import (
	"context"
	"fmt"
	"testing/synctest"
	"time"
)

// This example demonstrates testing the context.AfterFunc function.
//
// AfterFunc registers a function to execute in a new goroutine
// after a context is canceled.
//
// The test verifies that the function is not run before the context is canceled,
// and is run after the context is canceled.
func Example_contextAfterFunc() {
	synctest.Run(func() {
		// Create a context.Context which can be canceled.
		ctx, cancel := context.WithCancel(context.Background())

		// context.AfterFunc registers a function to be called
		// when a context is canceled.
		afterFuncCalled := false
		context.AfterFunc(ctx, func() {
			afterFuncCalled = true
		})

		// The context has not been canceled, so the AfterFunc is not called.
		synctest.Wait()
		fmt.Printf("before context is canceled: afterFuncCalled=%v\n", afterFuncCalled)

		// Cancel the context and wait for the AfterFunc to finish executing.
		// Verify that the AfterFunc ran.
		cancel()
		synctest.Wait()
		fmt.Printf("after context is canceled:  afterFuncCalled=%v\n", afterFuncCalled)

		// Output:
		// before context is canceled: afterFuncCalled=false
		// after context is canceled:  afterFuncCalled=true
	})
}

// This example demonstrates testing the context.WithTimeout function.
//
// WithTimeout creates a context which is canceled after a timeout.
//
// The test verifies that the context is not canceled before the timeout expires,
// and is canceled after the timeout expires.
func Example_contextWithTimeout() {
	synctest.Run(func() {
		// Create a context.Context which is canceled after a timeout.
		const timeout = 5 * time.Second
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()

		// Wait just less than the timeout.
		time.Sleep(timeout - time.Nanosecond)
		synctest.Wait()
		fmt.Printf("before timeout: ctx.Err() = %v\n", ctx.Err())

		// Wait the rest of the way until the timeout.
		time.Sleep(time.Nanosecond)
		synctest.Wait()
		fmt.Printf("after timeout:  ctx.Err() = %v\n", ctx.Err())

		// Output:
		// before timeout: ctx.Err() = <nil>
		// after timeout:  ctx.Err() = context deadline exceeded
	})
}
