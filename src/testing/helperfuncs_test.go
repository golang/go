// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"sync"
	"testing"
)

// The line numbering of this file is important for TestTBHelper.

func notHelper(t *testing.T, msg string) {
	t.Error(msg)
}

func helper(t *testing.T, msg string) {
	t.Helper()
	t.Error(msg)
}

func notHelperCallingHelper(t *testing.T, msg string) {
	helper(t, msg)
}

func helperCallingHelper(t *testing.T, msg string) {
	t.Helper()
	helper(t, msg)
}

func genericHelper[G any](t *testing.T, msg string) {
	t.Helper()
	t.Error(msg)
}

var genericIntHelper = genericHelper[int]

func testTestHelper(t *testing.T) {
	testHelper(t)
}

func testHelper(t *testing.T) {
	// Check combinations of directly and indirectly
	// calling helper functions.
	notHelper(t, "0")
	helper(t, "1")
	notHelperCallingHelper(t, "2")
	helperCallingHelper(t, "3")

	// Check a function literal closing over t that uses Helper.
	fn := func(msg string) {
		t.Helper()
		t.Error(msg)
	}
	fn("4")

	t.Run("sub", func(t *testing.T) {
		helper(t, "5")
		notHelperCallingHelper(t, "6")
		// Check that calling Helper from inside a subtest entry function
		// works as if it were in an ordinary function call.
		t.Helper()
		t.Error("7")
	})

	// Check that right caller is reported for func passed to Cleanup when
	// multiple cleanup functions have been registered.
	t.Cleanup(func() {
		t.Helper()
		t.Error("10")
	})
	t.Cleanup(func() {
		t.Helper()
		t.Error("9")
	})

	// Check that helper-ness propagates up through subtests
	// to helpers above. See https://golang.org/issue/44887.
	helperSubCallingHelper(t, "11")

	// Check that helper-ness propagates up through panic/recover.
	// See https://golang.org/issue/31154.
	recoverHelper(t, "12")

	genericHelper[float64](t, "GenericFloat64")
	genericIntHelper(t, "GenericInt")
}

func parallelTestHelper(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			notHelperCallingHelper(t, "parallel")
			wg.Done()
		}()
	}
	wg.Wait()
}

func helperSubCallingHelper(t *testing.T, msg string) {
	t.Helper()
	t.Run("sub2", func(t *testing.T) {
		t.Helper()
		t.Fatal(msg)
	})
}

func recoverHelper(t *testing.T, msg string) {
	t.Helper()
	defer func() {
		t.Helper()
		if err := recover(); err != nil {
			t.Errorf("recover %s", err)
		}
	}()
	doPanic(t, msg)
}

func doPanic(t *testing.T, msg string) {
	t.Helper()
	panic(msg)
}
