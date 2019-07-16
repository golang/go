// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package failfast

import "testing"

func TestA(t *testing.T) {
	// Edge-case testing, mixing unparallel tests too
	t.Logf("LOG: %s", t.Name())
}

func TestFailingA(t *testing.T) {
	t.Errorf("FAIL - %s", t.Name())
}

func TestB(t *testing.T) {
	// Edge-case testing, mixing unparallel tests too
	t.Logf("LOG: %s", t.Name())
}

func TestParallelFailingA(t *testing.T) {
	t.Parallel()
	t.Errorf("FAIL - %s", t.Name())
}

func TestParallelFailingB(t *testing.T) {
	t.Parallel()
	t.Errorf("FAIL - %s", t.Name())
}

func TestParallelFailingSubtestsA(t *testing.T) {
	t.Parallel()
	t.Run("TestFailingSubtestsA1", func(t *testing.T) {
		t.Errorf("FAIL - %s", t.Name())
	})
	t.Run("TestFailingSubtestsA2", func(t *testing.T) {
		t.Errorf("FAIL - %s", t.Name())
	})
}

func TestFailingSubtestsA(t *testing.T) {
	t.Run("TestFailingSubtestsA1", func(t *testing.T) {
		t.Errorf("FAIL - %s", t.Name())
	})
	t.Run("TestFailingSubtestsA2", func(t *testing.T) {
		t.Errorf("FAIL - %s", t.Name())
	})
}

func TestFailingB(t *testing.T) {
	t.Errorf("FAIL - %s", t.Name())
}

func TestFatalC(t *testing.T) {
	t.Fatalf("FAIL - %s", t.Name())
}

func TestFatalD(t *testing.T) {
	t.Fatalf("FAIL - %s", t.Name())
}
