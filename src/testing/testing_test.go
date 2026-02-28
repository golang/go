// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"os"
	"path/filepath"
	"testing"
)

// This is exactly what a test would do without a TestMain.
// It's here only so that there is at least one package in the
// standard library with a TestMain, so that code is executed.

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}

func TestTempDirInCleanup(t *testing.T) {
	var dir string

	t.Run("test", func(t *testing.T) {
		t.Cleanup(func() {
			dir = t.TempDir()
		})
		_ = t.TempDir()
	})

	fi, err := os.Stat(dir)
	if fi != nil {
		t.Fatalf("Directory %q from user Cleanup still exists", dir)
	}
	if !os.IsNotExist(err) {
		t.Fatalf("Unexpected error: %v", err)
	}
}

func TestTempDirInBenchmark(t *testing.T) {
	testing.Benchmark(func(b *testing.B) {
		if !b.Run("test", func(b *testing.B) {
			// Add a loop so that the test won't fail. See issue 38677.
			for i := 0; i < b.N; i++ {
				_ = b.TempDir()
			}
		}) {
			t.Fatal("Sub test failure in a benchmark")
		}
	})
}

func TestTempDir(t *testing.T) {
	testTempDir(t)
	t.Run("InSubtest", testTempDir)
	t.Run("test/subtest", testTempDir)
	t.Run("test\\subtest", testTempDir)
	t.Run("test:subtest", testTempDir)
	t.Run("test/..", testTempDir)
	t.Run("../test", testTempDir)
	t.Run("test[]", testTempDir)
	t.Run("test*", testTempDir)
	t.Run("äöüéè", testTempDir)
}

func testTempDir(t *testing.T) {
	dirCh := make(chan string, 1)
	t.Cleanup(func() {
		// Verify directory has been removed.
		select {
		case dir := <-dirCh:
			fi, err := os.Stat(dir)
			if os.IsNotExist(err) {
				// All good
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			t.Errorf("directory %q still exists: %v, isDir=%v", dir, fi, fi.IsDir())
		default:
			if !t.Failed() {
				t.Fatal("never received dir channel")
			}
		}
	})

	dir := t.TempDir()
	if dir == "" {
		t.Fatal("expected dir")
	}
	dir2 := t.TempDir()
	if dir == dir2 {
		t.Fatal("subsequent calls to TempDir returned the same directory")
	}
	if filepath.Dir(dir) != filepath.Dir(dir2) {
		t.Fatalf("calls to TempDir do not share a parent; got %q, %q", dir, dir2)
	}
	dirCh <- dir
	fi, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !fi.IsDir() {
		t.Errorf("dir %q is not a dir", dir)
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) > 0 {
		t.Errorf("unexpected %d files in TempDir: %v", len(files), files)
	}

	glob := filepath.Join(dir, "*.txt")
	if _, err := filepath.Glob(glob); err != nil {
		t.Error(err)
	}
}

func TestSetenv(t *testing.T) {
	tests := []struct {
		name               string
		key                string
		initialValueExists bool
		initialValue       string
		newValue           string
	}{
		{
			name:               "initial value exists",
			key:                "GO_TEST_KEY_1",
			initialValueExists: true,
			initialValue:       "111",
			newValue:           "222",
		},
		{
			name:               "initial value exists but empty",
			key:                "GO_TEST_KEY_2",
			initialValueExists: true,
			initialValue:       "",
			newValue:           "222",
		},
		{
			name:               "initial value is not exists",
			key:                "GO_TEST_KEY_3",
			initialValueExists: false,
			initialValue:       "",
			newValue:           "222",
		},
	}

	for _, test := range tests {
		if test.initialValueExists {
			if err := os.Setenv(test.key, test.initialValue); err != nil {
				t.Fatalf("unable to set env: got %v", err)
			}
		} else {
			os.Unsetenv(test.key)
		}

		t.Run(test.name, func(t *testing.T) {
			t.Setenv(test.key, test.newValue)
			if os.Getenv(test.key) != test.newValue {
				t.Fatalf("unexpected value after t.Setenv: got %s, want %s", os.Getenv(test.key), test.newValue)
			}
		})

		got, exists := os.LookupEnv(test.key)
		if got != test.initialValue {
			t.Fatalf("unexpected value after t.Setenv cleanup: got %s, want %s", got, test.initialValue)
		}
		if exists != test.initialValueExists {
			t.Fatalf("unexpected value after t.Setenv cleanup: got %t, want %t", exists, test.initialValueExists)
		}
	}
}

func TestSetenvWithParallelAfterSetenv(t *testing.T) {
	defer func() {
		want := "testing: t.Parallel called after t.Setenv; cannot set environment variables in parallel tests"
		if got := recover(); got != want {
			t.Fatalf("expected panic; got %#v want %q", got, want)
		}
	}()

	t.Setenv("GO_TEST_KEY_1", "value")

	t.Parallel()
}

func TestSetenvWithParallelBeforeSetenv(t *testing.T) {
	defer func() {
		want := "testing: t.Setenv called after t.Parallel; cannot set environment variables in parallel tests"
		if got := recover(); got != want {
			t.Fatalf("expected panic; got %#v want %q", got, want)
		}
	}()

	t.Parallel()

	t.Setenv("GO_TEST_KEY_1", "value")
}
