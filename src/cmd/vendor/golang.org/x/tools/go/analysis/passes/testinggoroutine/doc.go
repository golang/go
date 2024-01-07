// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testinggoroutine defines an Analyzerfor detecting calls to
// Fatal from a test goroutine.
//
// # Analyzer testinggoroutine
//
// testinggoroutine: report calls to (*testing.T).Fatal from goroutines started by a test
//
// Functions that abruptly terminate a test, such as the Fatal, Fatalf, FailNow, and
// Skip{,f,Now} methods of *testing.T, must be called from the test goroutine itself.
// This checker detects calls to these functions that occur within a goroutine
// started by the test. For example:
//
//	func TestFoo(t *testing.T) {
//	    go func() {
//	        t.Fatal("oops") // error: (*T).Fatal called from non-test goroutine
//	    }()
//	}
package testinggoroutine
