// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains wrappers so t.Errorf etc. have documentation.
// TODO: delete when godoc shows exported methods for unexported embedded fields.
// TODO: need to change the argument to runtime.Caller in testing.go from 4 to 3 at that point.

package testing

// Fail marks the function as having failed but continues execution.
func (b *B) Fail() {
	b.common.Fail()
}

// Failed returns whether the function has failed.
func (b *B) Failed() bool {
	return b.common.Failed()
}

// FailNow marks the function as having failed and stops its execution.
// Execution will continue at the next Test.
func (b *B) FailNow() {
	b.common.FailNow()
}

// Log formats its arguments using default formatting, analogous to Println(),
// and records the text in the error log.
func (b *B) Log(args ...interface{}) {
	b.common.Log(args...)
}

// Logf formats its arguments according to the format, analogous to Printf(),
// and records the text in the error log.
func (b *B) Logf(format string, args ...interface{}) {
	b.common.Logf(format, args...)
}

// Error is equivalent to Log() followed by Fail().
func (b *B) Error(args ...interface{}) {
	b.common.Error(args...)
}

// Errorf is equivalent to Logf() followed by Fail().
func (b *B) Errorf(format string, args ...interface{}) {
	b.common.Errorf(format, args...)
}

// Fatal is equivalent to Log() followed by FailNow().
func (b *B) Fatal(args ...interface{}) {
	b.common.Fatal(args...)
}

// Fatalf is equivalent to Logf() followed by FailNow().
func (b *B) Fatalf(format string, args ...interface{}) {
	b.common.Fatalf(format, args...)
}

// Fail marks the function as having failed but continues execution.
func (t *T) Fail() {
	t.common.Fail()
}

// Failed returns whether the function has failed.
func (t *T) Failed() bool {
	return t.common.Failed()
}

// FailNow marks the function as having failed and stops its execution.
// Execution will continue at the next Test.
func (t *T) FailNow() {
	t.common.FailNow()
}

// Log formats its arguments using default formatting, analogous to Println(),
// and records the text in the error log.
func (t *T) Log(args ...interface{}) {
	t.common.Log(args...)
}

// Logf formats its arguments according to the format, analogous to Printf(),
// and records the text in the error log.
func (t *T) Logf(format string, args ...interface{}) {
	t.common.Logf(format, args...)
}

// Error is equivalent to Log() followed by Fail().
func (t *T) Error(args ...interface{}) {
	t.common.Error(args...)
}

// Errorf is equivalent to Logf() followed by Fail().
func (t *T) Errorf(format string, args ...interface{}) {
	t.common.Errorf(format, args...)
}

// Fatal is equivalent to Log() followed by FailNow().
func (t *T) Fatal(args ...interface{}) {
	t.common.Fatal(args...)
}

// Fatalf is equivalent to Logf() followed by FailNow().
func (t *T) Fatalf(format string, args ...interface{}) {
	t.common.Fatalf(format, args...)
}
