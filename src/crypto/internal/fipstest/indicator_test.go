// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"crypto/internal/fips"
	"testing"
)

func TestIndicator(t *testing.T) {
	fips.ResetServiceIndicator()
	if fips.ServiceIndicator() {
		t.Error("indicator should be false if no calls are made")
	}

	fips.ResetServiceIndicator()
	fips.RecordApproved()
	if !fips.ServiceIndicator() {
		t.Error("indicator should be true if RecordApproved is called")
	}

	fips.ResetServiceIndicator()
	fips.RecordApproved()
	fips.RecordApproved()
	if !fips.ServiceIndicator() {
		t.Error("indicator should be true if RecordApproved is called multiple times")
	}

	fips.ResetServiceIndicator()
	fips.RecordNonApproved()
	if fips.ServiceIndicator() {
		t.Error("indicator should be false if RecordNonApproved is called")
	}

	fips.ResetServiceIndicator()
	fips.RecordApproved()
	fips.RecordNonApproved()
	if fips.ServiceIndicator() {
		t.Error("indicator should be false if both RecordApproved and RecordNonApproved are called")
	}

	fips.ResetServiceIndicator()
	fips.RecordNonApproved()
	fips.RecordApproved()
	if fips.ServiceIndicator() {
		t.Error("indicator should be false if both RecordNonApproved and RecordApproved are called")
	}

	fips.ResetServiceIndicator()
	fips.RecordNonApproved()
	done := make(chan struct{})
	go func() {
		fips.ResetServiceIndicator()
		fips.RecordApproved()
		close(done)
	}()
	<-done
	if fips.ServiceIndicator() {
		t.Error("indicator should be false if RecordApproved is called in a different goroutine")
	}

	fips.ResetServiceIndicator()
	fips.RecordApproved()
	done = make(chan struct{})
	go func() {
		fips.ResetServiceIndicator()
		fips.RecordNonApproved()
		close(done)
	}()
	<-done
	if !fips.ServiceIndicator() {
		t.Error("indicator should be true if RecordNonApproved is called in a different goroutine")
	}
}
