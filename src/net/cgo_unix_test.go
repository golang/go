// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !netgo && ((cgo && unix) || darwin)

package net

import (
	"context"
	"errors"
	"runtime"
	"testing"
)

func TestCgoLookupIP(t *testing.T) {
	defer dnsWaitGroup.Wait()
	ctx := context.Background()
	_, err := cgoLookupIP(ctx, "ip", "localhost")
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupIPWithCancel(t *testing.T) {
	defer dnsWaitGroup.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err := cgoLookupIP(ctx, "ip", "localhost")
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupPort(t *testing.T) {
	defer dnsWaitGroup.Wait()
	ctx := context.Background()
	_, err := cgoLookupPort(ctx, "tcp", "smtp")
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupPortWithCancel(t *testing.T) {
	defer dnsWaitGroup.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err := cgoLookupPort(ctx, "tcp", "smtp")
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupPTR(t *testing.T) {
	defer dnsWaitGroup.Wait()
	ctx := context.Background()
	_, err := cgoLookupPTR(ctx, "127.0.0.1")
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupPTRWithCancel(t *testing.T) {
	defer dnsWaitGroup.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err := cgoLookupPTR(ctx, "127.0.0.1")
	if err != nil {
		t.Error(err)
	}
}

func TestCgoLookupCNAMEHErrno(t *testing.T) {
	defer dnsWaitGroup.Wait()
	_, err := cgoLookupCNAME(context.Background(), "invalid.invalid")

	if runtime.GOOS == "openbsd" || runtime.GOOS == "android" {
		if err != errCgoDNSLookupFailed {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}

	var dnsErr *DNSError
	errors.As(err, &dnsErr)
	if dnsErr == nil || dnsErr.Err != errNoSuchHost.Error() || !dnsErr.IsNotFound {
		t.Fatalf("unexpected error: %v", err)
	}
}
