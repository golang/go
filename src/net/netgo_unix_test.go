// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!cgo || netgo) && (darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris)

package net

import (
	"context"
	"testing"
)

func TestGoLookupIP(t *testing.T) {
	defer dnsWaitGroup.Wait()
	host := "localhost"
	ctx := context.Background()
	_, err, ok := cgoLookupIP(ctx, "ip", host)
	if ok {
		t.Errorf("cgoLookupIP must be a placeholder")
	}
	if err != nil {
		t.Error(err)
	}
	if _, err := DefaultResolver.goLookupIP(ctx, "ip", host); err != nil {
		t.Error(err)
	}
}
