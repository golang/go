// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context_test

import (
	"context"
	"net"
	"testing"
)

func TestDeadlineExceededIsNetError(t *testing.T) {
	err, ok := context.DeadlineExceeded.(net.Error)
	if !ok {
		t.Fatal("DeadlineExceeded does not implement net.Error")
	}
	if !err.Timeout() || !err.Temporary() {
		t.Fatalf("Timeout() = %v, Temporary() = %v, want true, true", err.Timeout(), err.Temporary())
	}
}
