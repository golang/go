// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ctfe

import (
	"testing"
)

func TestStepChargeScalesWithLiveBytes(t *testing.T) {
	cfg := Config{MaxSteps: 100, MaxValueBytes: 1 << 20, WordSize: 8}
	src := "package p\nfunc CopyLoop() int { var a [4096]int64\nfor i := 0; i < 10; i++ { a = a }\nreturn len(a) }"
	_, _, err := evalConstFunc(t, cfg, constSpec{
		src:    src,
		rtypes: []*Type{BasicType("int")},
		name:   "CopyLoop",
		call:   "CopyLoop()",
	})
	if err == nil {
		t.Fatalf("CopyLoop succeeded, want step limit")
	}
	cerr, ok := err.(*Error)
	if !ok {
		t.Fatalf("want *Error, got %T (%v)", err, err)
	}
	if cerr.Code != CodeStepLimit {
		t.Fatalf("code = %d, want %d", cerr.Code, CodeStepLimit)
	}
}
