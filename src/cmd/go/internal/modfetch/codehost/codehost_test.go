// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"testing"

	"cmd/go/internal/vcscmd"
)

func TestRun(t *testing.T) {
	dir := t.TempDir()
	bad := vcscmd.New("fossil", "zip").Option("-R", "")
	_, buildErr := bad.Build()
	out, err := run(t.Context(), dir, bad)
	runErr, ok := err.(*RunError)
	if !ok || buildErr == nil || runErr.Err != buildErr || runErr.Cmd != "fossil zip in "+dir || out != nil {
		t.Fatalf("run invalid command = %q, %#v; want build error with partial command", out, err)
	}
}
