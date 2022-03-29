// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.7
// +build go1.7

package gc

import (
	"os"
	tracepkg "runtime/trace"

	"cmd/compile/internal/base"
)

func init() {
	traceHandler = traceHandlerGo17
}

func traceHandlerGo17(traceprofile string) {
	f, err := os.Create(traceprofile)
	if err != nil {
		base.Fatalf("%v", err)
	}
	if err := tracepkg.Start(f); err != nil {
		base.Fatalf("%v", err)
	}
	base.AtExit(tracepkg.Stop)
}
