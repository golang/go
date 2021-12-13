// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Helper to print system metadata (CPU model, etc). This uses packages that
// may not be available in the bootstrap toolchain. It only needs to be built
// on the dist build using the final toolchain.

//go:build go1.18
// +build go1.18

package main

import (
	"fmt"
	"internal/sysinfo"
	"runtime"
)

func logMetadata() error {
	fmt.Printf("# GOARCH: %s\n", runtime.GOARCH)
	fmt.Printf("# CPU: %s\n", sysinfo.CPU.Name())
	return nil
}
