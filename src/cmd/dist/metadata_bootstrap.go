// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No-op metadata implementation when building with an old bootstrap toolchain.

//go:build !go1.18
// +build !go1.18

package main

import (
	"fmt"
)

func logMetadata() error {
	// We don't return an error so we don't completely preclude running
	// tests with a bootstrap dist.
	fmt.Printf("# Metadata unavailable: bootstrap build\n")
	return nil
}
