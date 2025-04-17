// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Metadata prints basic system metadata to include in test logs. This is
// separate from cmd/dist so it does not need to build with the bootstrap
// toolchain.

// This program is only used by cmd/dist. Add an "ignore" build tag so it
// is not installed. cmd/dist does "go run main.go" directly.

//go:build ignore

package main

import (
	"cmd/internal/osinfo"
	"fmt"
	"internal/sysinfo"
	"runtime"
)

func main() {
	fmt.Printf("# GOARCH: %s\n", runtime.GOARCH)
	fmt.Printf("# CPU: %s\n", sysinfo.CPUName())

	fmt.Printf("# GOOS: %s\n", runtime.GOOS)
	ver, err := osinfo.Version()
	if err != nil {
		ver = fmt.Sprintf("UNKNOWN: error determining OS version: %v", err)
	}
	fmt.Printf("# OS Version: %s\n", ver)
}
