// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Metadata prints basic system metadata to include in test logs. This is
// separate from cmd/dist so it does not need to build with the bootstrap
// toolchain.
package main

import (
	"fmt"
	"internal/sysinfo"
	"runtime"
)

func main() {
	fmt.Printf("# GOARCH: %s\n", runtime.GOARCH)
	fmt.Printf("# CPU: %s\n", sysinfo.CPU.Name())
}
