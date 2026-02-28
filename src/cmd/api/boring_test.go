// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package main

import (
	"fmt"
	"os"
)

func init() {
	fmt.Printf("SKIP with boringcrypto enabled\n")
	os.Exit(0)
}
