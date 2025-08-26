// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"plugin"
)

func init() {
	_, err := plugin.Open("issue75102.so")
	if err == nil {
		panic("unexpected success to open a different version plugin")
	}
}

func main() {
	fmt.Println("done")
}
