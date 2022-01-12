// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"diameter"
)

func main() {
	diameter.NewInboundHandler("hello", "world", "hi")
}
