// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

func main() {
	_ = p.S{1} // ERROR "too few values.*unexported fields"
}
