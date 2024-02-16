// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "crypto/elliptic"

var curve elliptic.Curve

func main() {
	switch curve {
	case elliptic.P224():
	}
}
