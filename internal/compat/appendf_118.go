// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19

package compat

import "fmt"

func Appendf(b []byte, format string, a ...interface{}) []byte {
	return append(b, fmt.Sprintf(format, a...)...)
}
