// build -race

//go:build race

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var o any = uint64(5)
	switch o.(type) {
	case int:
		goto ret
	case int8:
		goto ret
	case int16:
		goto ret
	case int32:
		goto ret
	case int64:
		goto ret
	case float32:
		goto ret
	case float64:
		goto ret
	default:
		goto ret
	}
ret:
}
