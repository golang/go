// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func Conv(v interface{}) string {
	return conv[string](v)
}

func conv[T any](v interface{}) T {
	return v.(T)
}

func Conv2(v interface{}) (string, bool) {
	return conv2[string](v)
}
func conv2[T any](v interface{}) (T, bool) {
	x, ok := v.(T)
	return x, ok
}
