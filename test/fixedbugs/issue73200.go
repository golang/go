// build

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var g bool

func main() {
	l_4 := uint32(0x6E54EE87)
	v4 := int8(-Int64FromInt64(1))
	g = int32(v4) >= safe_mod_func_int32_t_s_s(BoolInt32(l_4 >= 1), 7)
}

func safe_mod_func_int32_t_s_s(si1 int32, si2 int32) (r int32) {
	var v1 int32
	if si2 == 0 {
		v1 = si1
	} else {
		v1 = si1 % si2
	}
	return v1
}

func Int64FromInt64(n int64) int64 {
	return n
}

func BoolInt32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}
