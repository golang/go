// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

func Itob64(i int64, base uint) string {
	if i == 0 {
		return "0"
	}

	u := uint64(i);
	if i < 0 {
		u = -u;
	}

	// Assemble decimal in reverse order.
	var buf [32]byte;
	j := len(buf);
	b := uint64(base);
	for u > 0 {
		j--;
		buf[j] = "0123456789abcdefghijklmnopqrstuvwxyz"[u%b];
		u /= b;
	}

	if i < 0 {	// add sign
		j--;
		buf[j] = '-'
	}

	return string(buf[j:len(buf)])
}


func Itoa64(i int64) string {
	return Itob64(i, 10);
}


func Itob(i int, base uint) string {
	return Itob64(int64(i), base);
}


func Itoa(i int) string {
	return Itob64(int64(i), 10);
}
