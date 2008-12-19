// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

export func itoa64(i int64) string {
	if i == 0 {
		return "0"
	}

	neg := false;	// negative
	u := uint64(i);
	if i < 0 {
		neg = true;
		u = -u;
	}

	// Assemble decimal in reverse order.
	var b [32]byte;
	bp := len(b);
	for ; u > 0; u /= 10 {
		bp--;
		b[bp] = byte(u%10) + '0'
	}
	if neg {	// add sign
		bp--;
		b[bp] = '-'
	}

	return string(b[bp:len(b)])
	//return string((&b)[bp:len(b)])
}

export func itoa(i int) string {
	return itoa64(int64(i));
}

