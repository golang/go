// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Utils


func BaseName(s string) string {
	// TODO this is not correct for non-ASCII strings!
	i := len(s) - 1;
	for i >= 0 && s[i] != '/' {
		if s[i] > 128 {
			panic("non-ASCII string");
		}
		i--;
	}
	return s[i + 1 : len(s)];
}


func Contains(s, sub string, pos int) bool {
	end := pos + len(sub);
	return pos >= 0 && end <= len(s) && s[pos : end] == sub;
}


func TrimExt(s, ext string) string {
	i := len(s) - len(ext);
	if i >= 0 && s[i : len(s)] == ext {
		s = s[0 : i];
	}
	return s;
}


func IntToString(x, base int) string {
	x0 := x;
	if x < 0 {
		x = -x;
		if x < 0 {
			panic("smallest int not handled");
		}
	} else if x == 0 {
		return "0";
	}

	// x > 0
	hex := "0123456789ABCDEF";
	var buf [32] byte;
	i := len(buf);
	for x > 0 {
		i--;
		buf[i] = hex[x % base];
		x /= base;
	}

	if x0 < 0 {
		i--;
		buf[i] = '-';
	}

	return string(buf)[i : len(buf)];
}
