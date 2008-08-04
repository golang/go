// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Utils


export func BaseName(s string) string {
	// TODO this is not correct for non-ASCII strings!
	i := len(s) - 1;
	for i >= 0 && s[i] != '/' {
		if s[i] > 128 {
			panic "non-ASCII string"
		}
		i--;
	}
	return s[i + 1 : len(s)];
}


export func FixExt(s string) string {
	i := len(s) - 3;  // 3 == len(".go");
	if i >= 0 && s[i : len(s)] == ".go" {
		s = s[0 : i];
	}
	return s + ".7";
}


export func GetEnv(key string) string {
	n := len(key);
	for i := 0; i < sys.envc(); i++ {
		v := sys.envv(i);
		if v[0 : n] == key {
			return v[n + 1 : len(v)];  // +1: skip "="
		}
	}
	return "";
}
