// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Utils


// Environment
export var
	GOARCH,
	GOOS,
	GOROOT,
	USER string;


func GetEnv(key string) string {
	n := len(key);
	for i := 0; i < sys.envc(); i++ {
		v := sys.envv(i);
		if v[0 : n] == key {
			return v[n + 1 : len(v)];  // +1: trim "="
		}
	}
	return "";
}


func init() {
	GOARCH = GetEnv("GOARCH");
	GOOS = GetEnv("GOOS");
	GOROOT = GetEnv("GOROOT");
	USER = GetEnv("USER");
}


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


export func TrimExt(s, ext string) string {
	i := len(s) - len(ext);
	if i >= 0 && s[i : len(s)] == ext {
		s = s[0 : i];
	}
	return s;
}
