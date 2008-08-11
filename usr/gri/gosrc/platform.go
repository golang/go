// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Platform

// ----------------------------------------------------------------------------
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


// ----------------------------------------------------------------------------
// I/O

export const (
	MAGIC_obj_file = "@gri-go.7@v0";  // make it clear thar it cannot be a source file
	src_file_ext = ".go";
	obj_file_ext = ".7";
)


export func ReadObjectFile(filename string) (data string, ok bool) {
	data, ok = sys.readfile(filename + obj_file_ext);
	magic := MAGIC_obj_file;  // TODO remove once len(constant) works
	if ok && len(data) >= len(magic) && data[0 : len(magic)] == magic {
		return data, ok;
	}
	return "", false;
}


export func ReadSourceFile(filename string) (data string, ok bool) {
	data, ok = sys.readfile(filename + src_file_ext);
	return data, ok;
}
