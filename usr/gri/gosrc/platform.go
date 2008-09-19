// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Platform

import Utils "utils"


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
		if n < len(v) && v[0 : n] == key && v[n] == '=' {
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


export func ReadSourceFile(name string) (data string, ok bool) {
	name = Utils.TrimExt(name, src_file_ext) + src_file_ext;
	data, ok = sys.readfile(name);
	return data, ok;
}


export func WriteObjectFile(name string, data string) bool {
	name = Utils.TrimExt(Utils.BaseName(name), src_file_ext) + obj_file_ext;
	return sys.writefile(name, data);
}
