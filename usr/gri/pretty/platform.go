// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Platform

import IO "io"
import OS "os"
import Utils "utils"


// ----------------------------------------------------------------------------
// Environment

export var
	GOARCH,
	GOOS,
	GOROOT,
	USER string;

func init() {
	var e *OS.Error;

	GOARCH, e = OS.Getenv("GOARCH");
	GOOS, e = OS.Getenv("GOOS");
	GOROOT, e = OS.Getenv("GOROOT");
	USER, e = OS.Getenv("USER");
}


// ----------------------------------------------------------------------------
// I/O

export const (
	MAGIC_obj_file = "@gri-go.7@v0";  // make it clear that it cannot be a source file
	Src_file_ext = ".go";
	Obj_file_ext = ".7";
)

func readfile(filename string) (string, *OS.Error) {
	fd, err := OS.Open(filename, OS.O_RDONLY, 0);
	if err != nil {
		return "", err;
	}
	var buf [1<<20]byte;
	n, err1 := IO.Readn(fd, buf);
	fd.Close();
	if err1 == IO.ErrEOF {
		err1 = nil;
	}
	return string(buf[0:n]), err1;
}

func writefile(name, data string) *OS.Error {
	fd, err := OS.Open(name, OS.O_WRONLY, 0);
	if err != nil {
		return err;
	}
	n, err1 := IO.WriteString(fd, data);
	fd.Close();
	return err1;
}

export func ReadObjectFile(filename string) (string, bool) {
	data, err := readfile(filename + Obj_file_ext);
	magic := MAGIC_obj_file;  // TODO remove once len(constant) works
	if err == nil && len(data) >= len(magic) && data[0 : len(magic)] == magic {
		return data, true;
	}
	return "", false;
}


export func ReadSourceFile(name string) (string, bool) {
	name = Utils.TrimExt(name, Src_file_ext) + Src_file_ext;
	data, err := readfile(name);
	return data, err == nil;
}


export func WriteObjectFile(name string, data string) bool {
	name = Utils.TrimExt(Utils.BaseName(name), Src_file_ext) + Obj_file_ext;
	return writefile(name, data) != nil;
}
