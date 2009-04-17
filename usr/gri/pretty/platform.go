// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Platform

import IO "io"
import OS "os"
import Utils "utils"


// ----------------------------------------------------------------------------
// Environment

var
	GOARCH,
	GOOS,
	GOROOT,
	USER string;

func init() {
	var e OS.Error;

	GOARCH, e = OS.Getenv("GOARCH");
	GOOS, e = OS.Getenv("GOOS");
	GOROOT, e = OS.Getenv("GOROOT");
	USER, e = OS.Getenv("USER");
}


// ----------------------------------------------------------------------------
// I/O

const (
	MAGIC_obj_file = "@gri-go.7@v0";  // make it clear that it cannot be a source file
	Src_file_ext = ".go";
	Obj_file_ext = ".7";
)

func readfile(filename string) ([]byte, OS.Error) {
	f, err := OS.Open(filename, OS.O_RDONLY, 0);
	if err != nil {
		return []byte{}, err;
	}
	var buf [1<<20]byte;
	n, err1 := IO.Readn(f, &buf);
	f.Close();
	if err1 == IO.ErrEOF {
		err1 = nil;
	}
	return buf[0:n], err1;
}

func writefile(name, data string) OS.Error {
	fd, err := OS.Open(name, OS.O_WRONLY, 0);
	if err != nil {
		return err;
	}
	n, err1 := IO.WriteString(fd, data);
	fd.Close();
	return err1;
}

func ReadObjectFile(filename string) ([]byte, bool) {
	data, err := readfile(filename + Obj_file_ext);
	magic := MAGIC_obj_file;  // TODO remove once len(constant) works
	if err == nil && len(data) >= len(magic) && string(data[0 : len(magic)]) == magic {
		return data, true;
	}
	return []byte{}, false;
}


func ReadSourceFile(name string) ([]byte, bool) {
	name = Utils.TrimExt(name, Src_file_ext) + Src_file_ext;
	data, err := readfile(name);
	return data, err == nil;
}


func WriteObjectFile(name string, data string) bool {
	name = Utils.TrimExt(Utils.BaseName(name), Src_file_ext) + Obj_file_ext;
	return writefile(name, data) != nil;
}
