// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Utility functions.

package io

import (
	"bytes";
	"os";
	"sort";
)

// ReadAll reads from r until an error or EOF and returns the data it read.
func ReadAll(r Reader) ([]byte, os.Error) {
	var buf bytes.Buffer;
	_, err := Copy(&buf, r);
	return buf.Bytes(), err;
}

// ReadFile reads the file named by filename and returns the contents.
func ReadFile(filename string) ([]byte, os.Error) {
	f, err := os.Open(filename, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}
	defer f.Close();
	return ReadAll(f);
}

// WriteFile writes data to a file named by filename.
// If the file does not exist, WriteFile creates it with permissions perm;
// otherwise WriteFile truncates it before writing.
func WriteFile(filename string, data []byte, perm int) os.Error {
	f, err := os.Open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, perm);
	if err != nil {
		return err;
	}
	n, err := f.Write(data);
	f.Close();
	if err == nil && n < len(data) {
		err = ErrShortWrite;
	}
	return err;
}

// A dirList implements sort.Interface.
type dirList []*os.Dir

func (d dirList) Len() int		{ return len(d) }
func (d dirList) Less(i, j int) bool	{ return d[i].Name < d[j].Name }
func (d dirList) Swap(i, j int)		{ d[i], d[j] = d[j], d[i] }

// ReadDir reads the directory named by dirname and returns
// a list of sorted directory entries.
func ReadDir(dirname string) ([]*os.Dir, os.Error) {
	f, err := os.Open(dirname, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}
	list, err := f.Readdir(-1);
	f.Close();
	if err != nil {
		return nil, err;
	}
	dirs := make(dirList, len(list));
	for i := range list {
		dirs[i] = &list[i];
	}
	sort.Sort(dirs);
	return dirs, nil;
}
