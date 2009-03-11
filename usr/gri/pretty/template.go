// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"os";
	"io";
)


type Template struct {
	template []byte;
}


func (T *Template) Init(filename string) *os.Error {
	f, err0 := os.Open(filename, os.O_RDONLY, 0);
	defer f.Close();
	if err0 != nil {
		return err0;
	}

	var buf io.ByteBuffer;
	len, err1 := io.Copy(f, &buf);
	if err1 == io.ErrEOF {
		err1 = nil;
	}
	if err1 != nil {
		return err1;
	}

	T.template = buf.Data();

	return nil;
}


// Returns true if buf starts with s, returns false otherwise.

func match(buf []byte, s string) bool {
	if len(buf) < len(s) {
		return false;
	}
	for i := 0; i < len(s); i++ {
		if buf[i] != s[i] {
			return false;
		}
	}
	return true;
}


// Find the position of string s in buf, starting at i.
// Returns a value < 0 if not found.

func find(buf []byte, s string, i int) int {
    if s == "" {
        return i;
    }
L:	for ; i + len(s) <= len(buf); i++ {
		for k := 0; k < len(s); k++ {
			if buf[i+k] != s[k] {
				continue L;
			}
		}
		return i;
    }
    return -1
}


type Substitution map [string] func()

func (T *Template) Apply(w io.Write, prefix string, subs Substitution) *os.Error {
	i0 := 0;  // position from which to write from the template
	i1 := 0;  // position from which to look for the next prefix

	for {
		// look for a prefix
		i2 := find(T.template, prefix, i1);  // position of prefix, if any
		if i2 < 0 {
			// no prefix found, we are done
			break;
		}

		// we have a prefix, look for a matching key
		i1 = i2 + len(prefix);
		for key, action := range subs {
			if match(T.template[i1 : len(T.template)], key) {
				// found a match
				i1 += len(key);  // next search starting pos
				len, err := w.Write(T.template[i0 : i2]);  // TODO handle errors
				i0 = i1;  // skip placeholder
				action();
				break;
			}
		}
	}

	// write the rest of the template
	len, err := w.Write(T.template[i0 : len(T.template)]);  // TODO handle errors
	return err;
}
