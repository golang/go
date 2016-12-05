// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"reflect"
	"testing"
)

func TestRemoveDevNull(t *testing.T) {
	fi, err := os.Lstat(os.DevNull)
	if err != nil {
		t.Skip(err)
	}
	if fi.Mode().IsRegular() {
		t.Errorf("Lstat(%s).Mode().IsRegular() = true; expected false", os.DevNull)
	}
	mayberemovefile(os.DevNull)
	_, err = os.Lstat(os.DevNull)
	if err != nil {
		t.Errorf("mayberemovefile(%s) did remove it; oops", os.DevNull)
	}
}

func TestSplitPkgConfigOutput(t *testing.T) {
	for _, test := range []struct {
		in   []byte
		want []string
	}{
		{[]byte(`-r:foo -L/usr/white\ space/lib -lfoo\ bar -lbar\ baz`), []string{"-r:foo", "-L/usr/white space/lib", "-lfoo bar", "-lbar baz"}},
		{[]byte(`-lextra\ fun\ arg\\`), []string{`-lextra fun arg\`}},
		{[]byte(`broken flag\`), []string{"broken", "flag"}},
		{[]byte("\textra     whitespace\r\n"), []string{"extra", "whitespace"}},
		{[]byte("     \r\n      "), nil},
	} {
		got := splitPkgConfigOutput(test.in)
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("splitPkgConfigOutput(%v) = %v; want %v", test.in, got, test.want)
		}
	}
}
