// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ioutil_test

import (
	. "io/ioutil"
	"os"
	"regexp"
	"testing"
)

func TestTempFile(t *testing.T) {
	f, err := TempFile("/_not_exists_", "foo")
	if f != nil || err == nil {
		t.Errorf("TempFile(`/_not_exists_`, `foo`) = %v, %v", f, err)
	}

	dir := os.TempDir()
	f, err = TempFile(dir, "ioutil_test")
	if f == nil || err != nil {
		t.Errorf("TempFile(dir, `ioutil_test`) = %v, %v", f, err)
	}
	if f != nil {
		re := regexp.MustCompile("^" + regexp.QuoteMeta(dir) + "/ioutil_test[0-9]+$")
		if !re.MatchString(f.Name()) {
			t.Errorf("TempFile(`"+dir+"`, `ioutil_test`) created bad name %s", f.Name())
		}
		os.Remove(f.Name())
	}
	f.Close()
}
