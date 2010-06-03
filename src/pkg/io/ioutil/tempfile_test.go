// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ioutil_test

import (
	. "io/ioutil"
	"os"
	"testing"
)

func TestTempFile(t *testing.T) {
	f, err := TempFile("/_not_exists_", "foo")
	if f != nil || err == nil {
		t.Errorf("TempFile(`/_not_exists_`, `foo`) = %v, %v", f, err)
	}

	f, err = TempFile("/tmp", "ioutil_test")
	if f == nil || err != nil {
		t.Errorf("TempFile(`/tmp`, `ioutil_test`) = %v, %v", f, err)
	}
	re := testing.MustCompile("^/tmp/ioutil_test[0-9]+$")
	if !re.MatchString(f.Name()) {
		t.Fatalf("TempFile(`/tmp`, `ioutil_test`) created bad name %s", f.Name())
	}
	os.Remove(f.Name())
	f.Close()
}
