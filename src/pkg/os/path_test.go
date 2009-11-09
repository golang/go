// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	. "os";
	"testing";
)

func TestMkdirAll(t *testing.T) {
	// Create new dir, in _obj so it will get
	// cleaned up by make if not by us.
	path := "_obj/_TestMkdirAll_/dir/./dir2";
	err := MkdirAll(path, 0777);
	if err != nil {
		t.Fatalf("MkdirAll %q: %s", path, err)
	}

	// Already exists, should succeed.
	err = MkdirAll(path, 0777);
	if err != nil {
		t.Fatalf("MkdirAll %q (second time): %s", path, err)
	}

	// Make file.
	fpath := path+"/file";
	_, err = Open(fpath, O_WRONLY|O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %q: %s", fpath, err)
	}

	// Can't make directory named after file.
	err = MkdirAll(fpath, 0777);
	if err == nil {
		t.Fatalf("MkdirAll %q: no error")
	}
	perr, ok := err.(*PathError);
	if !ok {
		t.Fatalf("MkdirAll %q returned %T, not *PathError", fpath, err)
	}
	if perr.Path != fpath {
		t.Fatalf("MkdirAll %q returned wrong error path: %q not %q", fpath, perr.Path, fpath)
	}

	// Can't make subdirectory of file.
	ffpath := fpath + "/subdir";
	err = MkdirAll(ffpath, 0777);
	if err == nil {
		t.Fatalf("MkdirAll %q: no error")
	}
	perr, ok = err.(*PathError);
	if !ok {
		t.Fatalf("MkdirAll %q returned %T, not *PathError", ffpath, err)
	}
	if perr.Path != fpath {
		t.Fatalf("MkdirAll %q returned wrong error path: %q not %q", ffpath, perr.Path, fpath)
	}

	RemoveAll("_obj/_TestMkdirAll_");
}

func TestRemoveAll(t *testing.T) {
	// Work directory.
	path := "_obj/_TestRemoveAll_";
	fpath := path+"/file";
	dpath := path+"/dir";

	// Make directory with 1 file and remove.
	if err := MkdirAll(path, 0777); err != nil {
		t.Fatalf("MkdirAll %q: %s", path, err)
	}
	fd, err := Open(fpath, O_WRONLY|O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %q: %s", fpath, err)
	}
	fd.Close();
	if err = RemoveAll(path); err != nil {
		t.Fatalf("RemoveAll %q (first): %s", path, err)
	}
	if _, err := Lstat(path); err == nil {
		t.Fatalf("Lstat %q succeeded after RemoveAll (first)", path)
	}

	// Make directory with file and subdirectory and remove.
	if err = MkdirAll(dpath, 0777); err != nil {
		t.Fatalf("MkdirAll %q: %s", dpath, err)
	}
	fd, err = Open(fpath, O_WRONLY|O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %q: %s", fpath, err)
	}
	fd.Close();
	fd, err = Open(dpath+"/file", O_WRONLY|O_CREAT, 0666);
	if err != nil {
		t.Fatalf("create %q: %s", fpath, err)
	}
	fd.Close();
	if err = RemoveAll(path); err != nil {
		t.Fatalf("RemoveAll %q (second): %s", path, err)
	}
	if _, err := Lstat(path); err == nil {
		t.Fatalf("Lstat %q succeeded after RemoveAll (second)", path)
	}

	// Make directory with file and subdirectory and trigger error.
	if err = MkdirAll(dpath, 0777); err != nil {
		t.Fatalf("MkdirAll %q: %s", dpath, err)
	}

	for _, s := range []string{fpath, dpath+"/file1", path+"/zzz"} {
		fd, err = Open(s, O_WRONLY|O_CREAT, 0666);
		if err != nil {
			t.Fatalf("create %q: %s", s, err)
		}
		fd.Close();
	}
	if err = Chmod(dpath, 0); err != nil {
		t.Fatalf("Chmod %q 0: %s", dpath, err)
	}
	if err = RemoveAll(path); err == nil {
		_, err := Lstat(path);
		if err == nil {
			t.Errorf("Can lstat %q after supposed RemoveAll", path)
		}
		t.Fatalf("RemoveAll %q succeeded with chmod 0 subdirectory", path, err);
	}
	perr, ok := err.(*PathError);
	if !ok {
		t.Fatalf("RemoveAll %q returned %T not *PathError", path, err)
	}
	if perr.Path != dpath {
		t.Fatalf("RemoveAll %q failed at %q not %q", path, perr.Path, dpath)
	}
	if err = Chmod(dpath, 0777); err != nil {
		t.Fatalf("Chmod %q 0777: %s", dpath, err)
	}
	for _, s := range []string{fpath, path+"/zzz"} {
		if _, err := Lstat(s); err == nil {
			t.Fatalf("Lstat %q succeeded after partial RemoveAll", s)
		}
	}
	if err = RemoveAll(path); err != nil {
		t.Fatalf("RemoveAll %q after partial RemoveAll: %s", path, err)
	}
	if _, err := Lstat(path); err == nil {
		t.Fatalf("Lstat %q succeeded after RemoveAll (final)", path)
	}
}
