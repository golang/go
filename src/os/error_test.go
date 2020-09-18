// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestErrIsExist(t *testing.T) {
	f, err := ioutil.TempFile("", "_Go_ErrIsExist")
	if err != nil {
		t.Fatalf("open ErrIsExist tempfile: %s", err)
		return
	}
	defer os.Remove(f.Name())
	defer f.Close()
	f2, err := os.OpenFile(f.Name(), os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
	if err == nil {
		f2.Close()
		t.Fatal("Open should have failed")
		return
	}
	if s := checkErrorPredicate("os.IsExist", os.IsExist, err, os.ErrExist); s != "" {
		t.Fatal(s)
		return
	}
}

func testErrNotExist(name string) string {
	f, err := os.Open(name)
	if err == nil {
		f.Close()
		return "Open should have failed"
	}
	if s := checkErrorPredicate("os.IsNotExist", os.IsNotExist, err, os.ErrNotExist); s != "" {
		return s
	}

	err = os.Chdir(name)
	if err == nil {
		return "Chdir should have failed"
	}
	if s := checkErrorPredicate("os.IsNotExist", os.IsNotExist, err, os.ErrNotExist); s != "" {
		return s
	}
	return ""
}

func TestErrIsNotExist(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "_Go_ErrIsNotExist")
	if err != nil {
		t.Fatalf("create ErrIsNotExist tempdir: %s", err)
		return
	}
	defer os.RemoveAll(tmpDir)

	name := filepath.Join(tmpDir, "NotExists")
	if s := testErrNotExist(name); s != "" {
		t.Fatal(s)
		return
	}

	name = filepath.Join(name, "NotExists2")
	if s := testErrNotExist(name); s != "" {
		t.Fatal(s)
		return
	}
}

func checkErrorPredicate(predName string, pred func(error) bool, err, target error) string {
	if !pred(err) {
		return fmt.Sprintf("%s does not work as expected for %#v", predName, err)
	}
	if !errors.Is(err, target) {
		return fmt.Sprintf("errors.Is(%#v, %#v) = false, want true", err, target)
	}
	return ""
}

type isExistTest struct {
	err   error
	is    bool
	isnot bool
}

var isExistTests = []isExistTest{
	{&os.PathError{Err: os.ErrInvalid}, false, false},
	{&os.PathError{Err: os.ErrPermission}, false, false},
	{&os.PathError{Err: os.ErrExist}, true, false},
	{&os.PathError{Err: os.ErrNotExist}, false, true},
	{&os.PathError{Err: os.ErrClosed}, false, false},
	{&os.LinkError{Err: os.ErrInvalid}, false, false},
	{&os.LinkError{Err: os.ErrPermission}, false, false},
	{&os.LinkError{Err: os.ErrExist}, true, false},
	{&os.LinkError{Err: os.ErrNotExist}, false, true},
	{&os.LinkError{Err: os.ErrClosed}, false, false},
	{&os.SyscallError{Err: os.ErrNotExist}, false, true},
	{&os.SyscallError{Err: os.ErrExist}, true, false},
	{nil, false, false},
}

func TestIsExist(t *testing.T) {
	for _, tt := range isExistTests {
		if is := os.IsExist(tt.err); is != tt.is {
			t.Errorf("os.IsExist(%T %v) = %v, want %v", tt.err, tt.err, is, tt.is)
		}
		if is := errors.Is(tt.err, os.ErrExist); is != tt.is {
			t.Errorf("errors.Is(%T %v, os.ErrExist) = %v, want %v", tt.err, tt.err, is, tt.is)
		}
		if isnot := os.IsNotExist(tt.err); isnot != tt.isnot {
			t.Errorf("os.IsNotExist(%T %v) = %v, want %v", tt.err, tt.err, isnot, tt.isnot)
		}
		if isnot := errors.Is(tt.err, os.ErrNotExist); isnot != tt.isnot {
			t.Errorf("errors.Is(%T %v, os.ErrNotExist) = %v, want %v", tt.err, tt.err, isnot, tt.isnot)
		}
	}
}

type isPermissionTest struct {
	err  error
	want bool
}

var isPermissionTests = []isPermissionTest{
	{nil, false},
	{&os.PathError{Err: os.ErrPermission}, true},
	{&os.SyscallError{Err: os.ErrPermission}, true},
}

func TestIsPermission(t *testing.T) {
	for _, tt := range isPermissionTests {
		if got := os.IsPermission(tt.err); got != tt.want {
			t.Errorf("os.IsPermission(%#v) = %v; want %v", tt.err, got, tt.want)
		}
		if got := errors.Is(tt.err, os.ErrPermission); got != tt.want {
			t.Errorf("errors.Is(%#v, os.ErrPermission) = %v; want %v", tt.err, got, tt.want)
		}
	}
}

func TestErrPathNUL(t *testing.T) {
	f, err := ioutil.TempFile("", "_Go_ErrPathNUL\x00")
	if err == nil {
		f.Close()
		t.Fatal("TempFile should have failed")
	}
	f, err = ioutil.TempFile("", "_Go_ErrPathNUL")
	if err != nil {
		t.Fatalf("open ErrPathNUL tempfile: %s", err)
	}
	defer os.Remove(f.Name())
	defer f.Close()
	f2, err := os.OpenFile(f.Name(), os.O_RDWR, 0600)
	if err != nil {
		t.Fatalf("open ErrPathNUL: %s", err)
	}
	f2.Close()
	f2, err = os.OpenFile(f.Name()+"\x00", os.O_RDWR, 0600)
	if err == nil {
		f2.Close()
		t.Fatal("Open should have failed")
	}
}

func TestPathErrorUnwrap(t *testing.T) {
	pe := &os.PathError{Err: os.ErrInvalid}
	if !errors.Is(pe, os.ErrInvalid) {
		t.Error("errors.Is failed, wanted success")
	}
}

type myErrorIs struct{ error }

func (e myErrorIs) Is(target error) bool { return target == e.error }

func TestErrorIsMethods(t *testing.T) {
	if os.IsPermission(myErrorIs{os.ErrPermission}) {
		t.Error("os.IsPermission(err) = true when err.Is(os.ErrPermission), wanted false")
	}
}
