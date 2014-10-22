package os_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func init() {
	tmpdir, err := ioutil.TempDir("", "symtest")
	if err != nil {
		panic("failed to create temp directory: " + err.Error())
	}
	defer os.RemoveAll(tmpdir)

	err = os.Symlink("target", filepath.Join(tmpdir, "symlink"))
	if err == nil {
		return
	}

	err = err.(*os.LinkError).Err
	switch err {
	case syscall.EWINDOWS, syscall.ERROR_PRIVILEGE_NOT_HELD:
		supportsSymlinks = false
	}
}

func TestSameWindowsFile(t *testing.T) {
	temp, err := ioutil.TempDir("", "TestSameWindowsFile")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(temp)

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	err = os.Chdir(temp)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)

	f, err := os.Create("a")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	ia1, err := os.Stat("a")
	if err != nil {
		t.Fatal(err)
	}

	path, err := filepath.Abs("a")
	if err != nil {
		t.Fatal(err)
	}
	ia2, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(ia1, ia2) {
		t.Errorf("files should be same")
	}

	p := filepath.VolumeName(path) + filepath.Base(path)
	if err != nil {
		t.Fatal(err)
	}
	ia3, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(ia1, ia3) {
		t.Errorf("files should be same")
	}
}
