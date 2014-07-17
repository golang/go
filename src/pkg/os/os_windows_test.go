package os_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
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
