package os_test

import (
	"io/ioutil"
	"os"
	osexec "os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

var supportJunctionLinks = true

func init() {
	tmpdir, err := ioutil.TempDir("", "symtest")
	if err != nil {
		panic("failed to create temp directory: " + err.Error())
	}
	defer os.RemoveAll(tmpdir)

	err = os.Symlink("target", filepath.Join(tmpdir, "symlink"))
	if err != nil {
		err = err.(*os.LinkError).Err
		switch err {
		case syscall.EWINDOWS, syscall.ERROR_PRIVILEGE_NOT_HELD:
			supportsSymlinks = false
		}
	}
	defer os.Remove("target")

	b, _ := osexec.Command("cmd", "/c", "mklink", "/?").Output()
	if !strings.Contains(string(b), " /J ") {
		supportJunctionLinks = false
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

func TestStatJunctionLink(t *testing.T) {
	if !supportJunctionLinks {
		t.Skip("skipping because junction links are not supported")
	}

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	link := filepath.Join(filepath.Dir(dir), filepath.Base(dir)+"-link")

	output, err := osexec.Command("cmd", "/c", "mklink", "/J", link, dir).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run mklink %v %v: %v %q", link, dir, err, output)
	}
	defer os.Remove(link)

	fi, err := os.Stat(link)
	if err != nil {
		t.Fatalf("failed to stat link %v: %v", link, err)
	}
	expected := filepath.Base(dir)
	got := fi.Name()
	if !fi.IsDir() || expected != got {
		t.Fatalf("link should point to %v but points to %v instead", expected, got)
	}
}
