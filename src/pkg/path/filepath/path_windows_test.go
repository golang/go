package filepath_test

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

func TestWinSplitListTestsAreValid(t *testing.T) {
	comspec := os.Getenv("ComSpec")
	if comspec == "" {
		t.Fatal("%ComSpec% must be set")
	}

	for ti, tt := range winsplitlisttests {
		testWinSplitListTestIsValid(t, ti, tt, comspec)
	}
}

func testWinSplitListTestIsValid(t *testing.T, ti int, tt SplitListTest,
	comspec string) {

	const (
		cmdfile             = `printdir.cmd`
		perm    os.FileMode = 0700
	)

	tmp, err := ioutil.TempDir("", "testWinSplitListTestIsValid")
	if err != nil {
		t.Fatalf("TempDir failed: %v", err)
	}
	defer os.RemoveAll(tmp)

	for i, d := range tt.result {
		if d == "" {
			continue
		}
		if cd := filepath.Clean(d); filepath.VolumeName(cd) != "" ||
			cd[0] == '\\' || cd == ".." || (len(cd) >= 3 && cd[0:3] == `..\`) {
			t.Errorf("%d,%d: %#q refers outside working directory", ti, i, d)
			return
		}
		dd := filepath.Join(tmp, d)
		if _, err := os.Stat(dd); err == nil {
			t.Errorf("%d,%d: %#q already exists", ti, i, d)
			return
		}
		if err = os.MkdirAll(dd, perm); err != nil {
			t.Errorf("%d,%d: MkdirAll(%#q) failed: %v", ti, i, dd, err)
			return
		}
		fn, data := filepath.Join(dd, cmdfile), []byte("@echo "+d+"\r\n")
		if err = ioutil.WriteFile(fn, data, perm); err != nil {
			t.Errorf("%d,%d: WriteFile(%#q) failed: %v", ti, i, fn, err)
			return
		}
	}

	for i, d := range tt.result {
		if d == "" {
			continue
		}
		exp := []byte(d + "\r\n")
		cmd := &exec.Cmd{
			Path: comspec,
			Args: []string{`/c`, cmdfile},
			Env:  []string{`Path=` + tt.list},
			Dir:  tmp,
		}
		out, err := cmd.CombinedOutput()
		switch {
		case err != nil:
			t.Errorf("%d,%d: execution error %v\n%q", ti, i, err, out)
			return
		case !reflect.DeepEqual(out, exp):
			t.Errorf("%d,%d: expected %#q, got %#q", ti, i, exp, out)
			return
		default:
			// unshadow cmdfile in next directory
			err = os.Remove(filepath.Join(tmp, d, cmdfile))
			if err != nil {
				t.Fatalf("Remove test command failed: %v", err)
			}
		}
	}
}
