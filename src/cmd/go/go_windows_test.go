// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestAbsolutePath(t *testing.T) {
	tmp, err := ioutil.TempDir("", "TestAbsolutePath")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	file := filepath.Join(tmp, "a.go")
	err = ioutil.WriteFile(file, []byte{}, 0644)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(tmp, "dir")
	err = os.Mkdir(dir, 0777)
	if err != nil {
		t.Fatal(err)
	}

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)

	// Chdir so current directory and a.go reside on the same drive.
	err = os.Chdir(dir)
	if err != nil {
		t.Fatal(err)
	}

	noVolume := file[len(filepath.VolumeName(file)):]
	wrongPath := filepath.Join(dir, noVolume)
	output, err := exec.Command(testenv.GoToolPath(t), "build", noVolume).CombinedOutput()
	if err == nil {
		t.Fatal("build should fail")
	}
	if strings.Contains(string(output), wrongPath) {
		t.Fatalf("wrong output found: %v %v", err, string(output))
	}
}

func runIcacls(t *testing.T, args ...string) string {
	t.Helper()
	out, err := exec.Command("icacls", args...).CombinedOutput()
	if err != nil {
		t.Fatalf("icacls failed: %v\n%v", err, string(out))
	}
	return string(out)
}

func runGetACL(t *testing.T, path string) string {
	t.Helper()
	cmd := fmt.Sprintf(`Get-Acl "%s" | Select -expand AccessToString`, path)
	out, err := exec.Command("powershell", "-Command", cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("Get-Acl failed: %v\n%v", err, string(out))
	}
	return string(out)
}

// For issue 22343: verify that executable file created by "go build" command
// has discretionary access control list (DACL) set as if the file
// was created in the destination directory.
func TestACL(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestACL")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	newtmpdir := filepath.Join(tmpdir, "tmp")
	err = os.Mkdir(newtmpdir, 0777)
	if err != nil {
		t.Fatal(err)
	}

	// When TestACL/tmp directory is created, it will have
	// the same security attributes as TestACL.
	// Add Guest account full access to TestACL/tmp - this
	// will make all files created in TestACL/tmp have different
	// security attributes to the files created in TestACL.
	runIcacls(t, newtmpdir,
		"/grant", "*S-1-5-32-546:(oi)(ci)f", // add Guests group to have full access
	)

	src := filepath.Join(tmpdir, "main.go")
	err = ioutil.WriteFile(src, []byte("package main; func main() { }\n"), 0644)
	if err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "main.exe")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", exe, src)
	cmd.Env = append(os.Environ(),
		"TMP="+newtmpdir,
		"TEMP="+newtmpdir,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go command failed: %v\n%v", err, string(out))
	}

	// exe file is expected to have the same security attributes as the src.
	if got, expected := runGetACL(t, exe), runGetACL(t, src); got != expected {
		t.Fatalf("expected Get-Acl output of \n%v\n, got \n%v\n", expected, got)
	}
}
