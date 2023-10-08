// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/internal/diffp"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

func init() {
	if os.Getenv("TestGonewMain") == "1" {
		main()
		os.Exit(0)
	}
}

func Test(t *testing.T) {
	if !testenv.HasExec() {
		t.Skipf("skipping test: exec not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	// Each file in testdata is a txtar file with the command to run,
	// the contents of modules to initialize in a fake proxy,
	// the expected stdout and stderr, and the expected file contents.
	files, err := filepath.Glob("testdata/*.txt")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no test cases")
	}

	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			data, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			ar := txtar.Parse(data)

			// If the command begins with ! it means it should fail.
			// After the optional ! the first argument must be 'gonew'
			// followed by the arguments to gonew.
			args := strings.Fields(string(ar.Comment))
			wantFail := false
			if len(args) > 0 && args[0] == "!" {
				wantFail = true
				args = args[1:]
			}
			if len(args) == 0 || args[0] != "gonew" {
				t.Fatalf("invalid command comment")
			}

			// Collect modules into proxy tree and store in temp directory.
			dir := t.TempDir()
			proxyDir := filepath.Join(dir, "proxy")
			writeProxyFiles(t, proxyDir, ar)
			extra := ""
			if runtime.GOOS == "windows" {
				// Windows absolute paths don't start with / so we need one more.
				extra = "/"
			}
			proxyURL := "file://" + extra + filepath.ToSlash(proxyDir)

			// Run gonew in a fresh 'out' directory.
			out := filepath.Join(dir, "out")
			if err := os.Mkdir(out, 0777); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command(exe, args[1:]...)
			cmd.Dir = out
			cmd.Env = append(os.Environ(), "TestGonewMain=1", "GOPROXY="+proxyURL, "GOSUMDB=off")
			var stdout bytes.Buffer
			var stderr bytes.Buffer
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr
			if err := cmd.Run(); err == nil && wantFail {
				t.Errorf("unexpected success exit")
			} else if err != nil && !wantFail {
				t.Errorf("unexpected failure exit")
			}

			// Collect the expected output from the txtar.
			want := make(map[string]txtar.File)
			for _, f := range ar.Files {
				if f.Name == "stdout" || f.Name == "stderr" || strings.HasPrefix(f.Name, "out/") {
					want[f.Name] = f
				}
			}

			// Check stdout and stderr.
			// Change \ to / so Windows output looks like Unix output.
			stdoutBuf := bytes.ReplaceAll(stdout.Bytes(), []byte(`\`), []byte("/"))
			stderrBuf := bytes.ReplaceAll(stderr.Bytes(), []byte(`\`), []byte("/"))
			// Note that stdout and stderr can be omitted from the archive if empty.
			if !bytes.Equal(stdoutBuf, want["stdout"].Data) {
				t.Errorf("wrong stdout: %s", diffp.Diff("want", want["stdout"].Data, "have", stdoutBuf))
			}
			if !bytes.Equal(stderrBuf, want["stderr"].Data) {
				t.Errorf("wrong stderr: %s", diffp.Diff("want", want["stderr"].Data, "have", stderrBuf))
			}
			delete(want, "stdout")
			delete(want, "stderr")

			// Check remaining expected outputs.
			err = filepath.WalkDir(out, func(name string, info fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if info.IsDir() {
					return nil
				}
				data, err := os.ReadFile(name)
				if err != nil {
					return err
				}
				short := "out" + filepath.ToSlash(strings.TrimPrefix(name, out))
				f, ok := want[short]
				if !ok {
					t.Errorf("unexpected file %s:\n%s", short, data)
					return nil
				}
				delete(want, short)
				if !bytes.Equal(data, f.Data) {
					t.Errorf("wrong %s: %s", short, diffp.Diff("want", f.Data, "have", data))
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			for name := range want {
				t.Errorf("missing file %s", name)
			}
		})
	}
}

// A Zip is a zip file being written.
type Zip struct {
	buf bytes.Buffer
	w   *zip.Writer
}

// writeProxyFiles collects all the module content from ar and writes
// files in the format of the proxy URL space, so that the 'proxy' directory
// can be used in a GOPROXY=file:/// URL.
func writeProxyFiles(t *testing.T, proxy string, ar *txtar.Archive) {
	zips := make(map[string]*Zip)
	others := make(map[string]string)
	for _, f := range ar.Files {
		i := strings.Index(f.Name, "@")
		if i < 0 {
			continue
		}
		j := strings.Index(f.Name[i:], "/")
		if j < 0 {
			t.Fatalf("unexpected archive file %s", f.Name)
		}
		j += i
		mod, vers, file := f.Name[:i], f.Name[i+1:j], f.Name[j+1:]
		zipName := mod + "/@v/" + vers + ".zip"
		z := zips[zipName]
		if z == nil {
			others[mod+"/@v/list"] += vers + "\n"
			others[mod+"/@v/"+vers+".info"] = fmt.Sprintf("{%q: %q}\n", "Version", vers)
			z = new(Zip)
			z.w = zip.NewWriter(&z.buf)
			zips[zipName] = z
		}
		if file == "go.mod" {
			others[mod+"/@v/"+vers+".mod"] = string(f.Data)
		}
		w, err := z.w.Create(f.Name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(f.Data); err != nil {
			t.Fatal(err)
		}
	}

	for name, z := range zips {
		if err := z.w.Close(); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(filepath.Join(proxy, name)), 0777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(proxy, name), z.buf.Bytes(), 0666); err != nil {
			t.Fatal(err)
		}
	}
	for name, data := range others {
		// zip loop already created directory
		if err := os.WriteFile(filepath.Join(proxy, name), []byte(data), 0666); err != nil {
			t.Fatal(err)
		}
	}
}
