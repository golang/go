// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

var godocTests = []struct {
	args      []string
	matches   []string // regular expressions
	dontmatch []string // regular expressions
}{
	{
		args: []string{"fmt"},
		matches: []string{
			`import "fmt"`,
			`Package fmt implements formatted I/O`,
		},
	},
	{
		args: []string{"io", "WriteString"},
		matches: []string{
			`func WriteString\(`,
			`WriteString writes the contents of the string s to w`,
		},
	},
	{
		args: []string{"nonexistingpkg"},
		matches: []string{
			`no such file or directory|does not exist|cannot find the file`,
		},
	},
	{
		args: []string{"fmt", "NonexistentSymbol"},
		matches: []string{
			`No match found\.`,
		},
	},
	{
		args: []string{"-src", "syscall", "Open"},
		matches: []string{
			`func Open\(`,
		},
		dontmatch: []string{
			`No match found\.`,
		},
	},
}

// buildGodoc builds the godoc executable.
// It returns its path, and a cleanup function.
//
// TODO(adonovan): opt: do this at most once, and do the cleanup
// exactly once.  How though?  There's no atexit.
func buildGodoc(t *testing.T) (bin string, cleanup func()) {
	tmp, err := ioutil.TempDir("", "godoc-regtest-")
	if err != nil {
		t.Fatal(err)
	}

	bin = filepath.Join(tmp, "godoc")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", bin)
	if err := cmd.Run(); err != nil {
		t.Fatalf("Building godoc: %v", err)
	}

	return bin, func() { os.RemoveAll(tmp) }
}

// Basic regression test for godoc command-line tool.
func TestCLI(t *testing.T) {
	bin, cleanup := buildGodoc(t)
	defer cleanup()
	for _, test := range godocTests {
		cmd := exec.Command(bin, test.args...)
		cmd.Args[0] = "godoc"
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("Running with args %#v: %v", test.args, err)
			continue
		}
		for _, pat := range test.matches {
			re := regexp.MustCompile(pat)
			if !re.Match(out) {
				t.Errorf("godoc %v =\n%s\nwanted /%v/", strings.Join(test.args, " "), out, pat)
			}
		}
		for _, pat := range test.dontmatch {
			re := regexp.MustCompile(pat)
			if re.Match(out) {
				t.Errorf("godoc %v =\n%s\ndid not want /%v/", strings.Join(test.args, " "), out, pat)
			}
		}
	}
}

func serverAddress(t *testing.T) string {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		ln, err = net.Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().String()
}

func waitForServer(t *testing.T, address string) {
	// Poll every 50ms for a total of 5s.
	for i := 0; i < 100; i++ {
		time.Sleep(50 * time.Millisecond)
		conn, err := net.Dial("tcp", address)
		if err != nil {
			continue
		}
		conn.Close()
		return
	}
	t.Fatalf("Server %q failed to respond in 5 seconds", address)
}

// Basic integration test for godoc HTTP interface.
func TestWeb(t *testing.T) {
	bin, cleanup := buildGodoc(t)
	defer cleanup()
	addr := serverAddress(t)
	cmd := exec.Command(bin, fmt.Sprintf("-http=%s", addr))
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.Args[0] = "godoc"
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start godoc: %s", err)
	}
	defer cmd.Process.Kill()
	waitForServer(t, addr)
	tests := []struct{ path, substr string }{
		{"/", "Go is an open source programming language"},
		{"/pkg/fmt/", "Package fmt implements formatted I/O"},
		{"/src/pkg/fmt/", "scan_test.go"},
		{"/src/pkg/fmt/print.go", "// Println formats using"},
	}
	for _, test := range tests {
		url := fmt.Sprintf("http://%s%s", addr, test.path)
		resp, err := http.Get(url)
		if err != nil {
			t.Errorf("GET %s failed: %s", url, err)
			continue
		}
		body, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Errorf("GET %s: failed to read body: %s (response: %v)", url, err, resp)
		}
		if bytes.Index(body, []byte(test.substr)) < 0 {
			t.Errorf("GET %s: want substring %q in body, got:\n%s",
				url, test.substr, string(body))
		}
	}
}
