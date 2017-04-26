// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
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

// buildGodoc builds the godoc executable.
// It returns its path, and a cleanup function.
//
// TODO(adonovan): opt: do this at most once, and do the cleanup
// exactly once.  How though?  There's no atexit.
func buildGodoc(t *testing.T) (bin string, cleanup func()) {
	if runtime.GOARCH == "arm" {
		t.Skip("skipping test on arm platforms; too slow")
	}
	tmp, err := ioutil.TempDir("", "godoc-regtest-")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if cleanup == nil { // probably, go build failed.
			os.RemoveAll(tmp)
		}
	}()

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

var isGo19 bool // godoc19_test.go sets it to true.

// Basic regression test for godoc command-line tool.
func TestCLI(t *testing.T) {
	bin, cleanup := buildGodoc(t)
	defer cleanup()

	// condStr returns s if cond is true, otherwise empty string.
	condStr := func(cond bool, s string) string {
		if !cond {
			return ""
		}
		return s
	}

	tests := []struct {
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
				`cannot find package` +
					// TODO: Remove this when support for Go 1.8 is dropped.
					condStr(!isGo19,
						// For Go 1.8 and older, because it doesn't have CL 33158 change applied to go/build.
						// The last pattern (does not e) is for plan9:
						// http://build.golang.org/log/2d8e5e14ed365bfa434b37ec0338cd9e6f8dd9bf
						`|no such file or directory|does not exist|cannot find the file|(?:' does not e)`),
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
	for _, test := range tests {
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

func waitForServerReady(t *testing.T, addr string) {
	waitForServer(t,
		fmt.Sprintf("http://%v/", addr),
		"The Go Programming Language",
		15*time.Second)
}

func waitForSearchReady(t *testing.T, addr string) {
	waitForServer(t,
		fmt.Sprintf("http://%v/search?q=FALLTHROUGH", addr),
		"The list of tokens.",
		2*time.Minute)
}

const pollInterval = 200 * time.Millisecond

func waitForServer(t *testing.T, url, match string, timeout time.Duration) {
	// "health check" duplicated from x/tools/cmd/tipgodoc/tip.go
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		time.Sleep(pollInterval)
		res, err := http.Get(url)
		if err != nil {
			continue
		}
		rbody, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err == nil && res.StatusCode == http.StatusOK &&
			bytes.Contains(rbody, []byte(match)) {
			return
		}
	}
	t.Fatalf("Server failed to respond in %v", timeout)
}

func killAndWait(cmd *exec.Cmd) {
	cmd.Process.Kill()
	cmd.Wait()
}

// Basic integration test for godoc HTTP interface.
func TestWeb(t *testing.T) {
	testWeb(t, false)
}

// Basic integration test for godoc HTTP interface.
func TestWebIndex(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in -short mode")
	}
	testWeb(t, true)
}

// Basic integration test for godoc HTTP interface.
func testWeb(t *testing.T, withIndex bool) {
	bin, cleanup := buildGodoc(t)
	defer cleanup()
	addr := serverAddress(t)
	args := []string{fmt.Sprintf("-http=%s", addr)}
	if withIndex {
		args = append(args, "-index", "-index_interval=-1s")
	}
	cmd := exec.Command(bin, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.Args[0] = "godoc"
	cmd.Env = godocEnv()
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start godoc: %s", err)
	}
	defer killAndWait(cmd)

	if withIndex {
		waitForSearchReady(t, addr)
	} else {
		waitForServerReady(t, addr)
	}

	tests := []struct {
		path      string
		match     []string
		dontmatch []string
		needIndex bool
	}{
		{
			path:  "/",
			match: []string{"Go is an open source programming language"},
		},
		{
			path:  "/pkg/fmt/",
			match: []string{"Package fmt implements formatted I/O"},
		},
		{
			path:  "/src/fmt/",
			match: []string{"scan_test.go"},
		},
		{
			path:  "/src/fmt/print.go",
			match: []string{"// Println formats using"},
		},
		{
			path: "/pkg",
			match: []string{
				"Standard library",
				"Package fmt implements formatted I/O",
			},
			dontmatch: []string{
				"internal/syscall",
				"cmd/gc",
			},
		},
		{
			path: "/pkg/?m=all",
			match: []string{
				"Standard library",
				"Package fmt implements formatted I/O",
				"internal/syscall/?m=all",
			},
			dontmatch: []string{
				"cmd/gc",
			},
		},
		{
			path: "/search?q=ListenAndServe",
			match: []string{
				"/src",
			},
			dontmatch: []string{
				"/pkg/bootstrap",
			},
			needIndex: true,
		},
		{
			path: "/pkg/strings/",
			match: []string{
				`href="/src/strings/strings.go"`,
			},
		},
		{
			path: "/cmd/compile/internal/amd64/",
			match: []string{
				`href="/src/cmd/compile/internal/amd64/ssa.go"`,
			},
		},
	}
	for _, test := range tests {
		if test.needIndex && !withIndex {
			continue
		}
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
		isErr := false
		for _, substr := range test.match {
			if !bytes.Contains(body, []byte(substr)) {
				t.Errorf("GET %s: wanted substring %q in body", url, substr)
				isErr = true
			}
		}
		for _, substr := range test.dontmatch {
			if bytes.Contains(body, []byte(substr)) {
				t.Errorf("GET %s: didn't want substring %q in body", url, substr)
				isErr = true
			}
		}
		if isErr {
			t.Errorf("GET %s: got:\n%s", url, body)
		}
	}
}

// Basic integration test for godoc -analysis=type (via HTTP interface).
func TestTypeAnalysis(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test on plan9 (issue #11974)") // see comment re: Plan 9 below
	}

	// Write a fake GOROOT/GOPATH.
	tmpdir, err := ioutil.TempDir("", "godoc-analysis")
	if err != nil {
		t.Fatalf("ioutil.TempDir failed: %s", err)
	}
	defer os.RemoveAll(tmpdir)
	for _, f := range []struct{ file, content string }{
		{"goroot/src/lib/lib.go", `
package lib
type T struct{}
const C = 3
var V T
func (T) F() int { return C }
`},
		{"gopath/src/app/main.go", `
package main
import "lib"
func main() { print(lib.V) }
`},
	} {
		file := filepath.Join(tmpdir, f.file)
		if err := os.MkdirAll(filepath.Dir(file), 0755); err != nil {
			t.Fatalf("MkdirAll(%s) failed: %s", filepath.Dir(file), err)
		}
		if err := ioutil.WriteFile(file, []byte(f.content), 0644); err != nil {
			t.Fatal(err)
		}
	}

	// Start the server.
	bin, cleanup := buildGodoc(t)
	defer cleanup()
	addr := serverAddress(t)
	cmd := exec.Command(bin, fmt.Sprintf("-http=%s", addr), "-analysis=type")
	cmd.Env = append(cmd.Env, fmt.Sprintf("GOROOT=%s", filepath.Join(tmpdir, "goroot")))
	cmd.Env = append(cmd.Env, fmt.Sprintf("GOPATH=%s", filepath.Join(tmpdir, "gopath")))
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "GOROOT=") || strings.HasPrefix(e, "GOPATH=") {
			continue
		}
		cmd.Env = append(cmd.Env, e)
	}
	cmd.Stdout = os.Stderr
	stderr, err := cmd.StderrPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Args[0] = "godoc"
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start godoc: %s", err)
	}
	defer killAndWait(cmd)
	waitForServerReady(t, addr)

	// Wait for type analysis to complete.
	reader := bufio.NewReader(stderr)
	for {
		s, err := reader.ReadString('\n') // on Plan 9 this fails
		if err != nil {
			t.Fatal(err)
		}
		fmt.Fprint(os.Stderr, s)
		if strings.Contains(s, "Type analysis complete.") {
			break
		}
	}
	go io.Copy(os.Stderr, reader)

	t0 := time.Now()

	// Make an HTTP request and check for a regular expression match.
	// The patterns are very crude checks that basic type information
	// has been annotated onto the source view.
tryagain:
	for _, test := range []struct{ url, pattern string }{
		{"/src/lib/lib.go", "L2.*package .*Package docs for lib.*/lib"},
		{"/src/lib/lib.go", "L3.*type .*type info for T.*struct"},
		{"/src/lib/lib.go", "L5.*var V .*type T struct"},
		{"/src/lib/lib.go", "L6.*func .*type T struct.*T.*return .*const C untyped int.*C"},

		{"/src/app/main.go", "L2.*package .*Package docs for app"},
		{"/src/app/main.go", "L3.*import .*Package docs for lib.*lib"},
		{"/src/app/main.go", "L4.*func main.*package lib.*lib.*var lib.V lib.T.*V"},
	} {
		url := fmt.Sprintf("http://%s%s", addr, test.url)
		resp, err := http.Get(url)
		if err != nil {
			t.Errorf("GET %s failed: %s", url, err)
			continue
		}
		body, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Errorf("GET %s: failed to read body: %s (response: %v)", url, err, resp)
			continue
		}

		if !bytes.Contains(body, []byte("Static analysis features")) {
			// Type analysis results usually become available within
			// ~4ms after godoc startup (for this input on my machine).
			if elapsed := time.Since(t0); elapsed > 500*time.Millisecond {
				t.Fatalf("type analysis results still unavailable after %s", elapsed)
			}
			time.Sleep(10 * time.Millisecond)
			goto tryagain
		}

		match, err := regexp.Match(test.pattern, body)
		if err != nil {
			t.Errorf("regexp.Match(%q) failed: %s", test.pattern, err)
			continue
		}
		if !match {
			// This is a really ugly failure message.
			t.Errorf("GET %s: body doesn't match %q, got:\n%s",
				url, test.pattern, string(body))
		}
	}
}

// godocEnv returns the process environment without the GOPATH variable.
// (We don't want the indexer looking at the local workspace during tests.)
func godocEnv() (env []string) {
	for _, v := range os.Environ() {
		if strings.HasPrefix(v, "GOPATH=") {
			continue
		}
		env = append(env, v)
	}
	return
}
