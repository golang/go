// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"fmt"
	"go/build"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	if os.Getenv("GODOC_TEST_IS_GODOC") != "" {
		main()
		os.Exit(0)
	}

	// Inform subprocesses that they should run the cmd/godoc main instead of
	// running tests. It's a close approximation to building and running the real
	// command, and much less complicated and expensive to build and clean up.
	os.Setenv("GODOC_TEST_IS_GODOC", "1")

	os.Exit(m.Run())
}

var exe struct {
	path string
	err  error
	once sync.Once
}

func godocPath(t *testing.T) string {
	if !testenv.HasExec() {
		t.Skipf("skipping test that requires exec")
	}

	exe.once.Do(func() {
		exe.path, exe.err = os.Executable()
	})
	if exe.err != nil {
		t.Fatal(exe.err)
	}
	return exe.path
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

func waitForServerReady(t *testing.T, ctx context.Context, cmd *exec.Cmd, addr string) {
	waitForServer(t, ctx,
		fmt.Sprintf("http://%v/", addr),
		"Go Documentation Server",
		false)
}

func waitForSearchReady(t *testing.T, ctx context.Context, cmd *exec.Cmd, addr string) {
	waitForServer(t, ctx,
		fmt.Sprintf("http://%v/search?q=FALLTHROUGH", addr),
		"The list of tokens.",
		false)
}

func waitUntilScanComplete(t *testing.T, ctx context.Context, addr string) {
	waitForServer(t, ctx,
		fmt.Sprintf("http://%v/pkg", addr),
		"Scan is not yet complete",
		// setting reverse as true, which means this waits
		// until the string is not returned in the response anymore
		true)
}

const pollInterval = 50 * time.Millisecond

// waitForServer waits for server to meet the required condition,
// failing the test if ctx is canceled before that occurs.
func waitForServer(t *testing.T, ctx context.Context, url, match string, reverse bool) {
	start := time.Now()
	for {
		if ctx.Err() != nil {
			t.Helper()
			t.Fatalf("server failed to respond in %v", time.Since(start))
		}

		time.Sleep(pollInterval)
		res, err := http.Get(url)
		if err != nil {
			continue
		}
		body, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err != nil || res.StatusCode != http.StatusOK {
			continue
		}
		switch {
		case !reverse && bytes.Contains(body, []byte(match)),
			reverse && !bytes.Contains(body, []byte(match)):
			return
		}
	}
}

// hasTag checks whether a given release tag is contained in the current version
// of the go binary.
func hasTag(t string) bool {
	for _, v := range build.Default.ReleaseTags {
		if t == v {
			return true
		}
	}
	return false
}

func TestURL(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; fails to start up quickly enough")
	}
	bin := godocPath(t)

	testcase := func(url string, contents string) func(t *testing.T) {
		return func(t *testing.T) {
			stdout, stderr := new(bytes.Buffer), new(bytes.Buffer)

			args := []string{fmt.Sprintf("-url=%s", url)}
			cmd := testenv.Command(t, bin, args...)
			cmd.Stdout = stdout
			cmd.Stderr = stderr
			cmd.Args[0] = "godoc"

			// Set GOPATH variable to a non-existing absolute path
			// and GOPROXY=off to disable module fetches.
			// We cannot just unset GOPATH variable because godoc would default it to ~/go.
			// (We don't want the indexer looking at the local workspace during tests.)
			cmd.Env = append(os.Environ(),
				"GOPATH=/does_not_exist",
				"GOPROXY=off",
				"GO111MODULE=off")

			if err := cmd.Run(); err != nil {
				t.Fatalf("failed to run godoc -url=%q: %s\nstderr:\n%s", url, err, stderr)
			}

			if !strings.Contains(stdout.String(), contents) {
				t.Errorf("did not find substring %q in output of godoc -url=%q:\n%s", contents, url, stdout)
			}
		}
	}

	t.Run("index", testcase("/", "These packages are part of the Go Project but outside the main Go tree."))
	t.Run("fmt", testcase("/pkg/fmt", "Package fmt implements formatted I/O"))
}

// Basic integration test for godoc HTTP interface.
func TestWeb(t *testing.T) {
	bin := godocPath(t)

	for _, x := range packagestest.All {
		t.Run(x.Name(), func(t *testing.T) {
			testWeb(t, x, bin, false)
		})
	}
}

// Basic integration test for godoc HTTP interface.
func TestWebIndex(t *testing.T) {
	t.Skip("slow test of to-be-deleted code (golang/go#59056)")
	if testing.Short() {
		t.Skip("skipping slow test in -short mode")
	}
	bin := godocPath(t)
	testWeb(t, packagestest.GOPATH, bin, true)
}

// Basic integration test for godoc HTTP interface.
func testWeb(t *testing.T, x packagestest.Exporter, bin string, withIndex bool) {
	switch runtime.GOOS {
	case "plan9":
		t.Skip("skipping on plan9: fails to start up quickly enough")
	case "android", "ios":
		t.Skip("skipping on mobile: lacks GOROOT/api in test environment")
	}

	// Write a fake GOROOT/GOPATH with some third party packages.
	e := packagestest.Export(t, x, []packagestest.Module{
		{
			Name: "godoc.test/repo1",
			Files: map[string]interface{}{
				"a/a.go": `// Package a is a package in godoc.test/repo1.
package a; import _ "godoc.test/repo2/a"; const Name = "repo1a"`,
				"b/b.go": `package b; const Name = "repo1b"`,
			},
		},
		{
			Name: "godoc.test/repo2",
			Files: map[string]interface{}{
				"a/a.go": `package a; const Name = "repo2a"`,
				"b/b.go": `package b; const Name = "repo2b"`,
			},
		},
	})
	defer e.Cleanup()

	// Start the server.
	addr := serverAddress(t)
	args := []string{fmt.Sprintf("-http=%s", addr)}
	if withIndex {
		args = append(args, "-index", "-index_interval=-1s")
	}
	cmd := testenv.Command(t, bin, args...)
	cmd.Dir = e.Config.Dir
	cmd.Env = e.Config.Env
	cmdOut := new(strings.Builder)
	cmd.Stdout = cmdOut
	cmd.Stderr = cmdOut
	cmd.Args[0] = "godoc"

	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start godoc: %s", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		err := cmd.Wait()
		t.Logf("%v: %v", cmd, err)
		cancel()
	}()
	defer func() {
		// Shut down the server cleanly if possible.
		if runtime.GOOS == "windows" {
			cmd.Process.Kill() // Windows doesn't support os.Interrupt.
		} else {
			cmd.Process.Signal(os.Interrupt)
		}
		<-ctx.Done()
		t.Logf("server output:\n%s", cmdOut)
	}()

	if withIndex {
		waitForSearchReady(t, ctx, cmd, addr)
	} else {
		waitForServerReady(t, ctx, cmd, addr)
		waitUntilScanComplete(t, ctx, addr)
	}

	tests := []struct {
		path        string
		contains    []string // substring
		match       []string // regexp
		notContains []string
		needIndex   bool
		releaseTag  string // optional release tag that must be in go/build.ReleaseTags
	}{
		{
			path: "/",
			contains: []string{
				"Go Documentation Server",
				"Standard library",
				"These packages are part of the Go Project but outside the main Go tree.",
			},
		},
		{
			path:     "/pkg/fmt/",
			contains: []string{"Package fmt implements formatted I/O"},
		},
		{
			path:     "/src/fmt/",
			contains: []string{"scan_test.go"},
		},
		{
			path:     "/src/fmt/print.go",
			contains: []string{"// Println formats using"},
		},
		{
			path: "/pkg",
			contains: []string{
				"Standard library",
				"Package fmt implements formatted I/O",
				"Third party",
				"Package a is a package in godoc.test/repo1.",
			},
			notContains: []string{
				"internal/syscall",
				"cmd/gc",
			},
		},
		{
			path: "/pkg/?m=all",
			contains: []string{
				"Standard library",
				"Package fmt implements formatted I/O",
				"internal/syscall/?m=all",
			},
			notContains: []string{
				"cmd/gc",
			},
		},
		{
			path: "/search?q=ListenAndServe",
			contains: []string{
				"/src",
			},
			notContains: []string{
				"/pkg/bootstrap",
			},
			needIndex: true,
		},
		{
			path: "/pkg/strings/",
			contains: []string{
				`href="/src/strings/strings.go"`,
			},
		},
		{
			path: "/cmd/compile/internal/amd64/",
			contains: []string{
				`href="/src/cmd/compile/internal/amd64/ssa.go"`,
			},
		},
		{
			path: "/pkg/math/bits/",
			contains: []string{
				`Added in Go 1.9`,
			},
		},
		{
			path: "/pkg/net/",
			contains: []string{
				`// IPv6 scoped addressing zone; added in Go 1.1`,
			},
		},
		{
			path: "/pkg/net/http/httptrace/",
			match: []string{
				`Got1xxResponse.*// Go 1\.11`,
			},
			releaseTag: "go1.11",
		},
		// Verify we don't add version info to a struct field added the same time
		// as the struct itself:
		{
			path: "/pkg/net/http/httptrace/",
			match: []string{
				`(?m)GotFirstResponseByte func\(\)\s*$`,
			},
		},
		// Remove trailing periods before adding semicolons:
		{
			path: "/pkg/database/sql/",
			contains: []string{
				"The number of connections currently in use; added in Go 1.11",
				"The number of idle connections; added in Go 1.11",
			},
			releaseTag: "go1.11",
		},

		// Third party packages.
		{
			path:     "/pkg/godoc.test/repo1/a",
			contains: []string{`const <span id="Name">Name</span> = &#34;repo1a&#34;`},
		},
		{
			path:     "/pkg/godoc.test/repo2/b",
			contains: []string{`const <span id="Name">Name</span> = &#34;repo2b&#34;`},
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
		strBody := string(body)
		resp.Body.Close()
		if err != nil {
			t.Errorf("GET %s: failed to read body: %s (response: %v)", url, err, resp)
		}
		isErr := false
		for _, substr := range test.contains {
			if test.releaseTag != "" && !hasTag(test.releaseTag) {
				continue
			}
			if !bytes.Contains(body, []byte(substr)) {
				t.Errorf("GET %s: wanted substring %q in body", url, substr)
				isErr = true
			}
		}
		for _, re := range test.match {
			if test.releaseTag != "" && !hasTag(test.releaseTag) {
				continue
			}
			if ok, err := regexp.MatchString(re, strBody); !ok || err != nil {
				if err != nil {
					t.Fatalf("Bad regexp %q: %v", re, err)
				}
				t.Errorf("GET %s: wanted to match %s in body", url, re)
				isErr = true
			}
		}
		for _, substr := range test.notContains {
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

// Test for golang.org/issue/35476.
func TestNoMainModule(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in -short mode")
	}
	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; for consistency with other tests that build godoc binary")
	}
	bin := godocPath(t)
	tempDir := t.TempDir()

	// Run godoc in an empty directory with module mode explicitly on,
	// so that 'go env GOMOD' reports os.DevNull.
	cmd := testenv.Command(t, bin, "-url=/")
	cmd.Dir = tempDir
	cmd.Env = append(os.Environ(), "GO111MODULE=on")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err != nil {
		t.Fatalf("godoc command failed: %v\nstderr=%q", err, stderr.String())
	}
	if strings.Contains(stderr.String(), "go mod download") {
		t.Errorf("stderr contains 'go mod download', is that intentional?\nstderr=%q", stderr.String())
	}
}
