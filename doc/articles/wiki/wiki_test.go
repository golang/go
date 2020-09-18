// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestSnippetsCompile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping slow builds in short mode")
	}

	goFiles, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}

	for _, f := range goFiles {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		f := f
		t.Run(f, func(t *testing.T) {
			t.Parallel()

			cmd := exec.Command("go", "build", "-o", os.DevNull, f)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Errorf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, out)
			}
		})
	}
}

func TestWikiServer(t *testing.T) {
	must := func(err error) {
		if err != nil {
			t.Helper()
			t.Fatal(err)
		}
	}

	dir, err := ioutil.TempDir("", t.Name())
	must(err)
	defer os.RemoveAll(dir)

	// We're testing a walkthrough example of how to write a server.
	//
	// That server hard-codes a port number to make the walkthrough simpler, but
	// we can't assume that the hard-coded port is available on an arbitrary
	// builder. So we'll patch out the hard-coded port, and replace it with a
	// function that writes the server's address to stdout
	// so that we can read it and know where to send the test requests.

	finalGo, err := ioutil.ReadFile("final.go")
	must(err)
	const patchOld = `log.Fatal(http.ListenAndServe(":8080", nil))`
	patched := bytes.ReplaceAll(finalGo, []byte(patchOld), []byte(`log.Fatal(serve())`))
	if bytes.Equal(patched, finalGo) {
		t.Fatalf("Can't patch final.go: %q not found.", patchOld)
	}
	must(ioutil.WriteFile(filepath.Join(dir, "final_patched.go"), patched, 0644))

	// Build the server binary from the patched sources.
	// The 'go' command requires that they all be in the same directory.
	// final_test.go provides the implemtation for our serve function.
	must(copyFile(filepath.Join(dir, "final_srv.go"), "final_test.go"))
	cmd := exec.Command("go", "build",
		"-o", filepath.Join(dir, "final.exe"),
		filepath.Join(dir, "final_patched.go"),
		filepath.Join(dir, "final_srv.go"))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, out)
	}

	// Run the server in our temporary directory so that it can
	// write its content there. It also needs a couple of template files,
	// and looks for them in the same directory.
	must(copyFile(filepath.Join(dir, "edit.html"), "edit.html"))
	must(copyFile(filepath.Join(dir, "view.html"), "view.html"))
	cmd = exec.Command(filepath.Join(dir, "final.exe"))
	cmd.Dir = dir
	stderr := bytes.NewBuffer(nil)
	cmd.Stderr = stderr
	stdout, err := cmd.StdoutPipe()
	must(err)
	must(cmd.Start())

	defer func() {
		cmd.Process.Kill()
		err := cmd.Wait()
		if stderr.Len() > 0 {
			t.Logf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, stderr)
		}
	}()

	var addr string
	if _, err := fmt.Fscanln(stdout, &addr); err != nil || addr == "" {
		t.Fatalf("Failed to read server address: %v", err)
	}

	// The server is up and has told us its address.
	// Make sure that its HTTP API works as described in the article.

	r, err := http.Get(fmt.Sprintf("http://%s/edit/Test", addr))
	must(err)
	responseMustMatchFile(t, r, "test_edit.good")

	r, err = http.Post(fmt.Sprintf("http://%s/save/Test", addr),
		"application/x-www-form-urlencoded",
		strings.NewReader("body=some%20content"))
	must(err)
	responseMustMatchFile(t, r, "test_view.good")

	gotTxt, err := ioutil.ReadFile(filepath.Join(dir, "Test.txt"))
	must(err)
	wantTxt, err := ioutil.ReadFile("test_Test.txt.good")
	must(err)
	if !bytes.Equal(wantTxt, gotTxt) {
		t.Fatalf("Test.txt differs from expected after posting to /save.\ngot:\n%s\nwant:\n%s", gotTxt, wantTxt)
	}

	r, err = http.Get(fmt.Sprintf("http://%s/view/Test", addr))
	must(err)
	responseMustMatchFile(t, r, "test_view.good")
}

func responseMustMatchFile(t *testing.T, r *http.Response, filename string) {
	t.Helper()

	defer r.Body.Close()
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}

	wantBody, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(body, wantBody) {
		t.Fatalf("%v: body does not match %s.\ngot:\n%s\nwant:\n%s", r.Request.URL, filename, body, wantBody)
	}
}

func copyFile(dst, src string) error {
	buf, err := ioutil.ReadFile(src)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(dst, buf, 0644)
}
