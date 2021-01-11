// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// authtest is a diagnostic tool for implementations of the GOAUTH protocol
// described in https://golang.org/issue/26232.
//
// It accepts a single URL as an argument, and executes the GOAUTH protocol to
// fetch and display the headers for that URL.
//
// CAUTION: authtest logs the GOAUTH responses, which may include user
// credentials, to stderr. Do not post its output unless you are certain that
// all of the credentials involved are fake!
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	exec "golang.org/x/sys/execabs"
	"io"
	"log"
	"net/http"
	"net/textproto"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

var v = flag.Bool("v", false, "if true, log GOAUTH responses to stderr")

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	flag.Parse()
	args := flag.Args()
	if len(args) != 1 {
		log.Fatalf("usage: [GOAUTH=CMD...] %s URL", filepath.Base(os.Args[0]))
	}

	resp := try(args[0], nil)
	if resp.StatusCode == http.StatusOK {
		return
	}

	resp = try(args[0], resp)
	if resp.StatusCode != http.StatusOK {
		os.Exit(1)
	}
}

func try(url string, prev *http.Response) *http.Response {
	req := new(http.Request)
	if prev != nil {
		*req = *prev.Request
	} else {
		var err error
		req, err = http.NewRequest("HEAD", os.Args[1], nil)
		if err != nil {
			log.Fatal(err)
		}
	}

goauth:
	for _, argList := range strings.Split(os.Getenv("GOAUTH"), ";") {
		// TODO(golang.org/issue/26849): If we escape quoted strings in GOFLAGS, use
		// the same quoting here.
		args := strings.Split(argList, " ")
		if len(args) == 0 || args[0] == "" {
			log.Fatalf("invalid or empty command in GOAUTH")
		}

		creds, err := getCreds(args, prev)
		if err != nil {
			log.Fatal(err)
		}
		for _, c := range creds {
			if c.Apply(req) {
				fmt.Fprintf(os.Stderr, "# request to %s\n", req.URL)
				fmt.Fprintf(os.Stderr, "%s %s %s\n", req.Method, req.URL, req.Proto)
				req.Header.Write(os.Stderr)
				fmt.Fprintln(os.Stderr)
				break goauth
			}
		}
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode < 400 || resp.StatusCode > 500 {
		log.Fatalf("unexpected status: %v", resp.Status)
	}

	fmt.Fprintf(os.Stderr, "# response from %s\n", resp.Request.URL)
	formatHead(os.Stderr, resp)
	return resp
}

func formatHead(out io.Writer, resp *http.Response) {
	fmt.Fprintf(out, "%s %s\n", resp.Proto, resp.Status)
	if err := resp.Header.Write(out); err != nil {
		log.Fatal(err)
	}
	fmt.Fprintln(out)
}

type Cred struct {
	URLPrefixes []*url.URL
	Header      http.Header
}

func (c Cred) Apply(req *http.Request) bool {
	if req.URL == nil {
		return false
	}
	ok := false
	for _, prefix := range c.URLPrefixes {
		if prefix.Host == req.URL.Host &&
			(req.URL.Path == prefix.Path ||
				(strings.HasPrefix(req.URL.Path, prefix.Path) &&
					(strings.HasSuffix(prefix.Path, "/") ||
						req.URL.Path[len(prefix.Path)] == '/'))) {
			ok = true
			break
		}
	}
	if !ok {
		return false
	}

	for k, vs := range c.Header {
		req.Header.Del(k)
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	return true
}

func (c Cred) String() string {
	var buf strings.Builder
	for _, u := range c.URLPrefixes {
		fmt.Fprintln(&buf, u)
	}
	buf.WriteString("\n")
	c.Header.Write(&buf)
	buf.WriteString("\n")
	return buf.String()
}

func getCreds(args []string, resp *http.Response) ([]Cred, error) {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stderr = os.Stderr

	if resp != nil {
		u := *resp.Request.URL
		u.RawQuery = ""
		cmd.Args = append(cmd.Args, u.String())
	}

	var head strings.Builder
	if resp != nil {
		formatHead(&head, resp)
	}
	cmd.Stdin = strings.NewReader(head.String())

	fmt.Fprintf(os.Stderr, "# %s\n", strings.Join(cmd.Args, " "))
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%s: %v", strings.Join(cmd.Args, " "), err)
	}
	os.Stderr.Write(out)
	os.Stderr.WriteString("\n")

	var creds []Cred
	r := textproto.NewReader(bufio.NewReader(bytes.NewReader(out)))
	line := 0
readLoop:
	for {
		var prefixes []*url.URL
		for {
			prefix, err := r.ReadLine()
			if err == io.EOF {
				if len(prefixes) > 0 {
					return nil, fmt.Errorf("line %d: %v", line, io.ErrUnexpectedEOF)
				}
				break readLoop
			}
			line++

			if prefix == "" {
				if len(prefixes) == 0 {
					return nil, fmt.Errorf("line %d: unexpected newline", line)
				}
				break
			}
			u, err := url.Parse(prefix)
			if err != nil {
				return nil, fmt.Errorf("line %d: malformed URL: %v", line, err)
			}
			if u.Scheme != "https" {
				return nil, fmt.Errorf("line %d: non-HTTPS URL %q", line, prefix)
			}
			if len(u.RawQuery) > 0 {
				return nil, fmt.Errorf("line %d: unexpected query string in URL %q", line, prefix)
			}
			if len(u.Fragment) > 0 {
				return nil, fmt.Errorf("line %d: unexpected fragment in URL %q", line, prefix)
			}
			prefixes = append(prefixes, u)
		}

		header, err := r.ReadMIMEHeader()
		if err != nil {
			return nil, fmt.Errorf("headers at line %d: %v", line, err)
		}
		if len(header) > 0 {
			creds = append(creds, Cred{
				URLPrefixes: prefixes,
				Header:      http.Header(header),
			})
		}
	}

	return creds, nil
}
