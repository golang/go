// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webtest

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	web "cmd/go/internal/web2"
)

var mode = flag.String("webtest", "replay", "set webtest `mode` - record, replay, bypass")

func Hook() {
	if *mode == "bypass" {
		return
	}
	web.SetHTTPDoForTesting(Do)
}

func Unhook() {
	web.SetHTTPDoForTesting(nil)
}

func Print() {
	web.SetHTTPDoForTesting(DoPrint)
}

var responses struct {
	mu    sync.Mutex
	byURL map[string]*respEntry
}

type respEntry struct {
	status string
	code   int
	hdr    http.Header
	body   []byte
}

func Serve(url string, status string, hdr http.Header, body []byte) {
	if status == "" {
		status = "200 OK"
	}
	code, err := strconv.Atoi(strings.Fields(status)[0])
	if err != nil {
		panic("bad Serve status - " + status + " - " + err.Error())
	}

	responses.mu.Lock()
	defer responses.mu.Unlock()

	if responses.byURL == nil {
		responses.byURL = make(map[string]*respEntry)
	}
	responses.byURL[url] = &respEntry{status: status, code: code, hdr: web.CopyHeader(hdr), body: body}
}

func Do(req *http.Request) (*http.Response, error) {
	if req.Method != "GET" {
		return nil, fmt.Errorf("bad method - must be GET")
	}

	responses.mu.Lock()
	e := responses.byURL[req.URL.String()]
	responses.mu.Unlock()

	if e == nil {
		if *mode == "record" {
			loaded.mu.Lock()
			if len(loaded.did) != 1 {
				loaded.mu.Unlock()
				return nil, fmt.Errorf("cannot use -webtest=record with multiple loaded response files")
			}
			var file string
			for file = range loaded.did {
				break
			}
			loaded.mu.Unlock()
			return doSave(file, req)
		}
		e = &respEntry{code: 599, status: "599 unexpected request (no canned response)"}
	}
	resp := &http.Response{
		Status:     e.status,
		StatusCode: e.code,
		Header:     web.CopyHeader(e.hdr),
		Body:       ioutil.NopCloser(bytes.NewReader(e.body)),
	}
	return resp, nil
}

func DoPrint(req *http.Request) (*http.Response, error) {
	return doSave("", req)
}

func doSave(file string, req *http.Request) (*http.Response, error) {
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	data, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, err
	}
	resp.Body = ioutil.NopCloser(bytes.NewReader(data))

	var f *os.File
	if file == "" {
		f = os.Stderr
	} else {
		f, err = os.OpenFile(file, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
	}

	fmt.Fprintf(f, "GET %s\n", req.URL.String())
	fmt.Fprintf(f, "%s\n", resp.Status)
	var keys []string
	for k := range resp.Header {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if k == "Set-Cookie" {
			continue
		}
		for _, v := range resp.Header[k] {
			fmt.Fprintf(f, "%s: %s\n", k, v)
		}
	}
	fmt.Fprintf(f, "\n")
	if utf8.Valid(data) && !bytes.Contains(data, []byte("\nGET")) && !isHexDump(data) {
		fmt.Fprintf(f, "%s\n\n", data)
	} else {
		fmt.Fprintf(f, "%s\n", hex.Dump(data))
	}
	return resp, err
}

var loaded struct {
	mu  sync.Mutex
	did map[string]bool
}

func LoadOnce(file string) {
	loaded.mu.Lock()
	if loaded.did[file] {
		loaded.mu.Unlock()
		return
	}
	if loaded.did == nil {
		loaded.did = make(map[string]bool)
	}
	loaded.did[file] = true
	loaded.mu.Unlock()

	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	b := bufio.NewReader(f)
	var ungetLine string
	nextLine := func() string {
		if ungetLine != "" {
			l := ungetLine
			ungetLine = ""
			return l
		}
		line, err := b.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return ""
			}
			log.Fatalf("%s: unexpected read error: %v", file, err)
		}
		return line
	}

	for {
		line := nextLine()
		if line == "" { // EOF
			break
		}
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		if !strings.HasPrefix(line, "GET ") {
			log.Fatalf("%s: malformed GET line: %s", file, line)
		}
		url := line[len("GET "):]
		status := nextLine()
		if _, err := strconv.Atoi(strings.Fields(status)[0]); err != nil {
			log.Fatalf("%s: malformed status line (after GET %s): %s", file, url, status)
		}
		hdr := make(http.Header)
		for {
			kv := strings.TrimSpace(nextLine())
			if kv == "" {
				break
			}
			i := strings.Index(kv, ":")
			if i < 0 {
				log.Fatalf("%s: malformed header line (after GET %s): %s", file, url, kv)
			}
			k, v := kv[:i], strings.TrimSpace(kv[i+1:])
			hdr[k] = append(hdr[k], v)
		}

		var body []byte
	Body:
		for n := 0; ; n++ {
			line := nextLine()
			if n == 0 && isHexDump([]byte(line)) {
				ungetLine = line
				b, err := parseHexDump(nextLine)
				if err != nil {
					log.Fatalf("%s: malformed hex dump (after GET %s): %v", file, url, err)
				}
				body = b
				break
			}
			if line == "" { // EOF
				for i := 0; i < 2; i++ {
					if len(body) > 0 && body[len(body)-1] == '\n' {
						body = body[:len(body)-1]
					}
				}
				break
			}
			body = append(body, line...)
			for line == "\n" {
				line = nextLine()
				if strings.HasPrefix(line, "GET ") {
					ungetLine = line
					body = body[:len(body)-1]
					if len(body) > 0 {
						body = body[:len(body)-1]
					}
					break Body
				}
				body = append(body, line...)
			}
		}

		Serve(url, status, hdr, body)
	}
}

func isHexDump(data []byte) bool {
	return bytes.HasPrefix(data, []byte("00000000  ")) || bytes.HasPrefix(data, []byte("0000000 "))
}

// parseHexDump parses the hex dump in text, which should be the
// output of "hexdump -C" or Plan 9's "xd -b" or Go's hex.Dump
// and returns the original data used to produce the dump.
// It is meant to enable storing golden binary files as text, so that
// changes to the golden files can be seen during code reviews.
func parseHexDump(nextLine func() string) ([]byte, error) {
	var out []byte
	for {
		line := nextLine()
		if line == "" || line == "\n" {
			break
		}
		if i := strings.Index(line, "|"); i >= 0 { // remove text dump
			line = line[:i]
		}
		f := strings.Fields(line)
		if len(f) > 1+16 {
			return nil, fmt.Errorf("parsing hex dump: too many fields on line %q", line)
		}
		if len(f) == 0 || len(f) == 1 && f[0] == "*" { // all zeros block omitted
			continue
		}
		addr64, err := strconv.ParseUint(f[0], 16, 0)
		if err != nil {
			return nil, fmt.Errorf("parsing hex dump: invalid address %q", f[0])
		}
		addr := int(addr64)
		if len(out) < addr {
			out = append(out, make([]byte, addr-len(out))...)
		}
		for _, x := range f[1:] {
			val, err := strconv.ParseUint(x, 16, 8)
			if err != nil {
				return nil, fmt.Errorf("parsing hexdump: invalid hex byte %q", x)
			}
			out = append(out, byte(val))
		}
	}
	return out, nil
}
