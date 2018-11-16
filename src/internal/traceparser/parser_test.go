// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var (
	// testfiles from the old trace parser
	otherDir = "../trace/testdata/"
	want     = map[string]bool{"http_1_9_good": true, "http_1_10_good": true, "http_1_11_good": true,
		"stress_1_9_good": true, "stress_1_10_good": true, "stress_1_11_good": true,
		"stress_start_stop_1_9_good": true, "stress_start_stop_1_10_good": true,
		"stress_start_stop_1_11_good": true, "user_task_span_1_11_good": true,

		"http_1_5_good": false, "http_1_7_good": false,
		"stress_1_5_good": false, "stress_1_5_unordered": false, "stress_1_7_good": false,
		"stress_start_stop_1_5_good": false, "stress_start_stop_1_7_good": false,
	}
)

func TestRemoteFiles(t *testing.T) {
	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skipf("files from outside the package are not available on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	files, err := ioutil.ReadDir(otherDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		fname := filepath.Join(otherDir, f.Name())
		p, err := New(fname)
		if err == nil {
			err = p.Parse(0, 1<<62, nil)
		}
		if err == nil != want[f.Name()] {
			t.Errorf("%s: got %v expected %v, err=%v",
				f.Name(), err == nil, want[f.Name()], err)
		}
	}
}

func TestLocalFiles(t *testing.T) {

	files, err := ioutil.ReadDir("./testdata")
	if err != nil {
		t.Fatalf("failed to read ./testdata: %v", err)
	}
	for _, f := range files {
		fname := filepath.Join("./testdata", f.Name())
		p, err := New(fname)
		if err == nil {
			err = p.Parse(0, 1<<62, nil)
		}
		switch {
		case strings.Contains(f.Name(), "good"),
			strings.Contains(f.Name(), "weird"):
			if err != nil {
				t.Errorf("unexpected failure %v %s", err, f.Name())
			}
		case strings.Contains(f.Name(), "bad"):
			if err == nil {
				t.Errorf("bad file did not fail %s", f.Name())
			}
		default:
			t.Errorf("untyped file %v %s", err, f.Name())
		}
	}
}

func TestStats(t *testing.T) {
	// Need just one good file to see that OSStats work properly,
	files, err := ioutil.ReadDir("./testdata")
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if !strings.HasPrefix(f.Name(), "good") {
			continue
		}
		fname := filepath.Join("./testdata", f.Name())
		p, err := New(fname)
		if err != nil {
			t.Fatal(err)
		}
		stat := p.OSStats()
		if stat.Bytes == 0 || stat.Seeks == 0 || stat.Reads == 0 {
			t.Errorf("OSStats impossible %v", stat)
		}
		fd, err := os.Open(fname)
		if err != nil {
			t.Fatal(err)
		}
		pb, err := ParseBuffer(fd)
		if err != nil {
			t.Fatal(err)
		}
		stat = pb.OSStats()
		if stat.Seeks != 0 || stat.Reads != 0 {
			t.Errorf("unexpected positive results %v", stat)
		}
	}
}
