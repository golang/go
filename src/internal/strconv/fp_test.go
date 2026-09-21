// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	_ "embed"
	"flag"
	"fmt"
	. "internal/strconv"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

func pow2(i int) float64 {
	switch {
	case i < 0:
		return 1 / pow2(-i)
	case i == 0:
		return 1
	case i == 1:
		return 2
	}
	return pow2(i/2) * pow2(i-i/2)
}

// Wrapper around ParseFloat(x, 64).  Handles dddddp+ddd (binary exponent)
// itself, passes the rest on to ParseFloat.
func myatof64(s string) (f float64, ok bool) {
	if mant, exp, ok := strings.Cut(s, "p"); ok {
		n, err := ParseInt(mant, 10, 64)
		if err != nil {
			return 0, false
		}
		e, err1 := Atoi(exp)
		if err1 != nil {
			println("bad e", exp)
			return 0, false
		}
		v := float64(n)
		// We expect that v*pow2(e) fits in a float64,
		// but pow2(e) by itself may not. Be careful.
		if e <= -1000 {
			v *= pow2(-1000)
			e += 1000
			for e < 0 {
				v /= 2
				e++
			}
			return v, true
		}
		if e >= 1000 {
			v *= pow2(1000)
			e -= 1000
			for e > 0 {
				v *= 2
				e--
			}
			return v, true
		}
		return v * pow2(e), true
	}
	f1, err := ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return f1, true
}

// Wrapper around strconv.ParseFloat(x, 32).  Handles dddddp+ddd (binary exponent)
// itself, passes the rest on to strconv.ParseFloat.
func myatof32(s string) (f float32, ok bool) {
	if mant, exp, ok := strings.Cut(s, "p"); ok {
		n, err := Atoi(mant)
		if err != nil {
			println("bad n", mant)
			return 0, false
		}
		e, err1 := Atoi(exp)
		if err1 != nil {
			println("bad p", exp)
			return 0, false
		}
		return float32(float64(n) * pow2(e)), true
	}
	f64, err1 := ParseFloat(s, 32)
	f1 := float32(f64)
	if err1 != nil {
		return 0, false
	}
	return f1, true
}

//go:embed testdata/testfp.txt
var testfp string

func TestFp(t *testing.T) {
	lineno := 0
	for line := range strings.Lines(testfp) {
		lineno++
		line, _, _ = strings.Cut(line, "#")
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		a := strings.Split(line, " ")
		if len(a) != 4 {
			t.Errorf("testdata/testfp.txt:%d: wrong field count", lineno)
			continue
		}
		var s string
		var v float64
		switch a[0] {
		case "float64":
			var ok bool
			v, ok = myatof64(a[2])
			if !ok {
				t.Errorf("testdata/testfp.txt:%d: cannot atof64 %s", lineno, a[2])
				continue
			}
			s = fmt.Sprintf(a[1], v)
		case "float32":
			v1, ok := myatof32(a[2])
			if !ok {
				t.Errorf("testdata/testfp.txt:%d: cannot atof32 %s", lineno, a[2])
				continue
			}
			s = fmt.Sprintf(a[1], v1)
			v = float64(v1)
		}
		if s != a[3] {
			t.Errorf("testdata/testfp.txt:%d: %s %s %s %s: have %s want %s", lineno, a[0], a[1], a[2], a[3], s, a[3])
		}
	}
}

// The -testbase flag runs the full testbase input set instead of the
// random sample in testdata/*1k.txt. See testdata/README for details.
var testbase = flag.Bool("testbase", false, "download and test full testbase testdata")

// testbaseURL is the URL for downloading the full testbase testdata.
// There is also a copy on "https://swtch.com/testbase/".
var testbaseURL = "https://gist.githubusercontent.com/rsc/606b378b0bf95c24a6fd6cef99e262e1/raw/128a03890e536bdf403e6cc768b0737405c6734d/"

//go:embed testdata/atof1k.txt
var atof1ktxt string

//go:embed testdata/ftoa1k.txt
var ftoa1ktxt string

// openTestbase opens the named testbase data file.
// By default it opens testdata/name1k.txt,
// but if the -testbase flag has been set,
// then it opens the full testdata/name.txt,
// downloading that file if necessary.
func openTestbase(t *testing.T, name string) (file, data string) {
	if !*testbase {
		switch name {
		case "atof":
			return "testdata/atof1k.txt", atof1ktxt
		case "ftoa":
			return "testdata/ftoa1k.txt", ftoa1ktxt
		}
		t.Fatalf("unknown file %s", name)
	}

	// Use cached copy if present.
	file = "testdata/" + name + ".txt"
	if data, err := os.ReadFile(file); err == nil {
		return file, string(data)
	}

	// Download copy.
	url := testbaseURL + name + ".txt"
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("%s: %s", url, err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("%s: %s", url, resp.Status)
	}
	bytes, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatalf("%s: %s", url, err)
	}
	if err := os.WriteFile(file, bytes, 0666); err != nil {
		t.Fatal(err)
	}
	return file, string(bytes)
}

func TestParseFloatTestdata(t *testing.T) {
	// Test testbase inputs, optimized against not.
	name, data := openTestbase(t, "atof")
	fail := 0
	lineno := 0
	for line := range strings.Lines(data) {
		lineno++
		s := strings.TrimSpace(line)
		if strings.HasPrefix(s, "#") || s == "" {
			continue
		}
		SetOptimize(false)
		want, err1 := ParseFloat(s, 64)
		SetOptimize(true)
		have, err2 := ParseFloat(s, 64)
		if err1 != nil {
			// Error in test data; should not happen.
			t.Errorf("%s:%d: ParseFloat(%#q): %v", name, lineno, s, err1)
			continue
		}
		if err2 != nil {
			t.Errorf("ParseFloat(%#q): %v", s, err2)
			if fail++; fail > 100 {
				t.Fatalf("too many failures")
			}
			continue
		}
		if have != want {
			t.Errorf("ParseFloat(%#q) = %#x, want %#x", s, have, want)
			if fail++; fail > 100 {
				t.Fatalf("too many failures")
			}
		}
	}
}

func TestFormatFloatTestdata(t *testing.T) {
	// Test testbase inputs, optimized against not.
	name, data := openTestbase(t, "ftoa")
	fail := 0
	lineno := 0
	for line := range strings.Lines(data) {
		lineno++
		s := strings.TrimSpace(line)
		if strings.HasPrefix(s, "#") || s == "" {
			continue
		}
		f, err := ParseFloat(s, 64)
		if err != nil {
			// Error in test data; should not happen.
			t.Errorf("%s:%d: ParseFloat(%#q): %v", name, lineno, s, err)
			continue
		}
		for i := range 19 {
			SetOptimize(false)
			want := FormatFloat(f, 'e', i, 64)
			SetOptimize(true)
			have := FormatFloat(f, 'e', i, 64)
			if have != want {
				t.Errorf("FormatFloat(%#x, 'e', %d) = %s, want %s", f, i, have, want)
				if fail++; fail > 100 {
					t.Fatalf("too many failures")
				}
			}
		}
	}
}
