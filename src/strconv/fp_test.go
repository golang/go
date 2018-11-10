// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
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

// Wrapper around strconv.ParseFloat(x, 64).  Handles dddddp+ddd (binary exponent)
// itself, passes the rest on to strconv.ParseFloat.
func myatof64(s string) (f float64, ok bool) {
	a := strings.SplitN(s, "p", 2)
	if len(a) == 2 {
		n, err := strconv.ParseInt(a[0], 10, 64)
		if err != nil {
			return 0, false
		}
		e, err1 := strconv.Atoi(a[1])
		if err1 != nil {
			println("bad e", a[1])
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
	f1, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return f1, true
}

// Wrapper around strconv.ParseFloat(x, 32).  Handles dddddp+ddd (binary exponent)
// itself, passes the rest on to strconv.ParseFloat.
func myatof32(s string) (f float32, ok bool) {
	a := strings.SplitN(s, "p", 2)
	if len(a) == 2 {
		n, err := strconv.Atoi(a[0])
		if err != nil {
			println("bad n", a[0])
			return 0, false
		}
		e, err1 := strconv.Atoi(a[1])
		if err1 != nil {
			println("bad p", a[1])
			return 0, false
		}
		return float32(float64(n) * pow2(e)), true
	}
	f64, err1 := strconv.ParseFloat(s, 32)
	f1 := float32(f64)
	if err1 != nil {
		return 0, false
	}
	return f1, true
}

func TestFp(t *testing.T) {
	f, err := os.Open("testdata/testfp.txt")
	if err != nil {
		t.Fatal("testfp: open testdata/testfp.txt:", err)
	}
	defer f.Close()

	s := bufio.NewScanner(f)

	for lineno := 1; s.Scan(); lineno++ {
		line := s.Text()
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		a := strings.Split(line, " ")
		if len(a) != 4 {
			t.Error("testdata/testfp.txt:", lineno, ": wrong field count")
			continue
		}
		var s string
		var v float64
		switch a[0] {
		case "float64":
			var ok bool
			v, ok = myatof64(a[2])
			if !ok {
				t.Error("testdata/testfp.txt:", lineno, ": cannot atof64 ", a[2])
				continue
			}
			s = fmt.Sprintf(a[1], v)
		case "float32":
			v1, ok := myatof32(a[2])
			if !ok {
				t.Error("testdata/testfp.txt:", lineno, ": cannot atof32 ", a[2])
				continue
			}
			s = fmt.Sprintf(a[1], v1)
			v = float64(v1)
		}
		if s != a[3] {
			t.Error("testdata/testfp.txt:", lineno, ": ", a[0], " ", a[1], " ", a[2], " (", v, ") ",
				"want ", a[3], " got ", s)
		}
	}
	if s.Err() != nil {
		t.Fatal("testfp: read testdata/testfp.txt: ", s.Err())
	}
}
