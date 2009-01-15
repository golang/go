// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv
import (
	"bufio";
	"fmt";
	"os";
	"strconv";
	"strings";
	"testing";
)

func pow2(i int) float64 {
	switch {
	case i < 0:
		return 1 / pow2(-i);
	case i == 0:
		return 1;
	case i == 1:
		return 2;
	}
	return pow2(i/2) * pow2(i-i/2);
}

// Wrapper around strconv.atof64.  Handles dddddp+ddd (binary exponent)
// itself, passes the rest on to strconv.atof64.
func myatof64(s string) (f float64, ok bool) {
	a := strings.split(s, "p");
	if len(a) == 2 {
		n, err := strconv.atoi64(a[0]);
		if err != nil {
			return 0, false;
		}
		e, err1 := strconv.atoi(a[1]);
		if err1 != nil {
			println("bad e", a[1]);
			return 0, false;
		}
		v := float64(n);
		// We expect that v*pow2(e) fits in a float64,
		// but pow2(e) by itself may not.  Be careful.
		if e <= -1000 {
			v *= pow2(-1000);
			e += 1000;
			for e < 0 {
				v /= 2;
				e++;
			}
			return v, true;
		}
		if e >= 1000 {
			v *= pow2(1000);
			e -= 1000;
			for e > 0 {
				v *= 2;
				e--;
			}
			return v, true;
		}
		return v*pow2(e), true;
	}
	f1, err := strconv.atof64(s);
	if err != nil {
		return 0, false;
	}
	return f1, true;
}

// Wrapper around strconv.atof32.  Handles dddddp+ddd (binary exponent)
// itself, passes the rest on to strconv.atof32.
func myatof32(s string) (f float32, ok bool) {
	a := strings.split(s, "p");
	if len(a) == 2 {
		n, err := strconv.atoi(a[0]);
		if err != nil {
			println("bad n", a[0]);
			return 0, false;
		}
		e, err1 := strconv.atoi(a[1]);
		if err1 != nil {
			println("bad p", a[1]);
			return 0, false;
		}
		return float32(float64(n)*pow2(e)), true;
	}
	f1, err1 := strconv.atof32(s);
	if err1 != nil {
		return 0, false;
	}
	return f1, true;
}

export func TestFp(t *testing.T) {
	fd, err := os.Open("testfp.txt", os.O_RDONLY, 0);
	if err != nil {
		panicln("testfp: open testfp.txt:", err.String());
	}

	b, err1 := bufio.NewBufRead(fd);
	if err1 != nil {
		panicln("testfp NewBufRead:", err1.String());
	}

	lineno := 0;
	for {
		line, err2 := b.ReadLineString('\n', false);
		if err2 == bufio.EndOfFile {
			break;
		}
		if err2 != nil {
			panicln("testfp: read testfp.txt:", err2.String());
		}
		lineno++;
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		a := strings.split(line, " ");
		if len(a) != 4 {
			t.Error("testfp.txt:", lineno, ": wrong field count\n");
			continue;
		}
		var s string;
		var v float64;
		switch a[0] {
		case "float64":
			var ok bool;
			v, ok = myatof64(a[2]);
			if !ok {
				t.Error("testfp.txt:", lineno, ": cannot atof64 ", a[2]);
				continue;
			}
			s = fmt.Sprintf(a[1], v);
		case "float32":
			v1, ok := myatof32(a[2]);
			if !ok {
				t.Error("testfp.txt:", lineno, ": cannot atof32 ", a[2]);
				continue;
			}
			s = fmt.Sprintf(a[1], v1);
			v = float64(v1);
		}
		if s != a[3] {
			t.Error("testfp.txt:", lineno, ": ", a[0], " ", a[1], " ", a[2], " (", v, ") ",
				"want ", a[3], " got ", s, "\n");
		}
//else print("testfp.txt:", lineno, ": worked! ", s, "\n");
	}
}
