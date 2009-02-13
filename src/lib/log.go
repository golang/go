// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Rudimentary logging package. Defines a type, Logger, with simple
// methods for formatting output to one or two destinations. Also has
// predefined Loggers accessible through helper functions Stdout[f],
// Stderr[f], Exit[f], and Crash[f].
// Exit exits when written to.
// Crash causes a crash when written to.

package log

import (
	"fmt";
	"io";
	"os";
	"time";
)

const (
	// Flags
	Lok = iota;
	Lexit;	// terminate execution when written
	Lcrash;	// crash (panic) when written
	// Bits or'ed together to control what's printed. There is no control over the
	// order they appear (the order listed here) or the format they present (as
	// described in the comments).  A colon appears after these items:
	//	2009/0123 01:23:23.123123 /a/b/c/d.go:23: message
	Ldate = 1 << iota;	// the date: 2009/0123
	Ltime;	// the time: 01:23:23
	Lmicroseconds;	// microsecond resolution: 01:23:23.123123.  assumes Ltime.
	Llongfile;	// full file name and line number: /a/b/c/d.go:23
	Lshortfile;	// final file name element and line number: d.go:23. overrides Llongfile
	lAllBits = Ldate | Ltime | Lmicroseconds | Llongfile | Lshortfile;
)

type Logger struct {
	out0	io.Write;
	out1	io.Write;
	prefix string;
	flag int;
}

func NewLogger(out0, out1 io.Write, prefix string, flag int) *Logger {
	return &Logger(out0, out1, prefix, flag)
}

var (
	stdout = NewLogger(os.Stdout, nil, "", Lok|Ldate|Ltime);
	stderr = NewLogger(os.Stderr, nil, "", Lok|Ldate|Ltime);
	exit = NewLogger(os.Stderr, nil, "", Lexit|Ldate|Ltime);
	crash = NewLogger(os.Stderr, nil, "", Lcrash|Ldate|Ltime);
)

var shortnames = make(map[string] string)	// cache of short names to avoid allocation.

// Cheap integer to fixed-width decimal ASCII.  Use a negative width to avoid zero-padding
func itoa(i int, wid int) string {
	var u uint = uint(i);
	if u == 0 && wid <= 1 {
		return "0"
	}

	// Assemble decimal in reverse order.
	var b [32]byte;
	bp := len(b);
	for ; u > 0 || wid > 0; u /= 10 {
		bp--;
		wid--;
		b[bp] = byte(u%10) + '0';
	}

	return string(b[bp:len(b)])
}

func (l *Logger) formatHeader(ns int64, calldepth int) string {
	h := l.prefix;
	if l.flag & (Ldate | Ltime | Lmicroseconds) != 0 {
		t := time.SecondsToLocalTime(ns/1e9);
		if l.flag & (Ldate) != 0 {
			h += itoa(int(t.Year), 4) + "/" + itoa(t.Month, 2) + itoa(t.Day, 2) + " "
		}
		if l.flag & (Ltime | Lmicroseconds) != 0 {
			h += itoa(t.Hour, 2) + ":" + itoa(t.Minute, 2) + ":" + itoa(t.Second, 2);
			if l.flag & Lmicroseconds != 0 {
				h += "." + itoa(int(ns % 1e9)/1e3, 6);
			}
			h += " ";
		}
	}
	if l.flag & (Lshortfile | Llongfile) != 0 {
		pc, file, line, ok := sys.Caller(calldepth);
		if ok {
			if l.flag & Lshortfile != 0 {
				short, ok := shortnames[file];
				if !ok {
					short = file;
					for i := len(file) - 1; i > 0; i-- {
						if file[i] == '/' {
							short = file[i+1:len(file)];
							break;
						}
					}
					shortnames[file] = short;
				}
				file = short;
			}
		} else {
			file = "???";
			line = 0;
		}
		h += file + ":" + itoa(line, -1) + ": ";
	}
	return h;
}

// The calldepth is provided for generality, although at the moment on all paths it will be 2.
func (l *Logger) Output(calldepth int, s string) {
	now := time.Nanoseconds();	// get this early.
	newline := "\n";
	if len(s) > 0 && s[len(s)-1] == '\n' {
		newline = ""
	}
	s = l.formatHeader(now, calldepth+1) + s + newline;
	io.WriteString(l.out0, s);
	if l.out1 != nil {
		io.WriteString(l.out1, s);
	}
	switch l.flag & ^lAllBits {
	case Lcrash:
		panic("log: fatal error");
	case Lexit:
		sys.Exit(1);
	}
}

// Basic methods on Logger, analogous to Printf and Print
func (l *Logger) Logf(format string, v ...) {
	l.Output(2, fmt.Sprintf(format, v))
}

func (l *Logger) Log(v ...) {
	l.Output(2, fmt.Sprintln(v))
}

// Helper functions for lightweight simple logging to predefined Loggers.
func Stdout(v ...) {
	stdout.Output(2, fmt.Sprint(v))
}

func Stderr(v ...) {
	stdout.Output(2, fmt.Sprintln(v))
}

func Stdoutf(format string, v ...) {
	stdout.Output(2, fmt.Sprintf(format, v))
}

func Stderrf(format string, v ...) {
	stderr.Output(2, fmt.Sprintf(format, v))
}

func Exit(v ...) {
	exit.Output(2, fmt.Sprintln(v))
}

func Exitf(format string, v ...) {
	exit.Output(2, fmt.Sprintf(format, v))
}

func Crash(v ...) {
	crash.Output(2, fmt.Sprintln(v))
}

func Crashf(format string, v ...) {
	crash.Output(2, fmt.Sprintf(format, v))
}
