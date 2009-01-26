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

// Lshortname can be or'd in to cause only the last element of the file name to be printed.
const (
	Lok = iota;
	Lexit;	// terminate execution when written
	Lcrash;	// crash (panic) when written
	Lshortname = 1 << 5;
)

type Logger struct {
	out0	io.Write;
	out1	io.Write;
	flag int;
}

func NewLogger(out0, out1 io.Write, flag int) *Logger {
	return &Logger{out0, out1, flag}
}

var (
	stdout = NewLogger(os.Stdout, nil, Lok);
	stderr = NewLogger(os.Stderr, nil, Lok);
	exit = NewLogger(os.Stderr, nil, Lexit);
	crash = NewLogger(os.Stderr, nil, Lcrash);
)

func timestamp(ns int64) string {
	t := time.SecondsToLocalTime(ns/1e9);
	// why are time fields private?
	s := t.RFC1123();
	return s[5:12] + s[17:25];	// TODO(r): placeholder. this gives "24 Jan 15:50:18"
}

var shortnames = make(map[string] string)	// cache of short names to avoid allocation.

// The calldepth is provided for generality, although at the moment on all paths it will be 2.
func (l *Logger) output(calldepth int, s string) {
	now := time.Nanoseconds();	// get this early.
	newline := "\n";
	if len(s) > 0 && s[len(s)-1] == '\n' {
		newline = ""
	}
	pc, file, line, ok := sys.Caller(calldepth);
	if ok {
		if l.flag & Lshortname == Lshortname {
			short, ok := shortnames[file];
			if !ok {
				short = file;
				for i := len(file) - 1; i > 0; i-- {
					if file[i] == '/' {
						short = file[i+1:len(file)];
						shortnames[file] = short;
						break;
					}
				}
			}
			file = short;
		}
	} else {
		file = "???";
		line = 0;
	}
	s = fmt.Sprintf("%s %s:%d: %s%s", timestamp(now), file, line, s, newline);
	io.WriteString(l.out0, s);
	if l.out1 != nil {
		io.WriteString(l.out1, s);
	}
	switch l.flag & ^Lshortname {
	case Lcrash:
		panic("log: fatal error");
	case Lexit:
		sys.Exit(1);
	}
}

// Basic methods on Logger, analogous to Printf and Print
func (l *Logger) Logf(format string, v ...) {
	l.output(2, fmt.Sprintf(format, v))
}

func (l *Logger) Log(v ...) {
	l.output(2, fmt.Sprintln(v))
}

// Helper functions for lightweight simple logging to predefined Loggers.
func Stdout(v ...) {
	stdout.output(2, fmt.Sprint(v))
}

func Stderr(v ...) {
	stdout.output(2, fmt.Sprintln(v))
}

func Stdoutf(format string, v ...) {
	stdout.output(2, fmt.Sprintf(format, v))
}

func Stderrf(format string, v ...) {
	stderr.output(2, fmt.Sprintf(format, v))
}

func Exit(v ...) {
	exit.output(2, fmt.Sprintln(v))
}

func Exitf(format string, v ...) {
	exit.output(2, fmt.Sprintf(format, v))
}

func Crash(v ...) {
	crash.output(2, fmt.Sprintln(v))
}

func Crashf(format string, v ...) {
	crash.output(2, fmt.Sprintf(format, v))
}
