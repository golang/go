// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Rudimentary logging package. Defines a type, Logger, with simple
// methods for formatting output to one or two destinations. Also has
// predefined Loggers accessible through helper functions Stdout[f],
// Stderr[f], Exit[f], and Crash[f], which are easier to use than creating
// a Logger manually.
// Exit exits when written to.
// Crash causes a crash when written to.
package log

import (
	"fmt";
	"io";
	"runtime";
	"os";
	"time";
)

// These flags define the properties of the Logger and the output they produce.
const (
	// Flags
	Lok	= iota;
	Lexit;	// terminate execution when written
	Lcrash;	// crash (panic) when written
	// Bits or'ed together to control what's printed. There is no control over the
	// order they appear (the order listed here) or the format they present (as
	// described in the comments).  A colon appears after these items:
	//	2009/0123 01:23:23.123123 /a/b/c/d.go:23: message
	Ldate		= 1<<iota;	// the date: 2009/0123
	Ltime;		// the time: 01:23:23
	Lmicroseconds;	// microsecond resolution: 01:23:23.123123.  assumes Ltime.
	Llongfile;	// full file name and line number: /a/b/c/d.go:23
	Lshortfile;	// final file name element and line number: d.go:23. overrides Llongfile
	lAllBits	= Ldate | Ltime | Lmicroseconds | Llongfile | Lshortfile;
)

// Logger represents an active logging object.
type Logger struct {
	out0	io.Writer;	// first destination for output
	out1	io.Writer;	// second destination for output; may be nil
	prefix	string;		// prefix to write at beginning of each line
	flag	int;		// properties
}

// New creates a new Logger.   The out0 and out1 variables set the
// destinations to which log data will be written; out1 may be nil.
// The prefix appears at the beginning of each generated log line.
// The flag argument defines the logging properties.
func New(out0, out1 io.Writer, prefix string, flag int) *Logger {
	return &Logger{out0, out1, prefix, flag};
}

var (
	stdout	= New(os.Stdout, nil, "", Lok|Ldate|Ltime);
	stderr	= New(os.Stderr, nil, "", Lok|Ldate|Ltime);
	exit	= New(os.Stderr, nil, "", Lexit|Ldate|Ltime);
	crash	= New(os.Stderr, nil, "", Lcrash|Ldate|Ltime);
)

var shortnames = make(map[string]string)	// cache of short names to avoid allocation.

// Cheap integer to fixed-width decimal ASCII.  Use a negative width to avoid zero-padding
func itoa(i int, wid int) string {
	var u uint = uint(i);
	if u == 0 && wid <= 1 {
		return "0";
	}

	// Assemble decimal in reverse order.
	var b [32]byte;
	bp := len(b);
	for ; u > 0 || wid > 0; u /= 10 {
		bp--;
		wid--;
		b[bp] = byte(u%10)+'0';
	}

	return string(b[bp:len(b)]);
}

func (l *Logger) formatHeader(ns int64, calldepth int) string {
	h := l.prefix;
	if l.flag & (Ldate | Ltime | Lmicroseconds) != 0 {
		t := time.SecondsToLocalTime(ns/1e9);
		if l.flag & (Ldate) != 0 {
			h += itoa(int(t.Year), 4) + "/" + itoa(t.Month, 2) + "/" + itoa(t.Day, 2) + " ";
		}
		if l.flag & (Ltime | Lmicroseconds) != 0 {
			h += itoa(t.Hour, 2) + ":" + itoa(t.Minute, 2) + ":" + itoa(t.Second, 2);
			if l.flag & Lmicroseconds != 0 {
				h += "." + itoa(int(ns%1e9)/1e3, 6);
			}
			h += " ";
		}
	}
	if l.flag & (Lshortfile | Llongfile) != 0 {
		_, file, line, ok := runtime.Caller(calldepth);
		if ok {
			if l.flag & Lshortfile != 0 {
				short, ok := shortnames[file];
				if !ok {
					short = file;
					for i := len(file)-1; i > 0; i-- {
						if file[i] == '/' {
							short = file[i+1 : len(file)];
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

// Output writes the output for a logging event.  The string s contains the text to print after
// the time stamp;  calldepth is used to recover the PC.  It is provided for generality, although
// at the moment on all pre-defined paths it will be 2.
func (l *Logger) Output(calldepth int, s string) {
	now := time.Nanoseconds();	// get this early.
	newline := "\n";
	if len(s) > 0 && s[len(s)-1] == '\n' {
		newline = "";
	}
	s = l.formatHeader(now, calldepth + 1) + s + newline;
	io.WriteString(l.out0, s);
	if l.out1 != nil {
		io.WriteString(l.out1, s);
	}
	switch l.flag & ^lAllBits {
	case Lcrash:
		panic("log: fatal error");
	case Lexit:
		os.Exit(1);
	}
}

// Logf is analogous to Printf() for a Logger.
func (l *Logger) Logf(format string, v ...)	{ l.Output(2, fmt.Sprintf(format, v)) }

// Log is analogouts to Print() for a Logger.
func (l *Logger) Log(v ...)	{ l.Output(2, fmt.Sprintln(v)) }

// Stdout is a helper function for easy logging to stdout. It is analogous to Print().
func Stdout(v ...)	{ stdout.Output(2, fmt.Sprint(v)) }

// Stderr is a helper function for easy logging to stderr. It is analogous to Fprint(os.Stderr).
func Stderr(v ...)	{ stderr.Output(2, fmt.Sprintln(v)) }

// Stdoutf is a helper functions for easy formatted logging to stdout. It is analogous to Printf().
func Stdoutf(format string, v ...)	{ stdout.Output(2, fmt.Sprintf(format, v)) }

// Stderrf is a helper function for easy formatted logging to stderr. It is analogous to Fprintf(os.Stderr).
func Stderrf(format string, v ...)	{ stderr.Output(2, fmt.Sprintf(format, v)) }

// Exit is equivalent to Stderr() followed by a call to os.Exit(1).
func Exit(v ...)	{ exit.Output(2, fmt.Sprintln(v)) }

// Exitf is equivalent to Stderrf() followed by a call to os.Exit(1).
func Exitf(format string, v ...)	{ exit.Output(2, fmt.Sprintf(format, v)) }

// Crash is equivalent to Stderr() followed by a call to panic().
func Crash(v ...)	{ crash.Output(2, fmt.Sprintln(v)) }

// Crashf is equivalent to Stderrf() followed by a call to panic().
func Crashf(format string, v ...)	{ crash.Output(2, fmt.Sprintf(format, v)) }
