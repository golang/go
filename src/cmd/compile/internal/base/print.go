// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"internal/buildcfg"
	"internal/types/errors"
	"os"
	"runtime/debug"
	"sort"
	"strings"

	"cmd/internal/src"
	"cmd/internal/telemetry/counter"
)

// An errorMsg is a queued error message, waiting to be printed.
type errorMsg struct {
	pos  src.XPos
	msg  string
	code errors.Code
}

// Pos is the current source position being processed,
// printed by Errorf, ErrorfLang, Fatalf, and Warnf.
var Pos src.XPos

var (
	errorMsgs       []errorMsg
	numErrors       int // number of entries in errorMsgs that are errors (as opposed to warnings)
	numSyntaxErrors int
)

// Errors returns the number of errors reported.
func Errors() int {
	return numErrors
}

// SyntaxErrors returns the number of syntax errors reported.
func SyntaxErrors() int {
	return numSyntaxErrors
}

// addErrorMsg adds a new errorMsg (which may be a warning) to errorMsgs.
func addErrorMsg(pos src.XPos, code errors.Code, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	// Only add the position if know the position.
	// See issue golang.org/issue/11361.
	if pos.IsKnown() {
		msg = fmt.Sprintf("%v: %s", FmtPos(pos), msg)
	}
	errorMsgs = append(errorMsgs, errorMsg{
		pos:  pos,
		msg:  msg + "\n",
		code: code,
	})
}

// FmtPos formats pos as a file:line string.
func FmtPos(pos src.XPos) string {
	if Ctxt == nil {
		return "???"
	}
	return Ctxt.OutermostPos(pos).Format(Flag.C == 0, Flag.L == 1)
}

// byPos sorts errors by source position.
type byPos []errorMsg

func (x byPos) Len() int           { return len(x) }
func (x byPos) Less(i, j int) bool { return x[i].pos.Before(x[j].pos) }
func (x byPos) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }

// FlushErrors sorts errors seen so far by line number, prints them to stdout,
// and empties the errors array.
func FlushErrors() {
	if Ctxt != nil && Ctxt.Bso != nil {
		Ctxt.Bso.Flush()
	}
	if len(errorMsgs) == 0 {
		return
	}
	sort.Stable(byPos(errorMsgs))
	for i, err := range errorMsgs {
		if i == 0 || err.msg != errorMsgs[i-1].msg {
			fmt.Print(err.msg)
		}
	}
	errorMsgs = errorMsgs[:0]
}

// lasterror keeps track of the most recently issued error,
// to avoid printing multiple error messages on the same line.
var lasterror struct {
	syntax src.XPos // source position of last syntax error
	other  src.XPos // source position of last non-syntax error
	msg    string   // error message of last non-syntax error
}

// sameline reports whether two positions a, b are on the same line.
func sameline(a, b src.XPos) bool {
	p := Ctxt.PosTable.Pos(a)
	q := Ctxt.PosTable.Pos(b)
	return p.Base() == q.Base() && p.Line() == q.Line()
}

// Errorf reports a formatted error at the current line.
func Errorf(format string, args ...any) {
	ErrorfAt(Pos, 0, format, args...)
}

// ErrorfAt reports a formatted error message at pos.
func ErrorfAt(pos src.XPos, code errors.Code, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)

	if strings.HasPrefix(msg, "syntax error") {
		numSyntaxErrors++
		// only one syntax error per line, no matter what error
		if sameline(lasterror.syntax, pos) {
			return
		}
		lasterror.syntax = pos
	} else {
		// only one of multiple equal non-syntax errors per line
		// (FlushErrors shows only one of them, so we filter them
		// here as best as we can (they may not appear in order)
		// so that we don't count them here and exit early, and
		// then have nothing to show for.)
		if sameline(lasterror.other, pos) && lasterror.msg == msg {
			return
		}
		lasterror.other = pos
		lasterror.msg = msg
	}

	addErrorMsg(pos, code, "%s", msg)
	numErrors++

	hcrash()
	if numErrors >= 10 && Flag.LowerE == 0 {
		FlushErrors()
		fmt.Printf("%v: too many errors\n", FmtPos(pos))
		ErrorExit()
	}
}

// UpdateErrorDot is a clumsy hack that rewrites the last error,
// if it was "LINE: undefined: NAME", to be "LINE: undefined: NAME in EXPR".
// It is used to give better error messages for dot (selector) expressions.
func UpdateErrorDot(line string, name, expr string) {
	if len(errorMsgs) == 0 {
		return
	}
	e := &errorMsgs[len(errorMsgs)-1]
	if strings.HasPrefix(e.msg, line) && e.msg == fmt.Sprintf("%v: undefined: %v\n", line, name) {
		e.msg = fmt.Sprintf("%v: undefined: %v in %v\n", line, name, expr)
	}
}

// Warn reports a formatted warning at the current line.
// In general the Go compiler does NOT generate warnings,
// so this should be used only when the user has opted in
// to additional output by setting a particular flag.
func Warn(format string, args ...any) {
	WarnfAt(Pos, format, args...)
}

// WarnfAt reports a formatted warning at pos.
// In general the Go compiler does NOT generate warnings,
// so this should be used only when the user has opted in
// to additional output by setting a particular flag.
func WarnfAt(pos src.XPos, format string, args ...any) {
	addErrorMsg(pos, 0, format, args...)
	if Flag.LowerM != 0 {
		FlushErrors()
	}
}

// Fatalf reports a fatal error - an internal problem - at the current line and exits.
// If other errors have already been printed, then Fatalf just quietly exits.
// (The internal problem may have been caused by incomplete information
// after the already-reported errors, so best to let users fix those and
// try again without being bothered about a spurious internal error.)
//
// But if no errors have been printed, or if -d panic has been specified,
// Fatalf prints the error as an "internal compiler error". In a released build,
// it prints an error asking to file a bug report. In development builds, it
// prints a stack trace.
//
// If -h has been specified, Fatalf panics to force the usual runtime info dump.
func Fatalf(format string, args ...any) {
	FatalfAt(Pos, format, args...)
}

var bugStack = counter.NewStack("compile/bug", 16) // 16 is arbitrary; used by gopls and crashmonitor

// FatalfAt reports a fatal error - an internal problem - at pos and exits.
// If other errors have already been printed, then FatalfAt just quietly exits.
// (The internal problem may have been caused by incomplete information
// after the already-reported errors, so best to let users fix those and
// try again without being bothered about a spurious internal error.)
//
// But if no errors have been printed, or if -d panic has been specified,
// FatalfAt prints the error as an "internal compiler error". In a released build,
// it prints an error asking to file a bug report. In development builds, it
// prints a stack trace.
//
// If -h has been specified, FatalfAt panics to force the usual runtime info dump.
func FatalfAt(pos src.XPos, format string, args ...any) {
	FlushErrors()

	bugStack.Inc()

	if Debug.Panic != 0 || numErrors == 0 {
		fmt.Printf("%v: internal compiler error: ", FmtPos(pos))
		fmt.Printf(format, args...)
		fmt.Printf("\n")

		// If this is a released compiler version, ask for a bug report.
		if Debug.Panic == 0 && strings.HasPrefix(buildcfg.Version, "go") && !strings.Contains(buildcfg.Version, "devel") {
			fmt.Printf("\n")
			fmt.Printf("Please file a bug report including a short program that triggers the error.\n")
			fmt.Printf("https://go.dev/issue/new\n")
		} else {
			// Not a release; dump a stack trace, too.
			fmt.Println()
			os.Stdout.Write(debug.Stack())
			fmt.Println()
		}
	}

	hcrash()
	ErrorExit()
}

// Assert reports "assertion failed" with Fatalf, unless b is true.
func Assert(b bool) {
	if !b {
		Fatalf("assertion failed")
	}
}

// Assertf reports a fatal error with Fatalf, unless b is true.
func Assertf(b bool, format string, args ...any) {
	if !b {
		Fatalf(format, args...)
	}
}

// AssertfAt reports a fatal error with FatalfAt, unless b is true.
func AssertfAt(b bool, pos src.XPos, format string, args ...any) {
	if !b {
		FatalfAt(pos, format, args...)
	}
}

// hcrash crashes the compiler when -h is set, to find out where a message is generated.
func hcrash() {
	if Flag.LowerH != 0 {
		FlushErrors()
		if Flag.LowerO != "" {
			os.Remove(Flag.LowerO)
		}
		panic("-h")
	}
}

// ErrorExit handles an error-status exit.
// It flushes any pending errors, removes the output file, and exits.
func ErrorExit() {
	FlushErrors()
	if Flag.LowerO != "" {
		os.Remove(Flag.LowerO)
	}
	os.Exit(2)
}

// ExitIfErrors calls ErrorExit if any errors have been reported.
func ExitIfErrors() {
	if Errors() > 0 {
		ErrorExit()
	}
}

var AutogeneratedPos src.XPos
