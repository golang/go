// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ogle is the beginning of a debugger for Go.
package ogle

import (
	"bufio"
	"debug/elf"
	"debug/proc"
	"exp/eval"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"strconv"
	"strings"
)

var fset = token.NewFileSet()
var world *eval.World
var curProc *Process

func Main() {
	world = eval.NewWorld()
	defineFuncs()
	r := bufio.NewReader(os.Stdin)
	for {
		print("; ")
		line, err := r.ReadSlice('\n')
		if err != nil {
			break
		}

		// Try line as a command
		cmd, rest := getCmd(line)
		if cmd != nil {
			err := cmd.handler(rest)
			if err != nil {
				scanner.PrintError(os.Stderr, err)
			}
			continue
		}

		// Try line as code
		code, err := world.Compile(fset, string(line))
		if err != nil {
			scanner.PrintError(os.Stderr, err)
			continue
		}
		v, err := code.Run()
		if err != nil {
			fmt.Fprintf(os.Stderr, err.String())
			continue
		}
		if v != nil {
			println(v.String())
		}
	}
}

// newScanner creates a new scanner that scans that given input bytes.
func newScanner(input []byte) (*scanner.Scanner, *scanner.ErrorVector) {
	sc := new(scanner.Scanner)
	ev := new(scanner.ErrorVector)
	file := fset.AddFile("input", fset.Base(), len(input))
	sc.Init(file, input, ev, 0)
	return sc, ev
}

/*
 * Commands
 */

// A UsageError occurs when a command is called with illegal arguments.
type UsageError string

func (e UsageError) String() string { return string(e) }

// A cmd represents a single command with a handler.
type cmd struct {
	cmd     string
	handler func([]byte) os.Error
}

var cmds = []cmd{
	{"load", cmdLoad},
	{"bt", cmdBt},
}

// getCmd attempts to parse an input line as a registered command.  If
// successful, it returns the command and the bytes remaining after
// the command, which should be passed to the command.
func getCmd(line []byte) (*cmd, []byte) {
	sc, _ := newScanner(line)
	pos, tok, lit := sc.Scan()
	if sc.ErrorCount != 0 || tok != token.IDENT {
		return nil, nil
	}

	slit := string(lit)
	for i := range cmds {
		if cmds[i].cmd == slit {
			return &cmds[i], line[fset.Position(pos).Offset+len(lit):]
		}
	}
	return nil, nil
}

// cmdLoad starts or attaches to a process.  Its form is similar to
// import:
//
//  load [sym] "path" [;]
//
// sym specifies the name to give to the process.  If not given, the
// name is derived from the path of the process.  If ".", then the
// packages from the remote process are defined into the current
// namespace.  If given, this symbol is defined as a package
// containing the process' packages.
//
// path gives the path of the process to start or attach to.  If it is
// "pid:<num>", then attach to the given PID.  Otherwise, treat it as
// a file path and space-separated arguments and start a new process.
//
// load always sets the current process to the loaded process.
func cmdLoad(args []byte) os.Error {
	ident, path, err := parseLoad(args)
	if err != nil {
		return err
	}
	if curProc != nil {
		return UsageError("multiple processes not implemented")
	}
	if ident != "." {
		return UsageError("process identifiers not implemented")
	}

	// Parse argument and start or attach to process
	var fname string
	var tproc proc.Process
	if len(path) >= 4 && path[0:4] == "pid:" {
		pid, err := strconv.Atoi(path[4:])
		if err != nil {
			return err
		}
		fname, err = os.Readlink(fmt.Sprintf("/proc/%d/exe", pid))
		if err != nil {
			return err
		}
		tproc, err = proc.Attach(pid)
		if err != nil {
			return err
		}
		println("Attached to", pid)
	} else {
		parts := strings.Split(path, " ", -1)
		if len(parts) == 0 {
			fname = ""
		} else {
			fname = parts[0]
		}
		tproc, err = proc.ForkExec(fname, parts, os.Environ(), "", []*os.File{os.Stdin, os.Stdout, os.Stderr})
		if err != nil {
			return err
		}
		println("Started", path)
		// TODO(austin) If we fail after this point, kill tproc
		// before detaching.
	}

	// Get symbols
	f, err := os.Open(fname, os.O_RDONLY, 0)
	if err != nil {
		tproc.Detach()
		return err
	}
	defer f.Close()
	elf, err := elf.NewFile(f)
	if err != nil {
		tproc.Detach()
		return err
	}
	curProc, err = NewProcessElf(tproc, elf)
	if err != nil {
		tproc.Detach()
		return err
	}

	// Prepare new process
	curProc.OnGoroutineCreate().AddHandler(EventPrint)
	curProc.OnGoroutineExit().AddHandler(EventPrint)

	err = curProc.populateWorld(world)
	if err != nil {
		tproc.Detach()
		return err
	}

	return nil
}

func parseLoad(args []byte) (ident string, path string, err os.Error) {
	err = UsageError("Usage: load [sym] \"path\"")
	sc, ev := newScanner(args)

	var toks [4]token.Token
	var lits [4][]byte
	for i := range toks {
		_, toks[i], lits[i] = sc.Scan()
	}
	if sc.ErrorCount != 0 {
		err = ev.GetError(scanner.NoMultiples)
		return
	}

	i := 0
	switch toks[i] {
	case token.PERIOD, token.IDENT:
		ident = string(lits[i])
		i++
	}

	if toks[i] != token.STRING {
		return
	}
	path, uerr := strconv.Unquote(string(lits[i]))
	if uerr != nil {
		err = uerr
		return
	}
	i++

	if toks[i] == token.SEMICOLON {
		i++
	}
	if toks[i] != token.EOF {
		return
	}

	return ident, path, nil
}

// cmdBt prints a backtrace for the current goroutine.  It takes no
// arguments.
func cmdBt(args []byte) os.Error {
	err := parseNoArgs(args, "Usage: bt")
	if err != nil {
		return err
	}

	if curProc == nil || curProc.curGoroutine == nil {
		return NoCurrentGoroutine{}
	}

	f := curProc.curGoroutine.frame
	if f == nil {
		fmt.Println("No frames on stack")
		return nil
	}

	for f.Inner() != nil {
		f = f.Inner()
	}

	for i := 0; i < 100; i++ {
		if f == curProc.curGoroutine.frame {
			fmt.Printf("=> ")
		} else {
			fmt.Printf("   ")
		}
		fmt.Printf("%8x %v\n", f.pc, f)
		f, err = f.Outer()
		if err != nil {
			return err
		}
		if f == nil {
			return nil
		}
	}

	fmt.Println("...")
	return nil
}

func parseNoArgs(args []byte, usage string) os.Error {
	sc, ev := newScanner(args)
	_, tok, _ := sc.Scan()
	if sc.ErrorCount != 0 {
		return ev.GetError(scanner.NoMultiples)
	}
	if tok != token.EOF {
		return UsageError(usage)
	}
	return nil
}

/*
 * Functions
 */

// defineFuncs populates world with the built-in functions.
func defineFuncs() {
	t, v := eval.FuncFromNativeTyped(fnOut, fnOutSig)
	world.DefineConst("Out", t, v)
	t, v = eval.FuncFromNativeTyped(fnContWait, fnContWaitSig)
	world.DefineConst("ContWait", t, v)
	t, v = eval.FuncFromNativeTyped(fnBpSet, fnBpSetSig)
	world.DefineConst("BpSet", t, v)
}

// printCurFrame prints the current stack frame, as it would appear in
// a backtrace.
func printCurFrame() {
	if curProc == nil || curProc.curGoroutine == nil {
		return
	}
	f := curProc.curGoroutine.frame
	if f == nil {
		return
	}
	fmt.Printf("=> %8x %v\n", f.pc, f)
}

// fnOut moves the current frame to the caller of the current frame.
func fnOutSig() {}
func fnOut(t *eval.Thread, args []eval.Value, res []eval.Value) {
	if curProc == nil {
		t.Abort(NoCurrentGoroutine{})
	}
	err := curProc.Out()
	if err != nil {
		t.Abort(err)
	}
	// TODO(austin) Only in the command form
	printCurFrame()
}

// fnContWait continues the current process and waits for a stopping event.
func fnContWaitSig() {}
func fnContWait(t *eval.Thread, args []eval.Value, res []eval.Value) {
	if curProc == nil {
		t.Abort(NoCurrentGoroutine{})
	}
	err := curProc.ContWait()
	if err != nil {
		t.Abort(err)
	}
	// TODO(austin) Only in the command form
	ev := curProc.Event()
	if ev != nil {
		fmt.Printf("%v\n", ev)
	}
	printCurFrame()
}

// fnBpSet sets a breakpoint at the entry to the named function.
func fnBpSetSig(string) {}
func fnBpSet(t *eval.Thread, args []eval.Value, res []eval.Value) {
	// TODO(austin) This probably shouldn't take a symbol name.
	// Perhaps it should take an interface that provides PC's.
	// Functions and instructions can implement that interface and
	// we can have something to translate file:line pairs.
	if curProc == nil {
		t.Abort(NoCurrentGoroutine{})
	}
	name := args[0].(eval.StringValue).Get(t)
	fn := curProc.syms.LookupFunc(name)
	if fn == nil {
		t.Abort(UsageError("no such function " + name))
	}
	curProc.OnBreakpoint(proc.Word(fn.Entry)).AddHandler(EventStop)
}
