// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package plugin defines the plugin implementations that the main pprof driver requires.
package plugin

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"cmd/internal/pprof/profile"
)

// A FlagSet creates and parses command-line flags.
// It is similar to the standard flag.FlagSet.
type FlagSet interface {
	// Bool, Int, Float64, and String define new flags,
	// like the functions of the same name in package flag.
	Bool(name string, def bool, usage string) *bool
	Int(name string, def int, usage string) *int
	Float64(name string, def float64, usage string) *float64
	String(name string, def string, usage string) *string

	// ExtraUsage returns any additional text that should be
	// printed after the standard usage message.
	// The typical use of ExtraUsage is to show any custom flags
	// defined by the specific pprof plugins being used.
	ExtraUsage() string

	// Parse initializes the flags with their values for this run
	// and returns the non-flag command line arguments.
	// If an unknown flag is encountered or there are no arguments,
	// Parse should call usage and return nil.
	Parse(usage func()) []string
}

// An ObjTool inspects shared libraries and executable files.
type ObjTool interface {
	// Open opens the named object file.
	// If the object is a shared library, start is the address where
	// it is mapped into memory in the address space being inspected.
	Open(file string, start uint64) (ObjFile, error)

	// Demangle translates a batch of symbol names from mangled
	// form to human-readable form.
	Demangle(names []string) (map[string]string, error)

	// Disasm disassembles the named object file, starting at
	// the start address and stopping at (before) the end address.
	Disasm(file string, start, end uint64) ([]Inst, error)

	// SetConfig configures the tool.
	// The implementation defines the meaning of the string
	// and can ignore it entirely.
	SetConfig(config string)
}

// NoObjTool returns a trivial implementation of the ObjTool interface.
// Open returns an error indicating that the requested file does not exist.
// Demangle returns an empty map and a nil error.
// Disasm returns an error.
// SetConfig is a no-op.
func NoObjTool() ObjTool {
	return noObjTool{}
}

type noObjTool struct{}

func (noObjTool) Open(file string, start uint64) (ObjFile, error) {
	return nil, &os.PathError{Op: "open", Path: file, Err: os.ErrNotExist}
}

func (noObjTool) Demangle(name []string) (map[string]string, error) {
	return make(map[string]string), nil
}

func (noObjTool) Disasm(file string, start, end uint64) ([]Inst, error) {
	return nil, fmt.Errorf("disassembly not supported")
}

func (noObjTool) SetConfig(config string) {
}

// An ObjFile is a single object file: a shared library or executable.
type ObjFile interface {
	// Name returns the underlyinf file name, if available
	Name() string

	// Base returns the base address to use when looking up symbols in the file.
	Base() uint64

	// BuildID returns the GNU build ID of the file, or an empty string.
	BuildID() string

	// SourceLine reports the source line information for a given
	// address in the file. Due to inlining, the source line information
	// is in general a list of positions representing a call stack,
	// with the leaf function first.
	SourceLine(addr uint64) ([]Frame, error)

	// Symbols returns a list of symbols in the object file.
	// If r is not nil, Symbols restricts the list to symbols
	// with names matching the regular expression.
	// If addr is not zero, Symbols restricts the list to symbols
	// containing that address.
	Symbols(r *regexp.Regexp, addr uint64) ([]*Sym, error)

	// Close closes the file, releasing associated resources.
	Close() error
}

// A Frame describes a single line in a source file.
type Frame struct {
	Func string // name of function
	File string // source file name
	Line int    // line in file
}

// A Sym describes a single symbol in an object file.
type Sym struct {
	Name  []string // names of symbol (many if symbol was dedup'ed)
	File  string   // object file containing symbol
	Start uint64   // start virtual address
	End   uint64   // virtual address of last byte in sym (Start+size-1)
}

// An Inst is a single instruction in an assembly listing.
type Inst struct {
	Addr uint64 // virtual address of instruction
	Text string // instruction text
	File string // source file
	Line int    // source line
}

// A UI manages user interactions.
type UI interface {
	// Read returns a line of text (a command) read from the user.
	ReadLine() (string, error)

	// Print shows a message to the user.
	// It formats the text as fmt.Print would and adds a final \n if not already present.
	// For line-based UI, Print writes to standard error.
	// (Standard output is reserved for report data.)
	Print(...interface{})

	// PrintErr shows an error message to the user.
	// It formats the text as fmt.Print would and adds a final \n if not already present.
	// For line-based UI, PrintErr writes to standard error.
	PrintErr(...interface{})

	// IsTerminal returns whether the UI is known to be tied to an
	// interactive terminal (as opposed to being redirected to a file).
	IsTerminal() bool

	// SetAutoComplete instructs the UI to call complete(cmd) to obtain
	// the auto-completion of cmd, if the UI supports auto-completion at all.
	SetAutoComplete(complete func(string) string)
}

// StandardUI returns a UI that reads from standard input,
// prints messages to standard output,
// prints errors to standard error, and doesn't use auto-completion.
func StandardUI() UI {
	return &stdUI{r: bufio.NewReader(os.Stdin)}
}

type stdUI struct {
	r *bufio.Reader
}

func (ui *stdUI) ReadLine() (string, error) {
	os.Stdout.WriteString("(pprof) ")
	return ui.r.ReadString('\n')
}

func (ui *stdUI) Print(args ...interface{}) {
	ui.fprint(os.Stderr, args)
}

func (ui *stdUI) PrintErr(args ...interface{}) {
	ui.fprint(os.Stderr, args)
}

func (ui *stdUI) IsTerminal() bool {
	return false
}

func (ui *stdUI) SetAutoComplete(func(string) string) {
}

func (ui *stdUI) fprint(f *os.File, args []interface{}) {
	text := fmt.Sprint(args...)
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	f.WriteString(text)
}

// A Fetcher reads and returns the profile named by src.
// It gives up after the given timeout, unless src contains a timeout override
// (as defined by the implementation).
// It can print messages to ui.
type Fetcher func(src string, timeout time.Duration, ui UI) (*profile.Profile, error)

// A Symbolizer annotates a profile with symbol information.
// The profile was fetch from src.
// The meaning of mode is defined by the implementation.
type Symbolizer func(mode, src string, prof *profile.Profile, obj ObjTool, ui UI) error
