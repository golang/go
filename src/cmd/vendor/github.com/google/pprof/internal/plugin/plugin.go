// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package plugin defines the plugin implementations that the main pprof driver requires.
package plugin

import (
	"io"
	"net/http"
	"regexp"
	"time"

	"github.com/google/pprof/profile"
)

// Options groups all the optional plugins into pprof.
type Options struct {
	Writer  Writer
	Flagset FlagSet
	Fetch   Fetcher
	Sym     Symbolizer
	Obj     ObjTool
	UI      UI

	// HTTPServer is a function that should block serving http requests,
	// including the handlers specified in args.  If non-nil, pprof will
	// invoke this function if necessary to provide a web interface.
	//
	// If HTTPServer is nil, pprof will use its own internal HTTP server.
	//
	// A common use for a custom HTTPServer is to provide custom
	// authentication checks.
	HTTPServer    func(args *HTTPServerArgs) error
	HTTPTransport http.RoundTripper
}

// Writer provides a mechanism to write data under a certain name,
// typically a filename.
type Writer interface {
	Open(name string) (io.WriteCloser, error)
}

// A FlagSet creates and parses command-line flags.
// It is similar to the standard flag.FlagSet.
type FlagSet interface {
	// Bool, Int, Float64, and String define new flags,
	// like the functions of the same name in package flag.
	Bool(name string, def bool, usage string) *bool
	Int(name string, def int, usage string) *int
	Float64(name string, def float64, usage string) *float64
	String(name string, def string, usage string) *string

	// StringList is similar to String but allows multiple values for a
	// single flag
	StringList(name string, def string, usage string) *[]*string

	// ExtraUsage returns any additional text that should be printed after the
	// standard usage message. The extra usage message returned includes all text
	// added with AddExtraUsage().
	// The typical use of ExtraUsage is to show any custom flags defined by the
	// specific pprof plugins being used.
	ExtraUsage() string

	// AddExtraUsage appends additional text to the end of the extra usage message.
	AddExtraUsage(eu string)

	// Parse initializes the flags with their values for this run
	// and returns the non-flag command line arguments.
	// If an unknown flag is encountered or there are no arguments,
	// Parse should call usage and return nil.
	Parse(usage func()) []string
}

// A Fetcher reads and returns the profile named by src. src can be a
// local file path or a URL. duration and timeout are units specified
// by the end user, or 0 by default. duration refers to the length of
// the profile collection, if applicable, and timeout is the amount of
// time to wait for a profile before returning an error. Returns the
// fetched profile, the URL of the actual source of the profile, or an
// error.
type Fetcher interface {
	Fetch(src string, duration, timeout time.Duration) (*profile.Profile, string, error)
}

// A Symbolizer introduces symbol information into a profile.
type Symbolizer interface {
	Symbolize(mode string, srcs MappingSources, prof *profile.Profile) error
}

// MappingSources map each profile.Mapping to the source of the profile.
// The key is either Mapping.File or Mapping.BuildId.
type MappingSources map[string][]struct {
	Source string // URL of the source the mapping was collected from
	Start  uint64 // delta applied to addresses from this source (to represent Merge adjustments)
}

// An ObjTool inspects shared libraries and executable files.
type ObjTool interface {
	// Open opens the named object file. If the object is a shared
	// library, start/limit/offset are the addresses where it is mapped
	// into memory in the address space being inspected. If the object
	// is a linux kernel, relocationSymbol is the name of the symbol
	// corresponding to the start address.
	Open(file string, start, limit, offset uint64, relocationSymbol string) (ObjFile, error)

	// Disasm disassembles the named object file, starting at
	// the start address and stopping at (before) the end address.
	Disasm(file string, start, end uint64, intelSyntax bool) ([]Inst, error)
}

// An Inst is a single instruction in an assembly listing.
type Inst struct {
	Addr     uint64 // virtual address of instruction
	Text     string // instruction text
	Function string // function name
	File     string // source file
	Line     int    // source line
}

// An ObjFile is a single object file: a shared library or executable.
type ObjFile interface {
	// Name returns the underlyinf file name, if available
	Name() string

	// ObjAddr returns the objdump (linker) address corresponding to a runtime
	// address, and an error.
	ObjAddr(addr uint64) (uint64, error)

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

// A Frame describes a location in a single line in a source file.
type Frame struct {
	Func   string // name of function
	File   string // source file name
	Line   int    // line in file
	Column int    // column in line (if available)
}

// A Sym describes a single symbol in an object file.
type Sym struct {
	Name  []string // names of symbol (many if symbol was dedup'ed)
	File  string   // object file containing symbol
	Start uint64   // start virtual address
	End   uint64   // virtual address of last byte in sym (Start+size-1)
}

// A UI manages user interactions.
type UI interface {
	// Read returns a line of text (a command) read from the user.
	// prompt is printed before reading the command.
	ReadLine(prompt string) (string, error)

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

	// WantBrowser indicates whether a browser should be opened with the -http option.
	WantBrowser() bool

	// SetAutoComplete instructs the UI to call complete(cmd) to obtain
	// the auto-completion of cmd, if the UI supports auto-completion at all.
	SetAutoComplete(complete func(string) string)
}

// HTTPServerArgs contains arguments needed by an HTTP server that
// is exporting a pprof web interface.
type HTTPServerArgs struct {
	// Hostport contains the http server address (derived from flags).
	Hostport string

	Host string // Host portion of Hostport
	Port int    // Port portion of Hostport

	// Handlers maps from URL paths to the handler to invoke to
	// serve that path.
	Handlers map[string]http.Handler
}
