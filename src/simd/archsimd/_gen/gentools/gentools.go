// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gentools provides shared helper utilities for Go code generator tools
// in archsimd.
//
// Basic usage:
//
//	func main() {
//	    gentools.RegisterFlags(nil)
//	    flag.Parse()
//
//	    var files gentools.Files
//	    defer files.FlushOrExit()
//
//	    buf := files.NewGoFile("src/simd/archsimd/ops_amd64.go")
//	    fmt.Fprintln(buf, "package archsimd")
//	    // ... write generated code to buf ...
//	}
//
// By default (when -w is not specified), gentools outputs all generated files
// as a txtar archive to standard output. Pass -w to write files directly into
// the Go source tree.
package gentools

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"go/scanner"
	"go/token"
	"internal/diff"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Options contains standard options and CLI flags for code generators.
type Options struct {
	GOROOT string // -goroot: root of the input Go source tree
	outDir string // -outdir: root of the output tree (defaults to GOROOT)
	Write  bool   // -w: write generated files to disk under GOROOT
	Diff   bool   // -diff: check if generated files match disk, print diffs if not
	Txtar  bool   // -txtar: write generated files to output as a txtar archive (default output mode)

	Output    io.Writer // output writer for txtar and diff mode; defaults to os.Stdout if nil
	ErrOutput io.Writer // error writer for formatting errors; defaults to os.Stderr if nil
}

var globalOptions *Options

// RegisterFlags registers standard generator flags with the provided FlagSet
// (or [flag.CommandLine] if fs is nil) and returns a pointer to the Options
// struct.
//
// If fs is nil, the returned options are remembered globally as defaults for
// zero-value Files instances.
func RegisterFlags(fs *flag.FlagSet) *Options {
	o := new(Options)
	if fs == nil {
		fs = flag.CommandLine
		globalOptions = o
	}
	defaultGOROOT := findGOROOT()
	fs.StringVar(&o.GOROOT, "goroot", defaultGOROOT, "source Go dev tree")
	fs.StringVar(&o.outDir, "outdir", "", "output directory (default: set to -goroot)")
	fs.BoolVar(&o.Write, "w", false, "write generated files directly to disk under -outdir")
	fs.BoolVar(&o.Diff, "diff", false, "compare generated files against disk and print unified diffs")
	fs.BoolVar(&o.Txtar, "txtar", false, "output generated files as a txtar archive to stdout (default mode)")
	return o
}

// InputPath resolves relPath relative to either o.OutDir/src, if that file
// exists, or o.GOROOT/src. In effect, o.OutDir is treated as an overlay on
// o.GOROOT.
func (o *Options) InputPath(relPath string) string {
	if o.outDir != o.GOROOT {
		path := o.OutputPath(relPath)
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return filepath.Join(o.GOROOT, "src", relPath)
}

// ReadFile reads relPath from either o.OutDir/src or o.GOROOT/src.
func (o *Options) ReadFile(relPath string) ([]byte, error) {
	return os.ReadFile(o.InputPath(relPath))
}

// OutputPath returns relPath relative to o.OutDir/src.
func (o *Options) OutputPath(relPath string) string {
	outDir := o.outDir
	if outDir == "" {
		outDir = o.GOROOT
	}
	return filepath.Join(outDir, "src", relPath)
}

type fileInfo struct {
	relPath string
	isGo    bool
	buf     bytes.Buffer
}

// Files manages a collection of generated files for a single generator run.
// The zero value of Files is ready for immediate use and automatically honors
// the command-line flags registered via RegisterFlags.
type Files struct {
	// Options optionally overrides the generator options for this Files instance.
	// If nil, the globally registered options from RegisterFlags are used automatically.
	Options *Options

	files []*fileInfo
}

func (f *Files) getOptions() Options {
	var opts Options
	if f != nil && f.Options != nil {
		opts = *f.Options
	} else if globalOptions != nil {
		opts = *globalOptions
	}

	if opts.GOROOT == "" {
		opts.GOROOT = findGOROOT()
	}
	if opts.Output == nil {
		opts.Output = os.Stdout
	}
	if opts.ErrOutput == nil {
		opts.ErrOutput = os.Stderr
	}
	if !(opts.Write || opts.Diff || opts.Txtar) {
		opts.Txtar = true
	}

	return opts
}

// NewGoFile registers a Go source file at relPath (relative to GOROOT/src). It
// returns a *bytes.Buffer for the generator to populate. During Flush(), Go
// files are formatted with go/format.
func (f *Files) NewGoFile(relPath string) *bytes.Buffer {
	info := &fileInfo{
		relPath: relPath,
		isGo:    true,
	}
	f.files = append(f.files, info)
	return &info.buf
}

// NewRawFile registers a non-Go file (e.g. .rules, YAML, txtar) at relPath
// (relative to GOROOT/src). It returns a *bytes.Buffer for the generator to
// populate. During Flush(), content is written directly without go/format.
func (f *Files) NewRawFile(relPath string) *bytes.Buffer {
	info := &fileInfo{
		relPath: relPath,
		isGo:    false,
	}
	f.files = append(f.files, info)
	return &info.buf
}

// Flush outputs all registered files according to the mode in options.
//
// In default / -txtar mode, it outputs files as a txtar archive to Output. In
// write mode (-w), it writes all files to disk under GOROOT. In diff mode
// (-diff), it compares generated content against disk, prints diffs to Output,
// and returns an error if out of date.
func (f *Files) Flush() error {
	opts := f.getOptions()

	if (opts.Write || opts.Diff) && opts.GOROOT == "" {
		return fmt.Errorf("GOROOT not found; pass -goroot flag")
	}

	type preparedFile struct {
		relPath string
		content []byte
	}

	prepared := make([]preparedFile, len(f.files))
	for i, fi := range f.files {
		raw := fi.buf.Bytes()
		var content []byte
		if fi.isGo {
			formatted, err := format.Source(raw)
			if err != nil {
				printFormattingError(opts.ErrOutput, fi.relPath, raw, err)
				return fmt.Errorf("error formatting %s: %w", fi.relPath, err)
			}
			content = formatted
		} else {
			content = raw
		}

		prepared[i] = preparedFile{
			relPath: fi.relPath,
			content: content,
		}
	}
	f.files = nil

	if opts.Diff {
		hasDiffs := false
		for _, pf := range prepared {
			onDisk, err := opts.ReadFile(pf.relPath)
			if err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("reading %s for diff: %w", pf.relPath, err)
			}
			srcPath := filepath.Join("src", pf.relPath)
			d := diff.Diff(srcPath, onDisk, srcPath, pf.content)
			if len(d) > 0 {
				hasDiffs = true
				opts.Output.Write(d)
			}
		}
		if hasDiffs {
			return fmt.Errorf("generated files differ from disk")
		}
	}

	if opts.Txtar {
		for i, pf := range prepared {
			if i > 0 {
				fmt.Fprintln(opts.Output)
			}
			srcPath := filepath.Join("src", pf.relPath)
			fmt.Fprintf(opts.Output, "-- %s --\n", srcPath)
			opts.Output.Write(pf.content)
			// Ensure trailing \n
			if len(pf.content) > 0 && !bytes.HasSuffix(pf.content, []byte("\n")) {
				fmt.Fprintln(opts.Output)
			}
		}
	}

	if opts.Write {
		for _, pf := range prepared {
			path := opts.OutputPath(pf.relPath)
			dir := filepath.Dir(path)
			if err := os.MkdirAll(dir, 0755); err != nil {
				return fmt.Errorf("creating directory %s: %w", dir, err)
			}
			if err := os.WriteFile(path, pf.content, 0644); err != nil {
				return fmt.Errorf("writing %s: %w", path, err)
			}
		}
	}

	return nil
}

// FlushOrExit calls Flush(), prints any error to stderr, and exits with code 1 if Flush fails.
//
// It is intended to be deferred at the beginning of main (e.g., `defer files.FlushOrExit()`).
// Hence, if invoked as part of a panic, it skips flushing and instead allows the panic to propagate.
func (f *Files) FlushOrExit() {
	if r := recover(); r != nil {
		panic(r)
	}
	if err := f.Flush(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

// printFormattingError prints err, with 10 lines of context around the error
// line and a caret mark ("^") to indicate the column offset of the error.
func printFormattingError(out io.Writer, relPath string, raw []byte, err error) {
	var pos token.Position
	if el, ok := err.(scanner.ErrorList); ok && len(el) > 0 {
		el.Sort()
		pos = el[0].Pos
	} else if e, ok := err.(*scanner.Error); ok {
		pos = e.Pos
	} else if e, ok := err.(scanner.Error); ok {
		pos = e.Pos
	}

	lines := strings.Split(string(raw), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if pos.Line <= 0 || pos.Line > len(lines) {
		fmt.Fprintf(out, "error formatting %s: %v\n", relPath, err)
		fmt.Fprintf(out, "%s\n", raw)
		return
	}

	startLine := max(pos.Line-5, 1)
	endLine := min(pos.Line+5, len(lines))

	for i := startLine; i <= endLine; i++ {
		line := lines[i-1]
		fmt.Fprintf(out, "%s\n", line)
		if i == pos.Line {
			var indent strings.Builder
			for _, ch := range line {
				pos.Column--
				if pos.Column == 0 {
					break
				}
				if ch == '\t' {
					indent.WriteByte('\t')
				} else {
					indent.WriteByte(' ')
				}
			}
			fmt.Fprintf(out, "%s^\n", indent.String())
			fmt.Fprintf(out, "%s\n", strings.TrimRight(err.Error(), "\n"))
		}
	}
}

func findGOROOT() string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	dir := cwd
	for {
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		if filepath.Base(dir) == "src" {
			if b, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil {
				for line := range strings.SplitSeq(string(b), "\n") {
					fields := strings.Fields(line)
					if len(fields) >= 2 && fields[0] == "module" && fields[1] == "std" {
						return parent
					}
				}
			}
		}
		dir = parent
	}
}

func resolvePath(goroot, relPath string) string {
	clean := cleanRelPath(relPath)
	if goroot == "" {
		return clean
	}
	return filepath.Join(goroot, clean)
}

func cleanRelPath(p string) string {
	p = strings.ReplaceAll(p, "\\", "/")
	return filepath.Join("src", p)
}
