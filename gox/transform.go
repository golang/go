// Package gox transforms GoX source into ordinary Go.
package gox

import (
	"bytes"
	"fmt"
	"github.com/alfu32/go-shared/gox/internal/parser"
	"go/format"
	"go/printer"
	"go/token"
	"path/filepath"
)

type Options struct {
	BareJSX         bool
	Identity        string
	SourcePositions bool
	// HookBindings lists package-level names that shadow intrinsic hooks.
	HookBindings []string
}
type Result struct {
	Source []byte
	// SourceMap is populated when Options.SourcePositions is enabled.
	SourceMap []LineMapping
}

func TransformFile(filename string, src []byte, opts Options) (Result, error) {
	fs := token.NewFileSet()
	mode := parser.GoX | parser.ParseComments | parser.SkipObjectResolution | parser.AllErrors
	if opts.BareJSX {
		mode |= parser.BareJSX
	}
	identity := opts.Identity
	if identity == "" {
		identity = filepath.ToSlash(filename)
	}
	f, err := parser.ParseFileWithBindings(fs, filename, src, mode, identity, opts.HookBindings)
	if err != nil {
		return Result{}, err
	}
	var b bytes.Buffer
	pc := printer.Config{Mode: printer.UseSpaces | printer.TabIndent, Tabwidth: 8}
	if opts.SourcePositions {
		pc.Mode |= printer.SourcePos
	}
	if err = pc.Fprint(&b, fs, f); err != nil {
		return Result{}, fmt.Errorf("emit: %w", err)
	}
	out, err := format.Source(b.Bytes())
	if err != nil {
		return Result{}, fmt.Errorf("invalid generated Go: %w\n%s", err, b.Bytes())
	}
	out = protectCgoPreambles(out)
	r := Result{Source: out}
	if opts.SourcePositions {
		r.SourceMap = lineMappings(filename, out)
	}
	return r, nil
}
