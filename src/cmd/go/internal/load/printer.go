// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmd/go/internal/base"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

// A Printer reports output about a Package.
type Printer interface {
	// Output reports output from building pkg. The arguments are of the form
	// expected by fmt.Print.
	//
	// pkg may be nil if this output is not associated with the build of a
	// particular package.
	//
	// The caller is responsible for checking if printing output is appropriate,
	// for example by checking cfg.BuildN or cfg.BuildV.
	Output(pkg *Package, args ...any)

	// Errorf prints output in the form of `log.Errorf` and reports that
	// building pkg failed.
	//
	// This ensures the output is terminated with a new line if there's any
	// output, but does not do any other formatting. Callers should generally
	// use a higher-level output abstraction, such as (*Shell).reportCmd.
	//
	// pkg may be nil if this output is not associated with the build of a
	// particular package.
	//
	// This sets the process exit status to 1.
	Errorf(pkg *Package, format string, args ...any)
}

// DefaultPrinter returns the default Printer.
func DefaultPrinter() Printer {
	return defaultPrinter()
}

var defaultPrinter = sync.OnceValue(func() Printer {
	// TODO: This will return a JSON printer once that's an option.
	return &TextPrinter{os.Stderr}
})

func ensureNewline(s string) string {
	if s == "" {
		return ""
	}
	if strings.HasSuffix(s, "\n") {
		return s
	}
	return s + "\n"
}

// A TextPrinter emits text format output to Writer.
type TextPrinter struct {
	Writer io.Writer
}

func (p *TextPrinter) Output(_ *Package, args ...any) {
	fmt.Fprint(p.Writer, args...)
}

func (p *TextPrinter) Errorf(_ *Package, format string, args ...any) {
	fmt.Fprint(p.Writer, ensureNewline(fmt.Sprintf(format, args...)))
	base.SetExitStatus(1)
}
