// Package imports implements a Go pretty-printer (like package "go/format")
// that also adds or removes import statements as necessary.
package imports // import "golang.org/x/tools/imports"

import (
	"io/ioutil"
	"log"

	"golang.org/x/tools/internal/gocommand"
	intimp "golang.org/x/tools/internal/imports"
)

// Options specifies options for processing files.
type Options struct {
	Fragment  bool // Accept fragment of a source file (no package statement)
	AllErrors bool // Report all errors (not just the first 10 on different lines)

	Comments  bool // Print comments (true if nil *Options provided)
	TabIndent bool // Use tabs for indent (true if nil *Options provided)
	TabWidth  int  // Tab width (8 if nil *Options provided)

	FormatOnly bool // Disable the insertion and deletion of imports
}

// Debug controls verbose logging.
var Debug = false

// LocalPrefix is a comma-separated string of import path prefixes, which, if
// set, instructs Process to sort the import paths with the given prefixes
// into another group after 3rd-party packages.
var LocalPrefix string

// Process formats and adjusts imports for the provided file.
// If opt is nil the defaults are used, and if src is nil the source
// is read from the filesystem.
//
// Note that filename's directory influences which imports can be chosen,
// so it is important that filename be accurate.
// To process data ``as if'' it were in filename, pass the data as a non-nil src.
func Process(filename string, src []byte, opt *Options) ([]byte, error) {
	var err error
	if src == nil {
		src, err = ioutil.ReadFile(filename)
		if err != nil {
			return nil, err
		}
	}
	if opt == nil {
		opt = &Options{Comments: true, TabIndent: true, TabWidth: 8}
	}
	intopt := &intimp.Options{
		Env: &intimp.ProcessEnv{
			GocmdRunner: &gocommand.Runner{},
		},
		LocalPrefix: LocalPrefix,
		AllErrors:   opt.AllErrors,
		Comments:    opt.Comments,
		FormatOnly:  opt.FormatOnly,
		Fragment:    opt.Fragment,
		TabIndent:   opt.TabIndent,
		TabWidth:    opt.TabWidth,
	}
	if Debug {
		intopt.Env.Logf = log.Printf
	}
	return intimp.Process(filename, src, intopt)
}

// VendorlessPath returns the devendorized version of the import path ipath.
// For example, VendorlessPath("foo/bar/vendor/a/b") returns "a/b".
func VendorlessPath(ipath string) string {
	return intimp.VendorlessPath(ipath)
}
