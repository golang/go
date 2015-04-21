// The gorename command performs precise type-safe renaming of
// identifiers in Go source code.
//
// Run with -help for usage information, or view the Usage constant in
// package golang.org/x/tools/refactor/rename, which contains most of
// the implementation.
//
package main // import "golang.org/x/tools/cmd/gorename"

import (
	"flag"
	"fmt"
	"go/build"
	"os"
	"runtime"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/refactor/rename"
)

var (
	offsetFlag = flag.String("offset", "", "file and byte offset of identifier to be renamed, e.g. 'file.go:#123'.  For use by editors.")
	fromFlag   = flag.String("from", "", "identifier to be renamed; see -help for formats")
	toFlag     = flag.String("to", "", "new name for identifier")
	helpFlag   = flag.Bool("help", false, "show usage message")
)

func init() {
	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)
	flag.BoolVar(&rename.Force, "force", false, "proceed, even if conflicts were reported")
	flag.BoolVar(&rename.DryRun, "dryrun", false, "show the change, but do not apply it")
	flag.BoolVar(&rename.Verbose, "v", false, "print verbose information")

	// If $GOMAXPROCS isn't set, use the full capacity of the machine.
	// For small machines, use at least 4 threads.
	if os.Getenv("GOMAXPROCS") == "" {
		n := runtime.NumCPU()
		if n < 4 {
			n = 4
		}
		runtime.GOMAXPROCS(n)
	}
}

func main() {
	flag.Parse()
	if len(flag.Args()) > 0 {
		fmt.Fprintln(os.Stderr, "gorename: surplus arguments.")
		os.Exit(1)
	}

	if *helpFlag || (*offsetFlag == "" && *fromFlag == "" && *toFlag == "") {
		fmt.Println(rename.Usage)
		return
	}

	if err := rename.Main(&build.Default, *offsetFlag, *fromFlag, *toFlag); err != nil {
		if err != rename.ConflictError {
			fmt.Fprintf(os.Stderr, "gorename: %s\n", err)
		}
		os.Exit(1)
	}
}
