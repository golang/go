// The eg command performs example-based refactoring.
// For documentation, run the command, or see Help in
// golang.org/x/tools/refactor/eg.
package main // import "golang.org/x/tools/cmd/eg"

import (
	"flag"
	"fmt"
	"go/build"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"strings"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/refactor/eg"
)

var (
	beforeeditFlag = flag.String("beforeedit", "", "A command to exec before each file is edited (e.g. chmod, checkout).  Whitespace delimits argument words.  The string '{}' is replaced by the file name.")
	helpFlag       = flag.Bool("help", false, "show detailed help message")
	templateFlag   = flag.String("t", "", "template.go file specifying the refactoring")
	transitiveFlag = flag.Bool("transitive", false, "apply refactoring to all dependencies too")
	writeFlag      = flag.Bool("w", false, "rewrite input files in place (by default, the results are printed to standard output)")
	verboseFlag    = flag.Bool("v", false, "show verbose matcher diagnostics")
)

func init() {
	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)
}

const usage = `eg: an example-based refactoring tool.

Usage: eg -t template.go [-w] [-transitive] <args>...

-help            show detailed help message
-t template.go	 specifies the template file (use -help to see explanation)
-w          	 causes files to be re-written in place.
-transitive 	 causes all dependencies to be refactored too.
-v               show verbose matcher diagnostics
-beforeedit cmd  a command to exec before each file is modified.
                 "{}" represents the name of the file.
` + loader.FromArgsUsage

func main() {
	if err := doMain(); err != nil {
		fmt.Fprintf(os.Stderr, "eg: %s\n", err)
		os.Exit(1)
	}
}

func doMain() error {
	flag.Parse()
	args := flag.Args()

	if *helpFlag {
		fmt.Fprint(os.Stderr, eg.Help)
		os.Exit(2)
	}

	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	if *templateFlag == "" {
		return fmt.Errorf("no -t template.go file specified")
	}

	conf := loader.Config{
		Fset:       token.NewFileSet(),
		ParserMode: parser.ParseComments,
	}

	// The first Created package is the template.
	conf.CreateFromFilenames("template", *templateFlag)

	if _, err := conf.FromArgs(args, true); err != nil {
		return err
	}

	// Load, parse and type-check the whole program.
	iprog, err := conf.Load()
	if err != nil {
		return err
	}

	// Analyze the template.
	template := iprog.Created[0]
	xform, err := eg.NewTransformer(iprog.Fset, template.Pkg, template.Files[0], &template.Info, *verboseFlag)
	if err != nil {
		return err
	}

	// Apply it to the input packages.
	var pkgs []*loader.PackageInfo
	if *transitiveFlag {
		for _, info := range iprog.AllPackages {
			pkgs = append(pkgs, info)
		}
	} else {
		pkgs = iprog.InitialPackages()
	}
	var hadErrors bool
	for _, pkg := range pkgs {
		if pkg == template {
			continue
		}
		for _, file := range pkg.Files {
			n := xform.Transform(&pkg.Info, pkg.Pkg, file)
			if n == 0 {
				continue
			}
			filename := iprog.Fset.File(file.Pos()).Name()
			fmt.Fprintf(os.Stderr, "=== %s (%d matches)\n", filename, n)
			if *writeFlag {
				// Run the before-edit command (e.g. "chmod +w",  "checkout") if any.
				if *beforeeditFlag != "" {
					args := strings.Fields(*beforeeditFlag)
					// Replace "{}" with the filename, like find(1).
					for i := range args {
						if i > 0 {
							args[i] = strings.Replace(args[i], "{}", filename, -1)
						}
					}
					cmd := exec.Command(args[0], args[1:]...)
					cmd.Stdout = os.Stdout
					cmd.Stderr = os.Stderr
					if err := cmd.Run(); err != nil {
						fmt.Fprintf(os.Stderr, "Warning: edit hook %q failed (%s)\n",
							args, err)
					}
				}
				if err := eg.WriteAST(iprog.Fset, filename, file); err != nil {
					fmt.Fprintf(os.Stderr, "eg: %s\n", err)
					hadErrors = true
				}
			} else {
				format.Node(os.Stdout, iprog.Fset, file)
			}
		}
	}
	if hadErrors {
		os.Exit(1)
	}
	return nil
}
