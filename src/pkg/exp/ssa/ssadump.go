// +build ignore

package main

// ssadump: a tool for displaying and interpreting the SSA form of Go programs.

import (
	"exp/ssa"
	"exp/ssa/interp"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"strings"
)

// TODO(adonovan): perhaps these should each be separate flags?
var buildFlag = flag.String("build", "", `Options controlling the SSA builder.
The value is a sequence of zero or more of these letters:
C	perform sanity [C]hecking of the SSA form.
P	log [P]ackage inventory.
F	log [F]unction SSA code.
S	log [S]ource locations as SSA builder progresses.
G	use binary object files from gc to provide imports (no code).
N	build [N]aive SSA form: don't replace local loads/stores with registers.
`)

var runFlag = flag.Bool("run", false, "Invokes the SSA interpreter on the program.")

var interpFlag = flag.String("interp", "", `Options controlling the SSA test interpreter.
The value is a sequence of zero or more more of these letters:
R	disable [R]ecover() from panic; show interpreter crash instead.
T	[T]race execution of the program.  Best for single-threaded programs!
`)

const usage = `SSA builder and interpreter.
Usage: ssadump [<flag> ...] <file.go> ...
Use -help flag to display options.

Examples:
% ssadump -run -interp=T hello.go     # interpret a program, with tracing
% ssadump -build=FPG hello.go         # quickly dump SSA form of a single package
`

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
	flag.Parse()
	args := flag.Args()

	// TODO(adonovan): perhaps we need a more extensible option
	// API than a bitset, e.g. a struct with a sane zero value?
	var mode ssa.BuilderMode
	for _, c := range *buildFlag {
		switch c {
		case 'P':
			mode |= ssa.LogPackages
		case 'F':
			mode |= ssa.LogFunctions
		case 'S':
			mode |= ssa.LogSource
		case 'C':
			mode |= ssa.SanityCheckFunctions
		case 'N':
			mode |= ssa.NaiveForm
		case 'G':
			mode |= ssa.UseGCImporter
		default:
			log.Fatalf("Unknown -build option: '%c'.", c)
		}
	}

	var interpMode interp.Mode
	for _, c := range *interpFlag {
		switch c {
		case 'T':
			interpMode |= interp.EnableTracing
		case 'R':
			interpMode |= interp.DisableRecover
		default:
			log.Fatalf("Unknown -interp option: '%c'.", c)
		}
	}

	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	// Treat all leading consecutive "*.go" arguments as a single package.
	//
	// TODO(gri): make it a typechecker error for there to be
	// duplicate (e.g.) main functions in the same package.
	var gofiles []string
	for len(args) > 0 && strings.HasSuffix(args[0], ".go") {
		gofiles = append(gofiles, args[0])
		args = args[1:]
	}
	if gofiles == nil {
		log.Fatal("No *.go source files specified.")
	}

	// Profiling support.
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// TODO(adonovan): permit naming a package directly instead of
	// a list of .go files.

	// TODO(adonovan/gri): the cascade of errors is confusing due
	// to reentrant control flow.  Disable for now and re-think.
	var errh func(error)
	// errh = func(err error) { fmt.Println(err.Error()) }

	b := ssa.NewBuilder(mode, ssa.GorootLoader, errh)
	files, err := ssa.ParseFiles(b.Prog.Files, ".", gofiles...)
	if err != nil {
		log.Fatalf(err.Error())
	}
	mainpkg, err := b.CreatePackage("main", files)
	if err != nil {
		log.Fatalf(err.Error())
	}
	b.BuildPackage(mainpkg)
	b = nil // discard Builder

	if *runFlag {
		interp.Interpret(mainpkg, interpMode, gofiles[0], args)
	}
}
