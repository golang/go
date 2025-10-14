// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/cmdflag"
	"cmd/go/internal/work"
	"errors"
	"flag"
	"fmt"
	"internal/godebug"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

//go:generate go run ./genflags.go

// The flag handling part of go test is large and distracting.
// We can't use (*flag.FlagSet).Parse because some of the flags from
// our command line are for us, and some are for the test binary, and
// some are for both.

var gotestjsonbuildtext = godebug.New("gotestjsonbuildtext")

func init() {
	work.AddBuildFlags(CmdTest, work.OmitVFlag|work.OmitJSONFlag)

	cf := CmdTest.Flag
	cf.BoolVar(&testC, "c", false, "")
	cf.StringVar(&testO, "o", "", "")
	work.AddCoverFlags(CmdTest, &testCoverProfile)
	cf.Var((*base.StringsFlag)(&work.ExecCmd), "exec", "")
	cf.BoolVar(&testJSON, "json", false, "")
	cf.Var(&testVet, "vet", "")

	// Register flags to be forwarded to the test binary. We retain variables for
	// some of them so that cmd/go knows what to do with the test output, or knows
	// to build the test in a way that supports the use of the flag.

	cf.BoolVar(&testArtifacts, "artifacts", false, "")
	cf.StringVar(&testBench, "bench", "", "")
	cf.Bool("benchmem", false, "")
	cf.String("benchtime", "", "")
	cf.StringVar(&testBlockProfile, "blockprofile", "", "")
	cf.String("blockprofilerate", "", "")
	cf.Int("count", 0, "")
	cf.String("cpu", "", "")
	cf.StringVar(&testCPUProfile, "cpuprofile", "", "")
	cf.BoolVar(&testFailFast, "failfast", false, "")
	cf.StringVar(&testFuzz, "fuzz", "", "")
	cf.Bool("fullpath", false, "")
	cf.StringVar(&testList, "list", "", "")
	cf.StringVar(&testMemProfile, "memprofile", "", "")
	cf.String("memprofilerate", "", "")
	cf.StringVar(&testMutexProfile, "mutexprofile", "", "")
	cf.String("mutexprofilefraction", "", "")
	cf.Var(&testOutputDir, "outputdir", "")
	cf.Int("parallel", 0, "")
	cf.String("run", "", "")
	cf.Bool("short", false, "")
	cf.String("skip", "", "")
	cf.DurationVar(&testTimeout, "timeout", 10*time.Minute, "") // known to cmd/dist
	cf.String("fuzztime", "", "")
	cf.String("fuzzminimizetime", "", "")
	cf.StringVar(&testTrace, "trace", "", "")
	cf.Var(&testV, "v", "")
	cf.Var(&testShuffle, "shuffle", "")

	for name, ok := range passFlagToTest {
		if ok {
			cf.Var(cf.Lookup(name).Value, "test."+name, "")
		}
	}
}

// outputdirFlag implements the -outputdir flag.
// It interprets an empty value as the working directory of the 'go' command.
type outputdirFlag struct {
	abs string
}

func (f *outputdirFlag) String() string {
	return f.abs
}

func (f *outputdirFlag) Set(value string) (err error) {
	if value == "" {
		f.abs = ""
	} else {
		f.abs, err = filepath.Abs(value)
	}
	return err
}

func (f *outputdirFlag) getAbs() string {
	if f.abs == "" {
		return base.Cwd()
	}
	return f.abs
}

// vetFlag implements the special parsing logic for the -vet flag:
// a comma-separated list, with distinguished values "all" and
// "off", plus a boolean tracking whether it was set explicitly.
//
// "all" is encoded as vetFlag{true, false, nil}, since it will
// pass no flags to the vet binary, and by default, it runs all
// analyzers.
type vetFlag struct {
	explicit bool
	off      bool
	flags    []string // passed to vet when invoked automatically during 'go test'
}

func (f *vetFlag) String() string {
	switch {
	case !f.off && !f.explicit && len(f.flags) == 0:
		return "all"
	case f.off:
		return "off"
	}

	var buf strings.Builder
	for i, f := range f.flags {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(f)
	}
	return buf.String()
}

func (f *vetFlag) Set(value string) error {
	switch {
	case value == "":
		*f = vetFlag{flags: defaultVetFlags}
		return nil
	case strings.Contains(value, "="):
		return fmt.Errorf("-vet argument cannot contain equal signs")
	case strings.Contains(value, " "):
		return fmt.Errorf("-vet argument is comma-separated list, cannot contain spaces")
	}

	*f = vetFlag{explicit: true}
	var single string
	for arg := range strings.SplitSeq(value, ",") {
		switch arg {
		case "":
			return fmt.Errorf("-vet argument contains empty list element")
		case "all":
			single = arg
			*f = vetFlag{explicit: true}
			continue
		case "off":
			single = arg
			*f = vetFlag{
				explicit: true,
				off:      true,
			}
			continue
		default:
			if _, ok := passAnalyzersToVet[arg]; !ok {
				return fmt.Errorf("-vet argument must be a supported analyzer or a distinguished value; found %s", arg)
			}
			f.flags = append(f.flags, "-"+arg)
		}
	}
	if len(f.flags) > 1 && single != "" {
		return fmt.Errorf("-vet does not accept %q in a list with other analyzers", single)
	}
	if len(f.flags) > 1 && single != "" {
		return fmt.Errorf("-vet does not accept %q in a list with other analyzers", single)
	}
	return nil
}

type shuffleFlag struct {
	on   bool
	seed *int64
}

func (f *shuffleFlag) String() string {
	if !f.on {
		return "off"
	}
	if f.seed == nil {
		return "on"
	}
	return fmt.Sprintf("%d", *f.seed)
}

func (f *shuffleFlag) Set(value string) error {
	if value == "off" {
		*f = shuffleFlag{on: false}
		return nil
	}

	if value == "on" {
		*f = shuffleFlag{on: true}
		return nil
	}

	seed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return fmt.Errorf(`-shuffle argument must be "on", "off", or an int64: %v`, err)
	}

	*f = shuffleFlag{on: true, seed: &seed}
	return nil
}

// testFlags processes the command line, grabbing -x and -c, rewriting known flags
// to have "test" before them, and reading the command line for the test binary.
// Unfortunately for us, we need to do our own flag processing because go test
// grabs some flags but otherwise its command line is just a holding place for
// pkg.test's arguments.
// We allow known flags both before and after the package name list,
// to allow both
//
//	go test fmt -custom-flag-for-fmt-test
//	go test -x math
func testFlags(args []string) (packageNames, passToTest []string) {
	base.SetFromGOFLAGS(&CmdTest.Flag)
	addFromGOFLAGS := map[string]bool{}
	CmdTest.Flag.Visit(func(f *flag.Flag) {
		if short := strings.TrimPrefix(f.Name, "test."); passFlagToTest[short] {
			addFromGOFLAGS[f.Name] = true
		}
	})

	// firstUnknownFlag helps us report an error when flags not known to 'go
	// test' are used along with -i or -c.
	firstUnknownFlag := ""

	explicitArgs := make([]string, 0, len(args))
	inPkgList := false
	afterFlagWithoutValue := false
	for len(args) > 0 {
		f, remainingArgs, err := cmdflag.ParseOne(&CmdTest.Flag, args)

		wasAfterFlagWithoutValue := afterFlagWithoutValue
		afterFlagWithoutValue = false // provisionally

		if errors.Is(err, flag.ErrHelp) {
			exitWithUsage()
		}

		if errors.Is(err, cmdflag.ErrFlagTerminator) {
			// 'go list' allows package arguments to be named either before or after
			// the terminator, but 'go test' has historically allowed them only
			// before. Preserve that behavior and treat all remaining arguments —
			// including the terminator itself! — as arguments to the test.
			explicitArgs = append(explicitArgs, args...)
			break
		}

		if nf, ok := errors.AsType[cmdflag.NonFlagError](err); ok {
			if !inPkgList && packageNames != nil {
				// We already saw the package list previously, and this argument is not
				// a flag, so it — and everything after it — must be either a value for
				// a preceding flag or a literal argument to the test binary.
				if wasAfterFlagWithoutValue {
					// This argument could syntactically be a flag value, so
					// optimistically assume that it is and keep looking for go command
					// flags after it.
					//
					// (If we're wrong, we'll at least be consistent with historical
					// behavior; see https://golang.org/issue/40763.)
					explicitArgs = append(explicitArgs, nf.RawArg)
					args = remainingArgs
					continue
				} else {
					// This argument syntactically cannot be a flag value, so it must be a
					// positional argument, and so must everything after it.
					explicitArgs = append(explicitArgs, args...)
					break
				}
			}

			inPkgList = true
			packageNames = append(packageNames, nf.RawArg)
			args = remainingArgs // Consume the package name.
			continue
		}

		if inPkgList {
			// This argument is syntactically a flag, so if we were in the package
			// list we're not anymore.
			inPkgList = false
		}

		if nd, ok := errors.AsType[cmdflag.FlagNotDefinedError](err); ok {
			// This is a flag we do not know. We must assume that any args we see
			// after this might be flag arguments, not package names, so make
			// packageNames non-nil to indicate that the package list is complete.
			//
			// (Actually, we only strictly need to assume that if the flag is not of
			// the form -x=value, but making this more precise would be a breaking
			// change in the command line API.)
			if packageNames == nil {
				packageNames = []string{}
			}

			if nd.RawArg == "-args" || nd.RawArg == "--args" {
				// -args or --args signals that everything that follows
				// should be passed to the test.
				explicitArgs = append(explicitArgs, remainingArgs...)
				break
			}

			if firstUnknownFlag == "" {
				firstUnknownFlag = nd.RawArg
			}

			explicitArgs = append(explicitArgs, nd.RawArg)
			args = remainingArgs
			if !nd.HasValue {
				afterFlagWithoutValue = true
			}
			continue
		}

		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			exitWithUsage()
		}

		if short := strings.TrimPrefix(f.Name, "test."); passFlagToTest[short] {
			explicitArgs = append(explicitArgs, fmt.Sprintf("-test.%s=%v", short, f.Value))

			// This flag has been overridden explicitly, so don't forward its implicit
			// value from GOFLAGS.
			delete(addFromGOFLAGS, short)
			delete(addFromGOFLAGS, "test."+short)
		}

		args = remainingArgs
	}
	if firstUnknownFlag != "" && testC {
		fmt.Fprintf(os.Stderr, "go: unknown flag %s cannot be used with -c\n", firstUnknownFlag)
		exitWithUsage()
	}

	var injectedFlags []string
	if testJSON {
		// If converting to JSON, we need the full output in order to pipe it to test2json.
		// The -test.v=test2json flag is like -test.v=true but causes the test to add
		// extra ^V characters before testing output lines and other framing,
		// which helps test2json do a better job creating the JSON events.
		injectedFlags = append(injectedFlags, "-test.v=test2json")
		delete(addFromGOFLAGS, "v")
		delete(addFromGOFLAGS, "test.v")

		if gotestjsonbuildtext.Value() == "1" {
			gotestjsonbuildtext.IncNonDefault()
		} else {
			cfg.BuildJSON = true
		}
	}

	// Inject flags from GOFLAGS before the explicit command-line arguments.
	// (They must appear before the flag terminator or first non-flag argument.)
	// Also determine whether flags with awkward defaults have already been set.
	var timeoutSet, outputDirSet bool
	CmdTest.Flag.Visit(func(f *flag.Flag) {
		short := strings.TrimPrefix(f.Name, "test.")
		if addFromGOFLAGS[f.Name] {
			injectedFlags = append(injectedFlags, fmt.Sprintf("-test.%s=%v", short, f.Value))
		}
		switch short {
		case "timeout":
			timeoutSet = true
		case "outputdir":
			outputDirSet = true
		}
	})

	// 'go test' has a default timeout, but the test binary itself does not.
	// If the timeout wasn't set (and forwarded) explicitly, add the default
	// timeout to the command line.
	if testTimeout > 0 && !timeoutSet {
		injectedFlags = append(injectedFlags, fmt.Sprintf("-test.timeout=%v", testTimeout))
	}

	// Similarly, the test binary defaults -test.outputdir to its own working
	// directory, but 'go test' defaults it to the working directory of the 'go'
	// command. Set it explicitly if it is needed due to some other flag that
	// requests output.
	needOutputDir := testProfile() != "" || testArtifacts
	if needOutputDir && !outputDirSet {
		injectedFlags = append(injectedFlags, "-test.outputdir="+testOutputDir.getAbs())
	}

	// If the user is explicitly passing -help or -h, show output
	// of the test binary so that the help output is displayed
	// even though the test will exit with success.
	// This loop is imperfect: it will do the wrong thing for a case
	// like -args -test.outputdir -help. Such cases are probably rare,
	// and getting this wrong doesn't do too much harm.
helpLoop:
	for _, arg := range explicitArgs {
		switch arg {
		case "--":
			break helpLoop
		case "-h", "-help", "--help":
			testHelp = true
			break helpLoop
		}
	}

	// Forward any unparsed arguments (following --args) to the test binary.
	return packageNames, append(injectedFlags, explicitArgs...)
}

func exitWithUsage() {
	fmt.Fprintf(os.Stderr, "usage: %s\n", CmdTest.UsageLine)
	fmt.Fprintf(os.Stderr, "Run 'go help %s' and 'go help %s' for details.\n", CmdTest.LongName(), HelpTestflag.LongName())

	base.SetExitStatus(2)
	base.Exit()
}
