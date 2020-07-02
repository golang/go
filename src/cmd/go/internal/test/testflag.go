// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/cmdflag"
	"cmd/go/internal/work"
)

//go:generate go run ./genflags.go

// The flag handling part of go test is large and distracting.
// We can't use (*flag.FlagSet).Parse because some of the flags from
// our command line are for us, and some are for the test binary, and
// some are for both.

func init() {
	work.AddBuildFlags(CmdTest, work.OmitVFlag)

	cf := CmdTest.Flag
	cf.BoolVar(&testC, "c", false, "")
	cf.BoolVar(&cfg.BuildI, "i", false, "")
	cf.StringVar(&testO, "o", "", "")

	cf.BoolVar(&testCover, "cover", false, "")
	cf.Var(coverFlag{(*coverModeFlag)(&testCoverMode)}, "covermode", "")
	cf.Var(coverFlag{commaListFlag{&testCoverPaths}}, "coverpkg", "")

	cf.Var((*base.StringsFlag)(&work.ExecCmd), "exec", "")
	cf.BoolVar(&testJSON, "json", false, "")
	cf.Var(&testVet, "vet", "")

	// Register flags to be forwarded to the test binary. We retain variables for
	// some of them so that cmd/go knows what to do with the test output, or knows
	// to build the test in a way that supports the use of the flag.

	cf.StringVar(&testBench, "bench", "", "")
	cf.Bool("benchmem", false, "")
	cf.String("benchtime", "", "")
	cf.StringVar(&testBlockProfile, "blockprofile", "", "")
	cf.String("blockprofilerate", "", "")
	cf.Int("count", 0, "")
	cf.Var(coverFlag{stringFlag{&testCoverProfile}}, "coverprofile", "")
	cf.String("cpu", "", "")
	cf.StringVar(&testCPUProfile, "cpuprofile", "", "")
	cf.Bool("failfast", false, "")
	cf.StringVar(&testList, "list", "", "")
	cf.StringVar(&testMemProfile, "memprofile", "", "")
	cf.String("memprofilerate", "", "")
	cf.StringVar(&testMutexProfile, "mutexprofile", "", "")
	cf.String("mutexprofilefraction", "", "")
	cf.Var(outputdirFlag{&testOutputDir}, "outputdir", "")
	cf.Int("parallel", 0, "")
	cf.String("run", "", "")
	cf.Bool("short", false, "")
	cf.DurationVar(&testTimeout, "timeout", 10*time.Minute, "")
	cf.StringVar(&testTrace, "trace", "", "")
	cf.BoolVar(&testV, "v", false, "")

	for name, _ := range passFlagToTest {
		cf.Var(cf.Lookup(name).Value, "test."+name, "")
	}
}

// A coverFlag is a flag.Value that also implies -cover.
type coverFlag struct{ v flag.Value }

func (f coverFlag) String() string { return f.v.String() }

func (f coverFlag) Set(value string) error {
	if err := f.v.Set(value); err != nil {
		return err
	}
	testCover = true
	return nil
}

type coverModeFlag string

func (f *coverModeFlag) String() string { return string(*f) }
func (f *coverModeFlag) Set(value string) error {
	switch value {
	case "", "set", "count", "atomic":
		*f = coverModeFlag(value)
		return nil
	default:
		return errors.New(`valid modes are "set", "count", or "atomic"`)
	}
}

// A commaListFlag is a flag.Value representing a comma-separated list.
type commaListFlag struct{ vals *[]string }

func (f commaListFlag) String() string { return strings.Join(*f.vals, ",") }

func (f commaListFlag) Set(value string) error {
	if value == "" {
		*f.vals = nil
	} else {
		*f.vals = strings.Split(value, ",")
	}
	return nil
}

// A stringFlag is a flag.Value representing a single string.
type stringFlag struct{ val *string }

func (f stringFlag) String() string { return *f.val }
func (f stringFlag) Set(value string) error {
	*f.val = value
	return nil
}

// outputdirFlag implements the -outputdir flag.
// It interprets an empty value as the working directory of the 'go' command.
type outputdirFlag struct {
	resolved *string
}

func (f outputdirFlag) String() string { return *f.resolved }
func (f outputdirFlag) Set(value string) (err error) {
	if value == "" {
		// The empty string implies the working directory of the 'go' command.
		*f.resolved = base.Cwd
	} else {
		*f.resolved, err = filepath.Abs(value)
	}
	return err
}

// vetFlag implements the special parsing logic for the -vet flag:
// a comma-separated list, with a distinguished value "off" and
// a boolean tracking whether it was set explicitly.
type vetFlag struct {
	explicit bool
	off      bool
	flags    []string // passed to vet when invoked automatically during 'go test'
}

func (f *vetFlag) String() string {
	if f.off {
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
	if value == "" {
		*f = vetFlag{flags: defaultVetFlags}
		return nil
	}

	if value == "off" {
		*f = vetFlag{
			explicit: true,
			off:      true,
		}
		return nil
	}

	if strings.Contains(value, "=") {
		return fmt.Errorf("-vet argument cannot contain equal signs")
	}
	if strings.Contains(value, " ") {
		return fmt.Errorf("-vet argument is comma-separated list, cannot contain spaces")
	}
	*f = vetFlag{explicit: true}
	for _, arg := range strings.Split(value, ",") {
		if arg == "" {
			return fmt.Errorf("-vet argument contains empty list element")
		}
		f.flags = append(f.flags, "-"+arg)
	}
	return nil
}

// testFlags processes the command line, grabbing -x and -c, rewriting known flags
// to have "test" before them, and reading the command line for the test binary.
// Unfortunately for us, we need to do our own flag processing because go test
// grabs some flags but otherwise its command line is just a holding place for
// pkg.test's arguments.
// We allow known flags both before and after the package name list,
// to allow both
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

	explicitArgs := make([]string, 0, len(args))
	inPkgList := false
	for len(args) > 0 {
		f, remainingArgs, err := cmdflag.ParseOne(&CmdTest.Flag, args)

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

		if nf := (cmdflag.NonFlagError{}); errors.As(err, &nf) {
			if !inPkgList && packageNames != nil {
				// We already saw the package list previously, and this argument is not
				// a flag, so it — and everything after it — must be a literal argument
				// to the test binary.
				explicitArgs = append(explicitArgs, args...)
				break
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

		if nd := (cmdflag.FlagNotDefinedError{}); errors.As(err, &nd) {
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

			explicitArgs = append(explicitArgs, nd.RawArg)
			args = remainingArgs
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

	var injectedFlags []string
	if testJSON {
		// If converting to JSON, we need the full output in order to pipe it to
		// test2json.
		injectedFlags = append(injectedFlags, "-test.v=true")
		delete(addFromGOFLAGS, "v")
		delete(addFromGOFLAGS, "test.v")
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
	if testProfile() != "" && !outputDirSet {
		injectedFlags = append(injectedFlags, "-test.outputdir="+testOutputDir)
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

	// Ensure that -race and -covermode are compatible.
	if testCoverMode == "" {
		testCoverMode = "set"
		if cfg.BuildRace {
			// Default coverage mode is atomic when -race is set.
			testCoverMode = "atomic"
		}
	}
	if cfg.BuildRace && testCoverMode != "atomic" {
		base.Fatalf(`-covermode must be "atomic", not %q, when -race is enabled`, testCoverMode)
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
