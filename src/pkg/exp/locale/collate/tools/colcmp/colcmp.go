// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"exp/norm"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"text/template"
	"time"
)

var (
	doNorm  = flag.Bool("norm", false, "normalize input strings")
	cases   = flag.Bool("case", false, "generate case variants")
	verbose = flag.Bool("verbose", false, "print results")
	debug   = flag.Bool("debug", false, "output debug information")
	locale  = flag.String("locale", "en_US", "the locale to use. May be a comma-separated list for some commands.")
	col     = flag.String("col", "go", "collator to test")
	gold    = flag.String("gold", "go", "collator used as the gold standard")
	usecmp  = flag.Bool("usecmp", false,
		`use comparison instead of sort keys when sorting.  Must be "test", "gold" or "both"`)
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
	exclude    = flag.String("exclude", "", "exclude errors that contain any of the characters")
	limit      = flag.Int("limit", 5000000, "maximum number of samples to generate for one run")
)

func failOnError(err error) {
	if err != nil {
		log.Panic(err)
	}
}

// Test holds test data for testing a locale-collator pair.
// Test also provides functionality that is commonly used by the various commands.
type Test struct {
	ctxt    *Context
	Name    string
	Locale  string
	ColName string

	Col        Collator
	UseCompare bool

	Input    []Input
	Duration time.Duration

	start time.Time
	msg   string
	count int
}

func (t *Test) clear() {
	t.Col = nil
	t.Input = nil
}

const (
	msgGeneratingInput = "generating input"
	msgGeneratingKeys  = "generating keys"
	msgSorting         = "sorting"
)

var lastLen = 0

func (t *Test) SetStatus(msg string) {
	if *debug || *verbose {
		fmt.Printf("%s: %s...\n", t.Name, msg)
	} else if t.ctxt.out != nil {
		fmt.Fprint(t.ctxt.out, strings.Repeat(" ", lastLen))
		fmt.Fprint(t.ctxt.out, strings.Repeat("\b", lastLen))
		fmt.Fprint(t.ctxt.out, msg, "...")
		lastLen = len(msg) + 3
		fmt.Fprint(t.ctxt.out, strings.Repeat("\b", lastLen))
	}
}

// Start is used by commands to signal the start of an operation.
func (t *Test) Start(msg string) {
	t.SetStatus(msg)
	t.count = 0
	t.msg = msg
	t.start = time.Now()
}

// Stop is used by commands to signal the end of an operation.
func (t *Test) Stop() (time.Duration, int) {
	d := time.Now().Sub(t.start)
	t.Duration += d
	if *debug || *verbose {
		fmt.Printf("%s: %s done. (%.3fs /%dK ops)\n", t.Name, t.msg, d.Seconds(), t.count/1000)
	}
	return d, t.count
}

// generateKeys generates sort keys for all the inputs.
func (t *Test) generateKeys() {
	for i, s := range t.Input {
		b := t.Col.Key(s)
		t.Input[i].key = b
		if *debug {
			fmt.Printf("%s (%X): %X\n", string(s.UTF8), s.UTF16, b)
		}
	}
}

// Sort sorts the inputs. It generates sort keys if this is required by the
// chosen sort method.
func (t *Test) Sort() (tkey, tsort time.Duration, nkey, nsort int) {
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		failOnError(err)
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	if t.UseCompare || t.Col.Key(t.Input[0]) == nil {
		t.Start(msgSorting)
		sort.Sort(&testCompare{*t})
		tsort, nsort = t.Stop()
	} else {
		t.Start(msgGeneratingKeys)
		t.generateKeys()
		t.count = len(t.Input)
		tkey, nkey = t.Stop()
		t.Start(msgSorting)
		sort.Sort(t)
		tsort, nsort = t.Stop()
	}
	return
}

func (t *Test) Swap(a, b int) {
	t.Input[a], t.Input[b] = t.Input[b], t.Input[a]
}

func (t *Test) Less(a, b int) bool {
	t.count++
	return bytes.Compare(t.Input[a].key, t.Input[b].key) == -1
}

func (t Test) Len() int {
	return len(t.Input)
}

type testCompare struct {
	Test
}

func (t *testCompare) Less(a, b int) bool {
	t.count++
	return t.Col.Compare(t.Input[a], t.Input[b]) == -1
}

type testRestore struct {
	Test
}

func (t *testRestore) Less(a, b int) bool {
	return t.Input[a].index < t.Input[b].index
}

// GenerateInput generates input phrases for the locale tested by t.
func (t *Test) GenerateInput() {
	t.Input = nil
	if t.ctxt.lastLocale != t.Locale {
		gen := phraseGenerator{}
		gen.init(t.Locale)
		t.SetStatus(msgGeneratingInput)
		t.ctxt.lastInput = nil // allow the previous value to be garbage collected.
		t.Input = gen.generate(*doNorm)
		t.ctxt.lastInput = t.Input
		t.ctxt.lastLocale = t.Locale
	} else {
		t.Input = t.ctxt.lastInput
		for i := range t.Input {
			t.Input[i].key = nil
		}
		sort.Sort(&testRestore{*t})
	}
}

// Context holds all tests and settings translated from command line options.
type Context struct {
	test []*Test
	last *Test

	lastLocale string
	lastInput  []Input

	out io.Writer
}

func (ts *Context) Printf(format string, a ...interface{}) {
	ts.assertBuf()
	fmt.Fprintf(ts.out, format, a...)
}

func (ts *Context) Print(a ...interface{}) {
	ts.assertBuf()
	fmt.Fprint(ts.out, a...)
}

// assertBuf sets up an io.Writer for ouput, if it doesn't already exist.
// In debug and verbose mode, output is buffered so that the regular output
// will not interfere with the additional output.  Otherwise, output is
// written directly to stdout for a more responsive feel.
func (ts *Context) assertBuf() {
	if ts.out != nil {
		return
	}
	if *debug || *verbose {
		ts.out = &bytes.Buffer{}
	} else {
		ts.out = os.Stdout
	}
}

// flush flushes the contents of ts.out to stdout, if it is not stdout already.
func (ts *Context) flush() {
	if ts.out != nil {
		if _, ok := ts.out.(io.ReadCloser); !ok {
			io.Copy(os.Stdout, ts.out.(io.Reader))
		}
	}
}

// parseTests creates all tests from command lines and returns
// a Context to hold them.
func parseTests() *Context {
	ctxt := &Context{}
	colls := strings.Split(*col, ",")
	for _, loc := range strings.Split(*locale, ",") {
		loc = strings.TrimSpace(loc)
		for _, name := range colls {
			name = strings.TrimSpace(name)
			col := getCollator(name, loc)
			ctxt.test = append(ctxt.test, &Test{
				ctxt:       ctxt,
				Locale:     loc,
				ColName:    name,
				UseCompare: *usecmp,
				Col:        col,
			})
		}
	}
	return ctxt
}

func (c *Context) Len() int {
	return len(c.test)
}

func (c *Context) Test(i int) *Test {
	if c.last != nil {
		c.last.clear()
	}
	c.last = c.test[i]
	return c.last
}

func parseInput(args []string) []Input {
	input := []Input{}
	for _, s := range args {
		rs := []rune{}
		for len(s) > 0 {
			var r rune
			r, _, s, _ = strconv.UnquoteChar(s, '\'')
			rs = append(rs, r)
		}
		s = string(rs)
		if *doNorm {
			s = norm.NFC.String(s)
		}
		input = append(input, makeInputString(s))
	}
	return input
}

// A Command is an implementation of a colcmp command.
type Command struct {
	Run   func(cmd *Context, args []string)
	Usage string
	Short string
	Long  string
}

func (cmd Command) Name() string {
	return strings.SplitN(cmd.Usage, " ", 2)[0]
}

var commands = []*Command{
	cmdSort,
	cmdBench,
	cmdRegress,
}

const sortHelp = `
Sort sorts a given list of strings.  Strings are separated by whitespace.
`

var cmdSort = &Command{
	Run:   runSort,
	Usage: "sort <string>*",
	Short: "sort a given list of strings",
	Long:  sortHelp,
}

func runSort(ctxt *Context, args []string) {
	input := parseInput(args)
	if len(input) == 0 {
		log.Fatalf("Nothing to sort.")
	}
	if ctxt.Len() > 1 {
		ctxt.Print("COLL  LOCALE RESULT\n")
	}
	for i := 0; i < ctxt.Len(); i++ {
		t := ctxt.Test(i)
		t.Input = append(t.Input, input...)
		t.Sort()
		if ctxt.Len() > 1 {
			ctxt.Printf("%-5s %-5s  ", t.ColName, t.Locale)
		}
		for _, s := range t.Input {
			ctxt.Print(string(s.UTF8), " ")
		}
		ctxt.Print("\n")
	}
}

const benchHelp = `
Bench runs a benchmark for the given list of collator implementations.
If no collator implementations are given, the go collator will be used.
`

var cmdBench = &Command{
	Run:   runBench,
	Usage: "bench",
	Short: "benchmark a given list of collator implementations",
	Long:  benchHelp,
}

func runBench(ctxt *Context, args []string) {
	ctxt.Printf("%-7s %-5s %-6s %-24s %-24s %-5s %s\n", "LOCALE", "COLL", "N", "KEYS", "SORT", "AVGLN", "TOTAL")
	for i := 0; i < ctxt.Len(); i++ {
		t := ctxt.Test(i)
		ctxt.Printf("%-7s %-5s ", t.Locale, t.ColName)
		t.GenerateInput()
		ctxt.Printf("%-6s ", fmt.Sprintf("%dK", t.Len()/1000))
		tkey, tsort, nkey, nsort := t.Sort()
		p := func(dur time.Duration, n int) {
			s := ""
			if dur > 0 {
				s = fmt.Sprintf("%6.3fs ", dur.Seconds())
				if n > 0 {
					s += fmt.Sprintf("%15s", fmt.Sprintf("(%4.2f ns/op)", float64(dur)/float64(n)))
				}
			}
			ctxt.Printf("%-24s ", s)
		}
		p(tkey, nkey)
		p(tsort, nsort)

		total := 0
		for _, s := range t.Input {
			total += len(s.key)
		}
		ctxt.Printf("%-5d ", total/t.Len())
		ctxt.Printf("%6.3fs\n", t.Duration.Seconds())
		if *debug {
			for _, s := range t.Input {
				fmt.Print(string(s.UTF8), " ")
			}
			fmt.Println()
		}
	}
}

const regressHelp = `
Regress runs a monkey test by comparing the results of randomly generated tests
between two implementations of a collator. The user may optionally pass a list
of strings to regress against instead of the default test set.
`

var cmdRegress = &Command{
	Run:   runRegress,
	Usage: "regress -gold=<col> -test=<col> [string]*",
	Short: "run a monkey test between two collators",
	Long:  regressHelp,
}

const failedKeyCompare = `
%s:%d: incorrect comparison result for input:
    a:   %q (%.4X)
    key: %s
    b:   %q (%.4X)
    key: %s
    Compare(a, b) = %d; want %d.

  gold keys:
	a:   %s
	b:   %s
`

const failedCompare = `
%s:%d: incorrect comparison result for input:
    a:   %q (%.4X)
    b:   %q (%.4X)
    Compare(a, b) = %d; want %d.
`

func keyStr(b []byte) string {
	buf := &bytes.Buffer{}
	for _, v := range b {
		fmt.Fprintf(buf, "%.2X ", v)
	}
	return buf.String()
}

func runRegress(ctxt *Context, args []string) {
	input := parseInput(args)
	for i := 0; i < ctxt.Len(); i++ {
		t := ctxt.Test(i)
		if len(input) > 0 {
			t.Input = append(t.Input, input...)
		} else {
			t.GenerateInput()
		}
		t.Sort()
		count := 0
		gold := getCollator(*gold, t.Locale)
		for i := 1; i < len(t.Input); i++ {
			ia := t.Input[i-1]
			ib := t.Input[i]
			if bytes.IndexAny(ib.UTF8, *exclude) != -1 {
				i++
				continue
			}
			if bytes.IndexAny(ia.UTF8, *exclude) != -1 {
				continue
			}
			goldCmp := gold.Compare(ia, ib)
			if cmp := bytes.Compare(ia.key, ib.key); cmp != goldCmp {
				count++
				a := string(ia.UTF8)
				b := string(ib.UTF8)
				fmt.Printf(failedKeyCompare, t.Locale, i-1, a, []rune(a), keyStr(ia.key), b, []rune(b), keyStr(ib.key), cmp, goldCmp, keyStr(gold.Key(ia)), keyStr(gold.Key(ib)))
			} else if cmp := t.Col.Compare(ia, ib); cmp != goldCmp {
				count++
				a := string(ia.UTF8)
				b := string(ib.UTF8)
				fmt.Printf(failedCompare, t.Locale, i-1, a, []rune(a), b, []rune(b), cmp, goldCmp)
			}
		}
		if count > 0 {
			ctxt.Printf("Found %d inconsistencies in %d entries.\n", count, t.Len()-1)
		}
	}
}

const helpTemplate = `
colcmp is a tool for testing and benchmarking collation

Usage: colcmp command [arguments]

The commands are:
{{range .}}
    {{.Name | printf "%-11s"}} {{.Short}}{{end}}

Use "col help [topic]" for more information about that topic.
`

const detailedHelpTemplate = `
Usage: colcmp {{.Usage}}

{{.Long | trim}}
`

func runHelp(args []string) {
	t := template.New("help")
	t.Funcs(template.FuncMap{"trim": strings.TrimSpace})
	if len(args) < 1 {
		template.Must(t.Parse(helpTemplate))
		failOnError(t.Execute(os.Stderr, &commands))
	} else {
		for _, cmd := range commands {
			if cmd.Name() == args[0] {
				template.Must(t.Parse(detailedHelpTemplate))
				failOnError(t.Execute(os.Stderr, cmd))
				os.Exit(0)
			}
		}
		log.Fatalf("Unknown command %q. Run 'colcmp help'.", args[0])
	}
	os.Exit(0)
}

func main() {
	flag.Parse()
	log.SetFlags(0)

	ctxt := parseTests()

	if flag.NArg() < 1 {
		runHelp(nil)
	}
	args := flag.Args()[1:]
	if flag.Arg(0) == "help" {
		runHelp(args)
	}
	for _, cmd := range commands {
		if cmd.Name() == flag.Arg(0) {
			cmd.Run(ctxt, args)
			ctxt.flush()
			return
		}
	}
	runHelp(flag.Args())
}
