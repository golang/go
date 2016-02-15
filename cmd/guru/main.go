// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// guru: a tool for answering questions about Go source code.
//
//    http://golang.org/s/oracle-design
//    http://golang.org/s/oracle-user-manual
//
// Run with -help flag or help subcommand for usage information.
//
package main // import "golang.org/x/tools/cmd/guru"

import (
	"bufio"
	"bytes"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strconv"
	"strings"

	"golang.org/x/tools/go/buildutil"
)

// flags
var (
	modifiedFlag   = flag.Bool("modified", false, "read archive of modified files from standard input")
	scopeFlag      = flag.String("scope", "", "comma-separated list of `packages` the analysis should be limited to (default=all)")
	ptalogFlag     = flag.String("ptalog", "", "write points-to analysis log to `file`")
	formatFlag     = flag.String("format", "plain", "output `format`; one of {plain,json,xml}")
	reflectFlag    = flag.Bool("reflect", false, "analyze reflection soundly (slow)")
	cpuprofileFlag = flag.String("cpuprofile", "", "write CPU profile to `file`")
)

func init() {
	flag.Var((*buildutil.TagsFlag)(&build.Default.BuildTags), "tags", buildutil.TagsFlagDoc)
}

const useHelp = "Run 'guru -help' for more information.\n"

const helpMessage = `Go source code guru.
Usage: guru [flags] <mode> <position>

The mode argument determines the query to perform:

	callees	  	show possible targets of selected function call
	callers	  	show possible callers of selected function
	callstack 	show path from callgraph root to selected function
	definition	show declaration of selected identifier
	describe  	describe selected syntax: definition, methods, etc
	freevars  	show free variables of selection
	implements	show 'implements' relation for selected type or method
	peers     	show send/receive corresponding to selected channel op
	pointsto	show variables the selected pointer may point to
	referrers 	show all refs to entity denoted by selected identifier
	what		show basic information about the selected syntax node
	whicherrs	show possible values of the selected error variable

The position argument specifies the filename and byte offset (or range)
of the syntax element to query.  For example:

	foo.go:#123,#128
	bar.go:#123

The -format flag controls the output format:
	plain	an editor-friendly format in which every line of output
		is of the form "pos: text", where pos is "-" if unknown.
	json	structured data in JSON syntax.
	xml	structured data in XML syntax.

The -modified flag causes guru to read an archive from standard input.

	Files in this archive will be used in preference to those in
	the file system.  In this way, a text editor may supply guru
	with the contents of its unsaved buffers.  Each archive entry
	consists of the file name, a newline, the decimal file size,
	another newline, and the contents of the file.

User manual: http://golang.org/s/oracle-user-manual

Example: describe syntax at offset 530 in this file (an import spec):

  $ guru describe src/golang.org/x/tools/cmd/guru/main.go:#530
`

func printHelp() {
	fmt.Fprintln(os.Stderr, helpMessage)
	fmt.Fprintln(os.Stderr, "Flags:")
	flag.PrintDefaults()
}

func main() {
	log.SetPrefix("guru: ")
	log.SetFlags(0)

	// Don't print full help unless -help was requested.
	// Just gently remind users that it's there.
	flag.Usage = func() { fmt.Fprint(os.Stderr, useHelp) }
	flag.CommandLine.Init(os.Args[0], flag.ContinueOnError) // hack
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		// (err has already been printed)
		if err == flag.ErrHelp {
			printHelp()
		}
		os.Exit(2)
	}

	args := flag.Args()
	if len(args) != 2 {
		flag.Usage()
		os.Exit(2)
	}
	mode, posn := args[0], args[1]

	if mode == "help" {
		printHelp()
		os.Exit(2)
	}

	// Set up points-to analysis log file.
	var ptalog io.Writer
	if *ptalogFlag != "" {
		if f, err := os.Create(*ptalogFlag); err != nil {
			log.Fatalf("Failed to create PTA log file: %s", err)
		} else {
			buf := bufio.NewWriter(f)
			ptalog = buf
			defer func() {
				if err := buf.Flush(); err != nil {
					log.Printf("flush: %s", err)
				}
				if err := f.Close(); err != nil {
					log.Printf("close: %s", err)
				}
			}()
		}
	}

	// Profiling support.
	if *cpuprofileFlag != "" {
		f, err := os.Create(*cpuprofileFlag)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// -format flag
	switch *formatFlag {
	case "json", "plain", "xml":
		// ok
	default:
		log.Fatalf("illegal -format value: %q.\n"+useHelp, *formatFlag)
	}

	ctxt := &build.Default

	// If there were modified files,
	// read them from the standard input and
	// overlay them on the build context.
	if *modifiedFlag {
		modified, err := parseArchive(os.Stdin)
		if err != nil {
			log.Fatal(err)
		}

		// All I/O done by guru needs to consult the modified map.
		// The ReadFile done by referrers does,
		// but the loader's cgo preprocessing currently does not.

		if len(modified) > 0 {
			ctxt = useModifiedFiles(ctxt, modified)
		}
	}

	// Ask the guru.
	query := Query{
		Mode:       mode,
		Pos:        posn,
		Build:      ctxt,
		Scope:      strings.Split(*scopeFlag, ","),
		PTALog:     ptalog,
		Reflection: *reflectFlag,
	}

	if err := Run(&query); err != nil {
		log.Fatal(err)
	}

	// Print the result.
	switch *formatFlag {
	case "json":
		b, err := json.MarshalIndent(query.Serial(), "", "\t")
		if err != nil {
			log.Fatalf("JSON error: %s", err)
		}
		os.Stdout.Write(b)

	case "xml":
		b, err := xml.MarshalIndent(query.Serial(), "", "\t")
		if err != nil {
			log.Fatalf("XML error: %s", err)
		}
		os.Stdout.Write(b)

	case "plain":
		query.WriteTo(os.Stdout)
	}
}

func parseArchive(archive io.Reader) (map[string][]byte, error) {
	modified := make(map[string][]byte)
	r := bufio.NewReader(archive)
	for {
		// Read file name.
		filename, err := r.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break // OK
			}
			return nil, fmt.Errorf("reading modified file name: %v", err)
		}
		filename = filepath.Clean(strings.TrimSpace(filename))

		// Read file size.
		sz, err := r.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("reading size of modified file %s: %v", filename, err)
		}
		sz = strings.TrimSpace(sz)
		size, err := strconv.ParseInt(sz, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("parsing size of modified file %s: %v", filename, err)
		}

		// Read file content.
		var content bytes.Buffer
		content.Grow(int(size))
		if _, err := io.CopyN(&content, r, size); err != nil {
			return nil, fmt.Errorf("reading modified file %s: %v", filename, err)
		}
		modified[filename] = content.Bytes()
	}

	return modified, nil
}

// useModifiedFiles augments the provided build.Context by the
// mapping from file names to alternative contents.
func useModifiedFiles(orig *build.Context, modified map[string][]byte) *build.Context {
	rc := func(data []byte) (io.ReadCloser, error) {
		return ioutil.NopCloser(bytes.NewBuffer(data)), nil
	}

	copy := *orig // make a copy
	ctxt := &copy
	ctxt.OpenFile = func(path string) (io.ReadCloser, error) {
		// Fast path: names match exactly.
		if content, ok := modified[path]; ok {
			return rc(content)
		}

		// Slow path: check for same file under a different
		// alias, perhaps due to a symbolic link.
		for filename, content := range modified {
			if sameFile(path, filename) {
				return rc(content)
			}
		}

		return buildutil.OpenFile(orig, path)
	}
	return ctxt
}
