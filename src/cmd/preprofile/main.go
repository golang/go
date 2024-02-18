// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Preprofile handles pprof files.
//
// Usage:
//
//	go tool preprofile [-v] [-o output] [-i (pprof)input]
//
//

package main

import (
	"bufio"
	"flag"
	"fmt"
	"internal/profile"
	"log"
	"os"
	"path/filepath"
	"strconv"
)

// The current Go Compiler consumes significantly long compilation time when the PGO
// is enabled. To optimize the existing flow and reduce build time of multiple Go
// services, we create a standalone tool, PGO preprocessor, to extract information
// from collected profiling files and to cache the WeightedCallGraph in one time
// fashion. By adding the new tool to the Go compiler, it will reduce the time
// of repeated profiling file parsing and avoid WeightedCallGraph reconstruction
// in current Go Compiler.
// The format of the pre-processed output is as follows.
//
//	Header
//	caller_name
//      callee_name
//      "call site offset" "call edge weight"
//      ...
//	caller_name
//      callee_name
//      "call site offset" "call edge weight"

func usage() {
	fmt.Fprintf(os.Stderr, "MUST have (pprof) input file \n")
	fmt.Fprintf(os.Stderr, "usage: go tool preprofile [-v] [-o output] [-i (pprof)input] \n\n")
	flag.PrintDefaults()
	os.Exit(2)
}

type NodeMapKey struct {
	CallerName     string
	CalleeName     string
	CallSiteOffset int // Line offset from function start line.
}

func readPprofFile(profileFile string, outputFile string, verbose bool) bool {
	// open the pprof profile file
	f, err := os.Open(profileFile)
	if err != nil {
		log.Fatal("failed to open file " + profileFile)
		return false
	}
	defer f.Close()
	p, err := profile.Parse(f)
	if err != nil {
		log.Fatal("failed to Parse profile file.")
		return false
	}

	if len(p.Sample) == 0 {
		// We accept empty profiles, but there is nothing to do.
		return false
	}

	valueIndex := -1
	for i, s := range p.SampleType {
		// Samples count is the raw data collected, and CPU nanoseconds is just
		// a scaled version of it, so either one we can find is fine.
		if (s.Type == "samples" && s.Unit == "count") ||
			(s.Type == "cpu" && s.Unit == "nanoseconds") {
			valueIndex = i
			break
		}
	}

	if valueIndex == -1 {
		log.Fatal("failed to find CPU samples count or CPU nanoseconds value-types in profile.")
		return false
	}

	// The processing here is equivalent to cmd/compile/internal/pgo.createNamedEdgeMap.
	g := profile.NewGraph(p, &profile.Options{
		SampleValue: func(v []int64) int64 { return v[valueIndex] },
	})

	TotalEdgeWeight := int64(0)

	NodeMap := make(map[NodeMapKey]int64)

	for _, n := range g.Nodes {
		canonicalName := n.Info.Name
		// Create the key to the nodeMapKey.
		nodeinfo := NodeMapKey{
			CallerName:     canonicalName,
			CallSiteOffset: n.Info.Lineno - n.Info.StartLine,
		}

		if n.Info.StartLine == 0 {
			if verbose {
				log.Println("[PGO] warning: " + canonicalName + " relative line number is missing from the profile")
			}
		}

		for _, e := range n.Out {
			TotalEdgeWeight += e.WeightValue()
			nodeinfo.CalleeName = e.Dest.Info.Name
			if w, ok := NodeMap[nodeinfo]; ok {
				w += e.WeightValue()
			} else {
				w = e.WeightValue()
				NodeMap[nodeinfo] = w
			}
		}
	}

	var fNodeMap *os.File
	if outputFile == "" {
		fNodeMap = os.Stdout
	} else {
		dirPath := filepath.Dir(outputFile)
		_, err := os.Stat(dirPath)
		if err != nil {
			log.Fatal("Directory does not exist: ", dirPath)
		}
		base := filepath.Base(outputFile)
		outputFile = filepath.Join(dirPath, base)

		// write out NodeMap to a file
		fNodeMap, err = os.Create(outputFile)
		if err != nil {
			log.Fatal("Error creating output file:", err)
			return false
		}

		defer fNodeMap.Close() // Close the file when done writing
	}

	w := bufio.NewWriter(fNodeMap)
	w.WriteString("GO PREPROFILE V1\n")
	count := 1
	separator := " "
	for key, element := range NodeMap {
		line := key.CallerName + "\n"
		w.WriteString(line)
		line = key.CalleeName + "\n"
		w.WriteString(line)
		line = strconv.Itoa(key.CallSiteOffset)
		line = line + separator + strconv.FormatInt(element, 10) + "\n"
		w.WriteString(line)
		w.Flush()
		count += 1
	}

	if TotalEdgeWeight == 0 {
		return false
	}

	return true
}

var dumpCode = flag.String("o", "", "dump output file ")
var input = flag.String("i", "", "input pprof file ")
var verbose = flag.Bool("v", false, "verbose log")

func main() {
	log.SetFlags(0)
	log.SetPrefix("preprofile: ")

	flag.Usage = usage
	flag.Parse()
	if *input == "" {
		usage()
	} else {
		readPprofFile(*input, *dumpCode, *verbose)
	}
}
