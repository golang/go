// Unless explicitly stated otherwise all files in this repository are licensed
// under the Apache License Version 2.0.
// This product includes software developed at Datadog (https://www.datadoghq.com/).
// Copyright 2016-present Datadog, Inc.

package main

import (
	"debug/gosym/v2"
	"flag"
	"fmt"
	"os"
	"strconv"
)

var (
	inlines = flag.Bool("inlines", false, "List inline functions")
	lines   = flag.Bool("lines", false, "List function lines")
)

func functions(table *gosym.Table) error {
	for f := range table.Functions() {
		name, err := f.Name()
		if err != nil {
			return err
		}
		entry, err := f.Entry()
		if err != nil {
			return err
		}
		end, err := f.End()
		if err != nil {
			return err
		}
		deferreturn, err := f.DeferReturn()
		if err != nil {
			return err
		}
		file, err := f.File()
		if err != nil {
			return err
		}
		startLine, err := f.StartLine()
		if err != nil {
			return err
		}
		fmt.Printf("%s@%s:%d pc=[0x%x, 0x%x) deferreturn=0x%x\n", name.Value(), file.Value(), startLine, entry, end, deferreturn)
	}
	return nil
}

func describe(table *gosym.Table, pc uint64, inlines, lines bool) error {
	f, err := table.ResolveFunction(pc)
	name, err := f.Name()
	if err != nil {
		return err
	}
	entry, err := f.Entry()
	if err != nil {
		return err
	}
	end, err := f.End()
	if err != nil {
		return err
	}
	deferreturn, err := f.DeferReturn()
	if err != nil {
		return err
	}
	file, err := f.File()
	if err != nil {
		return err
	}
	startLine, err := f.StartLine()
	if err != nil {
		return err
	}
	fmt.Printf("%s@%s:%d pc=[0x%x, 0x%x) deferreturn=0x%x\n",
		name.Value(), file.Value(), startLine, entry, end, deferreturn)

	if inlines {
		inlines, err := f.InlineFunctions(nil)
		if err != nil {
			return err
		}
		if len(inlines) == 0 {
			fmt.Println("  (no inlines)")
		}
		for _, inline := range inlines {
			fmt.Printf("  %s@%s:%d\n",
				inline.Name.Value(), inline.File.Value(), inline.StartLine)
		}
	}
	if lines {
		r, err := f.Lines(gosym.LinesResult{})
		if err != nil {
			return err
		}
		if len(r.FunctionLines) == 0 {
			fmt.Println("  (no lines)")
		}
		for _, line := range r.FunctionLines {
			fmt.Printf("  [0x%x, 0x%x) %s@%s:%d parentPC=0x%x\n",
				line.PCLo, line.PCHi, line.Name.Value(), line.File.Value(), line.Line, line.ParentPC)
		}
	}
	return nil
}

func locate(table *gosym.Table, pc uint64) error {
	locs, err := table.ResolveLocations(pc, nil)
	if err != nil {
		return err
	}
	for _, loc := range locs {
		fmt.Printf("%s@%s:%d\n", loc.Function.Value(), loc.File.Value(), loc.Line)
	}
	return nil
}

func main() {
	usage := func() {
		fmt.Printf(`Usage:
$ %s functions <binary>                             # List all function symbols in this binary
$ %s describe <binary> <pc> (--inlines)? (--lines)? # Describe a function at a specific pc
$ %s locate <binary> <pc>                           # Locate a pc
`, os.Args[0], os.Args[0], os.Args[0])
		os.Exit(1)
	}
	if len(os.Args) < 3 {
		fmt.Printf("missing subcommand or binary argument\n")
		usage()
		os.Exit(1)
	}

	binary := os.Args[2]
	reader, err := os.Open(binary)
	if err != nil {
		fmt.Printf("failed to open binary: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()
	table, err := gosym.NewMagic(reader)
	if err != nil {
		fmt.Printf("failed to parse binary: %v\n", err)
		os.Exit(1)
	}

	var pc uint64
	if len(os.Args) > 3 {
		pc, err = strconv.ParseUint(os.Args[3], 0, 64)
		if err != nil {
			fmt.Printf("failed to parse pc: %v\n", err)
			os.Exit(1)
		}
	}

	subcommand := os.Args[1]
	switch subcommand {
	case "functions":
		if len(os.Args) != 3 {
			fmt.Printf("functions expects 1 argument, got %d\n", len(os.Args)-2)
			usage()
			os.Exit(1)
		}
		err := functions(table)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
	case "describe":
		if len(os.Args) < 4 {
			fmt.Printf("describe expects 2 arguments, got %d\n", len(os.Args)-2)
			usage()
			os.Exit(1)
		}
		flag.CommandLine.Parse(os.Args[4:])
		err := describe(table, pc, *inlines, *lines)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
	case "locate":
		if len(os.Args) < 4 {
			fmt.Printf("locate expects 2 arguments, got %d\n", len(os.Args)-2)
			usage()
			os.Exit(1)
		}
		err := locate(table, pc)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
	default:
		fmt.Printf("unknown subcommand: %s\n", subcommand)
		usage()
		os.Exit(1)
	}
}
