// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tool is an opinionated harness for writing Go tools.
package tool

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"reflect"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"time"
)

// This file is a very opinionated harness for writing your main function.
// The original version of the file is in golang.org/x/tools/internal/tool.
//
// It adds a method to the Application type
//     Main(name, usage string, args []string)
// which should normally be invoked from a true main as follows:
//     func main() {
//       (&Application{}).Main("myapp", "non-flag-command-line-arg-help", os.Args[1:])
//     }
// It recursively scans the application object for fields with a tag containing
//     `flag:"flagname" help:"short help text"``
// uses all those fields to build command line flags.
// It expects the Application type to have a method
//     Run(context.Context, args...string) error
// which it invokes only after all command line flag processing has been finished.
// If Run returns an error, the error will be printed to stderr and the
// application will quit with a non zero exit status.

// Profile can be embedded in your application struct to automatically
// add command line arguments and handling for the common profiling methods.
type Profile struct {
	CPU    string `flag:"profile.cpu" help:"write CPU profile to this file"`
	Memory string `flag:"profile.mem" help:"write memory profile to this file"`
	Trace  string `flag:"profile.trace" help:"write trace log to this file"`
}

// Application is the interface that must be satisfied by an object passed to Main.
type Application interface {
	// Name returns the application's name. It is used in help and error messages.
	Name() string
	// Most of the help usage is automatically generated, this string should only
	// describe the contents of non flag arguments.
	Usage() string
	// ShortHelp returns the one line overview of the command.
	ShortHelp() string
	// DetailedHelp should print a detailed help message. It will only ever be shown
	// when the ShortHelp is also printed, so there is no need to duplicate
	// anything from there.
	// It is passed the flag set so it can print the default values of the flags.
	// It should use the flag sets configured Output to write the help to.
	DetailedHelp(*flag.FlagSet)
	// Run is invoked after all flag processing, and inside the profiling and
	// error handling harness.
	Run(ctx context.Context, args ...string) error
}

// This is the type returned by CommandLineErrorf, which causes the outer main
// to trigger printing of the command line help.
type commandLineError string

func (e commandLineError) Error() string { return string(e) }

// CommandLineErrorf is like fmt.Errorf except that it returns a value that
// triggers printing of the command line help.
// In general you should use this when generating command line validation errors.
func CommandLineErrorf(message string, args ...interface{}) error {
	return commandLineError(fmt.Sprintf(message, args...))
}

// Main should be invoked directly by main function.
// It will only return if there was no error.
func Main(ctx context.Context, app Application, args []string) {
	s := flag.NewFlagSet(app.Name(), flag.ExitOnError)
	s.Usage = func() {
		fmt.Fprint(s.Output(), app.ShortHelp())
		fmt.Fprintf(s.Output(), "\n\nUsage: %v [flags] %v\n", app.Name(), app.Usage())
		app.DetailedHelp(s)
	}
	p := addFlags(s, reflect.StructField{}, reflect.ValueOf(app))
	s.Parse(args)
	err := func() error {
		if p != nil && p.CPU != "" {
			f, err := os.Create(p.CPU)
			if err != nil {
				return err
			}
			if err := pprof.StartCPUProfile(f); err != nil {
				return err
			}
			defer pprof.StopCPUProfile()
		}

		if p != nil && p.Trace != "" {
			f, err := os.Create(p.Trace)
			if err != nil {
				return err
			}
			if err := trace.Start(f); err != nil {
				return err
			}
			defer func() {
				trace.Stop()
				log.Printf("To view the trace, run:\n$ go tool trace view %s", p.Trace)
			}()
		}

		if p != nil && p.Memory != "" {
			f, err := os.Create(p.Memory)
			if err != nil {
				return err
			}
			defer func() {
				runtime.GC() // get up-to-date statistics
				if err := pprof.WriteHeapProfile(f); err != nil {
					log.Printf("Writing memory profile: %v", err)
				}
				f.Close()
			}()
		}
		return app.Run(ctx, s.Args()...)
	}()
	if err != nil {
		fmt.Fprintf(s.Output(), "%s: %v\n", app.Name(), err)
		if _, printHelp := err.(commandLineError); printHelp {
			s.Usage()
		}
		os.Exit(2)
	}
}

// addFlags scans fields of structs recursively to find things with flag tags
// and add them to the flag set.
func addFlags(f *flag.FlagSet, field reflect.StructField, value reflect.Value) *Profile {
	// is it a field we are allowed to reflect on?
	if field.PkgPath != "" {
		return nil
	}
	// now see if is actually a flag
	flagName, isFlag := field.Tag.Lookup("flag")
	help := field.Tag.Get("help")
	if !isFlag {
		// not a flag, but it might be a struct with flags in it
		if value.Elem().Kind() != reflect.Struct {
			return nil
		}
		p, _ := value.Interface().(*Profile)
		// go through all the fields of the struct
		sv := value.Elem()
		for i := 0; i < sv.Type().NumField(); i++ {
			child := sv.Type().Field(i)
			v := sv.Field(i)
			// make sure we have a pointer
			if v.Kind() != reflect.Ptr {
				v = v.Addr()
			}
			// check if that field is a flag or contains flags
			if fp := addFlags(f, child, v); fp != nil {
				p = fp
			}
		}
		return p
	}
	switch v := value.Interface().(type) {
	case flag.Value:
		f.Var(v, flagName, help)
	case *bool:
		f.BoolVar(v, flagName, *v, help)
	case *time.Duration:
		f.DurationVar(v, flagName, *v, help)
	case *float64:
		f.Float64Var(v, flagName, *v, help)
	case *int64:
		f.Int64Var(v, flagName, *v, help)
	case *int:
		f.IntVar(v, flagName, *v, help)
	case *string:
		f.StringVar(v, flagName, *v, help)
	case *uint:
		f.UintVar(v, flagName, *v, help)
	case *uint64:
		f.Uint64Var(v, flagName, *v, help)
	default:
		log.Fatalf("Cannot understand flag of type %T", v)
	}
	return nil
}
