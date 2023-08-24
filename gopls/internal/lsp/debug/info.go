// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package debug exports debug information for gopls.
package debug

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/source"
)

type PrintMode int

const (
	PlainText = PrintMode(iota)
	Markdown
	HTML
	JSON
)

// Version is a manually-updated mechanism for tracking versions.
func Version() string {
	if info, ok := debug.ReadBuildInfo(); ok {
		if info.Main.Version != "" {
			return info.Main.Version
		}
	}
	return "(unknown)"
}

// ServerVersion is the format used by gopls to report its version to the
// client. This format is structured so that the client can parse it easily.
type ServerVersion struct {
	*debug.BuildInfo
	Version string
}

// VersionInfo returns the build info for the gopls process. If it was not
// built in module mode, we return a GOPATH-specific message with the
// hardcoded version.
func VersionInfo() *ServerVersion {
	if info, ok := debug.ReadBuildInfo(); ok {
		return &ServerVersion{
			Version:   Version(),
			BuildInfo: info,
		}
	}
	return &ServerVersion{
		Version: Version(),
		BuildInfo: &debug.BuildInfo{
			Path:      "gopls, built in GOPATH mode",
			GoVersion: runtime.Version(),
		},
	}
}

// PrintServerInfo writes HTML debug info to w for the Instance.
func (i *Instance) PrintServerInfo(ctx context.Context, w io.Writer) {
	section(w, HTML, "Server Instance", func() {
		fmt.Fprintf(w, "Start time: %v\n", i.StartTime)
		fmt.Fprintf(w, "LogFile: %s\n", i.Logfile)
		fmt.Fprintf(w, "pid: %d\n", os.Getpid())
		fmt.Fprintf(w, "Working directory: %s\n", i.Workdir)
		fmt.Fprintf(w, "Address: %s\n", i.ServerAddress)
		fmt.Fprintf(w, "Debug address: %s\n", i.DebugAddress())
	})
	PrintVersionInfo(ctx, w, true, HTML)
	section(w, HTML, "Command Line", func() {
		fmt.Fprintf(w, "<a href=/debug/pprof/cmdline>cmdline</a>")
	})
}

// PrintVersionInfo writes version information to w, using the output format
// specified by mode. verbose controls whether additional information is
// written, including section headers.
func PrintVersionInfo(_ context.Context, w io.Writer, verbose bool, mode PrintMode) error {
	info := VersionInfo()
	if mode == JSON {
		return printVersionInfoJSON(w, info)
	}

	if !verbose {
		printBuildInfo(w, info, false, mode)
		return nil
	}
	section(w, mode, "Build info", func() {
		printBuildInfo(w, info, true, mode)
	})
	return nil
}

func printVersionInfoJSON(w io.Writer, info *ServerVersion) error {
	js, err := json.MarshalIndent(info, "", "\t")
	if err != nil {
		return err
	}
	_, err = fmt.Fprint(w, string(js))
	return err
}

func section(w io.Writer, mode PrintMode, title string, body func()) {
	switch mode {
	case PlainText:
		fmt.Fprintln(w, title)
		fmt.Fprintln(w, strings.Repeat("-", len(title)))
		body()
	case Markdown:
		fmt.Fprintf(w, "#### %s\n\n```\n", title)
		body()
		fmt.Fprintf(w, "```\n")
	case HTML:
		fmt.Fprintf(w, "<h3>%s</h3>\n<pre>\n", title)
		body()
		fmt.Fprint(w, "</pre>\n")
	}
}

func printBuildInfo(w io.Writer, info *ServerVersion, verbose bool, mode PrintMode) {
	fmt.Fprintf(w, "%v %v\n", info.Path, Version())
	printModuleInfo(w, info.Main, mode)
	if !verbose {
		return
	}
	for _, dep := range info.Deps {
		printModuleInfo(w, *dep, mode)
	}
	fmt.Fprintf(w, "go: %v\n", info.GoVersion)
}

func printModuleInfo(w io.Writer, m debug.Module, _ PrintMode) {
	fmt.Fprintf(w, "    %s@%s", m.Path, m.Version)
	if m.Sum != "" {
		fmt.Fprintf(w, " %s", m.Sum)
	}
	if m.Replace != nil {
		fmt.Fprintf(w, " => %v", m.Replace.Path)
	}
	fmt.Fprintf(w, "\n")
}

type field struct {
	index []int
}

var fields []field

// find all the options. The presumption is that the Options are nested structs
// and that pointers don't need to be dereferenced
func swalk(t reflect.Type, ix []int, indent string) {
	switch t.Kind() {
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			fld := t.Field(i)
			ixx := append(append([]int{}, ix...), i)
			swalk(fld.Type, ixx, indent+". ")
		}
	default:
		// everything is either a struct or a field (that's an assumption about Options)
		fields = append(fields, field{ix})
	}
}

type sessionOption struct {
	Name    string
	Type    string
	Current string
	Default string
}

func showOptions(o *source.Options) []sessionOption {
	var out []sessionOption
	t := reflect.TypeOf(*o)
	swalk(t, []int{}, "")
	v := reflect.ValueOf(*o)
	do := reflect.ValueOf(*source.DefaultOptions())
	for _, f := range fields {
		val := v.FieldByIndex(f.index)
		def := do.FieldByIndex(f.index)
		tx := t.FieldByIndex(f.index)
		is := strVal(val)
		was := strVal(def)
		out = append(out, sessionOption{
			Name:    tx.Name,
			Type:    tx.Type.String(),
			Current: is,
			Default: was,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		rd := out[i].Current == out[i].Default
		ld := out[j].Current == out[j].Default
		if rd != ld {
			return ld
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func strVal(val reflect.Value) string {
	switch val.Kind() {
	case reflect.Bool:
		return fmt.Sprintf("%v", val.Interface())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return fmt.Sprintf("%v", val.Interface())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return fmt.Sprintf("%v", val.Interface())
	case reflect.Uintptr, reflect.UnsafePointer:
		return fmt.Sprintf("0x%x", val.Pointer())
	case reflect.Complex64, reflect.Complex128:
		return fmt.Sprintf("%v", val.Complex())
	case reflect.Array, reflect.Slice:
		ans := []string{}
		for i := 0; i < val.Len(); i++ {
			ans = append(ans, strVal(val.Index(i)))
		}
		sort.Strings(ans)
		return fmt.Sprintf("%v", ans)
	case reflect.Chan, reflect.Func, reflect.Ptr:
		return val.Kind().String()
	case reflect.Struct:
		var x source.Analyzer
		if val.Type() != reflect.TypeOf(x) {
			return val.Kind().String()
		}
		// this is sort of ugly, but usable
		str := val.FieldByName("Analyzer").Elem().FieldByName("Doc").String()
		ix := strings.Index(str, "\n")
		if ix == -1 {
			ix = len(str)
		}
		return str[:ix]
	case reflect.String:
		return fmt.Sprintf("%q", val.Interface())
	case reflect.Map:
		ans := []string{}
		iter := val.MapRange()
		for iter.Next() {
			k := iter.Key()
			v := iter.Value()
			ans = append(ans, fmt.Sprintf("%s:%s, ", strVal(k), strVal(v)))
		}
		sort.Strings(ans)
		return fmt.Sprintf("%v", ans)
	}
	return fmt.Sprintf("??%s??", val.Type())
}
