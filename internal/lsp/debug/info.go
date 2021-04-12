// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package debug exports debug information for gopls.
package debug

import (
	"context"
	"fmt"
	"io"
	"reflect"
	"runtime/debug"
	"sort"
	"strings"

	"golang.org/x/tools/internal/lsp/source"
)

type PrintMode int

const (
	PlainText = PrintMode(iota)
	Markdown
	HTML
)

// Version is a manually-updated mechanism for tracking versions.
const Version = "master"

// ServerVersion is the format used by gopls to report its version to the
// client. This format is structured so that the client can parse it easily.
type ServerVersion struct {
	Module
	Deps []*Module `json:"deps,omitempty"`
}

type Module struct {
	ModuleVersion
	Replace *ModuleVersion `json:"replace,omitempty"`
}

type ModuleVersion struct {
	Path    string `json:"path,omitempty"`
	Version string `json:"version,omitempty"`
	Sum     string `json:"sum,omitempty"`
}

// VersionInfo returns the build info for the gopls process. If it was not
// built in module mode, we return a GOPATH-specific message with the
// hardcoded version.
func VersionInfo() *ServerVersion {
	if info, ok := debug.ReadBuildInfo(); ok {
		return getVersion(info)
	}
	path := "gopls, built in GOPATH mode"
	return &ServerVersion{
		Module: Module{
			ModuleVersion: ModuleVersion{
				Path:    path,
				Version: Version,
			},
		},
	}
}

func getVersion(info *debug.BuildInfo) *ServerVersion {
	serverVersion := ServerVersion{
		Module: Module{
			ModuleVersion: ModuleVersion{
				Path:    info.Main.Path,
				Version: info.Main.Version,
				Sum:     info.Main.Sum,
			},
		},
	}
	for _, d := range info.Deps {
		m := &Module{
			ModuleVersion: ModuleVersion{
				Path:    d.Path,
				Version: d.Version,
				Sum:     d.Sum,
			},
		}
		if d.Replace != nil {
			m.Replace = &ModuleVersion{
				Path:    d.Replace.Path,
				Version: d.Replace.Version,
			}
		}
		serverVersion.Deps = append(serverVersion.Deps, m)
	}
	return &serverVersion
}

// PrintServerInfo writes HTML debug info to w for the Instance.
func (i *Instance) PrintServerInfo(ctx context.Context, w io.Writer) {
	section(w, HTML, "Server Instance", func() {
		fmt.Fprintf(w, "Start time: %v\n", i.StartTime)
		fmt.Fprintf(w, "LogFile: %s\n", i.Logfile)
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
func PrintVersionInfo(ctx context.Context, w io.Writer, verbose bool, mode PrintMode) {
	info := VersionInfo()
	if !verbose {
		printBuildInfo(w, info, false, mode)
		return
	}
	section(w, mode, "Build info", func() {
		printBuildInfo(w, info, true, mode)
	})
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
	fmt.Fprintf(w, "%v %v\n", info.Path, Version)
	printModuleInfo(w, &info.Module, mode)
	if !verbose {
		return
	}
	for _, dep := range info.Deps {
		printModuleInfo(w, dep, mode)
	}
}

func printModuleInfo(w io.Writer, m *Module, mode PrintMode) {
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

func showOptions(o *source.Options) []string {
	// non-breaking spaces for indenting current and defaults when they are on a separate line
	const indent = "\u00a0\u00a0\u00a0\u00a0\u00a0"
	var ans strings.Builder
	t := reflect.TypeOf(*o)
	swalk(t, []int{}, "")
	v := reflect.ValueOf(*o)
	do := reflect.ValueOf(*source.DefaultOptions())
	for _, f := range fields {
		val := v.FieldByIndex(f.index)
		def := do.FieldByIndex(f.index)
		tx := t.FieldByIndex(f.index)
		prefix := fmt.Sprintf("%s (type is %s): ", tx.Name, tx.Type)
		is := strVal(val)
		was := strVal(def)
		if len(is) < 30 && len(was) < 30 {
			fmt.Fprintf(&ans, "%s current:%s, default:%s\n", prefix, is, was)
		} else {
			fmt.Fprintf(&ans, "%s\n%scurrent:%s\n%sdefault:%s\n", prefix, indent, is, indent, was)
		}
	}
	return strings.Split(ans.String(), "\n")
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
