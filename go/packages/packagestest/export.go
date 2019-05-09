// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package packagestest creates temporary projects on disk for testing go tools on.

By changing the exporter used, you can create projects for multiple build
systems from the same description, and run the same tests on them in many
cases.
*/
package packagestest

import (
	"flag"
	"fmt"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/span"
)

var (
	skipCleanup = flag.Bool("skip-cleanup", false, "Do not delete the temporary export folders") // for debugging
)

// Module is a representation of a go module.
type Module struct {
	// Name is the base name of the module as it would be in the go.mod file.
	Name string
	// Files is the set of source files for all packages that make up the module.
	// The keys are the file fragment that follows the module name, the value can
	// be a string or byte slice, in which case it is the contents of the
	// file, otherwise it must be a Writer function.
	Files map[string]interface{}

	// Overlay is the set of source file overlays for the module.
	// The keys are the file fragment as in the Files configuration.
	// The values are the in memory overlay content for the file.
	Overlay map[string][]byte
}

// A Writer is a function that writes out a test file.
// It is provided the name of the file to write, and may return an error if it
// cannot write the file.
// These are used as the content of the Files map in a Module.
type Writer func(filename string) error

// Exported is returned by the Export function to report the structure that was produced on disk.
type Exported struct {
	// Config is a correctly configured packages.Config ready to be passed to packages.Load.
	// Exactly what it will contain varies depending on the Exporter being used.
	Config *packages.Config

	// Modules is the module description that was used to produce this exported data set.
	Modules []Module

	ExpectFileSet *token.FileSet // The file set used when parsing expectations

	temp    string                       // the temporary directory that was exported to
	primary string                       // the first non GOROOT module that was exported
	written map[string]map[string]string // the full set of exported files
	notes   []*expect.Note               // The list of expectations extracted from go source files
	markers map[string]span.Range        // The set of markers extracted from go source files
}

// Exporter implementations are responsible for converting from the generic description of some
// test data to a driver specific file layout.
type Exporter interface {
	// Name reports the name of the exporter, used in logging and sub-test generation.
	Name() string
	// Filename reports the system filename for test data source file.
	// It is given the base directory, the module the file is part of and the filename fragment to
	// work from.
	Filename(exported *Exported, module, fragment string) string
	// Finalize is called once all files have been written to write any extra data needed and modify
	// the Config to match. It is handed the full list of modules that were encountered while writing
	// files.
	Finalize(exported *Exported) error
}

// All is the list of known exporters.
// This is used by TestAll to run tests with all the exporters.
var All []Exporter

// TestAll invokes the testing function once for each exporter registered in
// the All global.
// Each exporter will be run as a sub-test named after the exporter being used.
func TestAll(t *testing.T, f func(*testing.T, Exporter)) {
	t.Helper()
	for _, e := range All {
		t.Run(e.Name(), func(t *testing.T) {
			t.Helper()
			f(t, e)
		})
	}
}

// BenchmarkAll invokes the testing function once for each exporter registered in
// the All global.
// Each exporter will be run as a sub-test named after the exporter being used.
func BenchmarkAll(b *testing.B, f func(*testing.B, Exporter)) {
	b.Helper()
	for _, e := range All {
		b.Run(e.Name(), func(b *testing.B) {
			b.Helper()
			f(b, e)
		})
	}
}

// Export is called to write out a test directory from within a test function.
// It takes the exporter and the build system agnostic module descriptions, and
// uses them to build a temporary directory.
// It returns an Exported with the results of the export.
// The Exported.Config is prepared for loading from the exported data.
// You must invoke Exported.Cleanup on the returned value to clean up.
// The file deletion in the cleanup can be skipped by setting the skip-cleanup
// flag when invoking the test, allowing the temporary directory to be left for
// debugging tests.
func Export(t testing.TB, exporter Exporter, modules []Module) *Exported {
	t.Helper()
	dirname := strings.Replace(t.Name(), "/", "_", -1)
	dirname = strings.Replace(dirname, "#", "_", -1) // duplicate subtests get a #NNN suffix.
	temp, err := ioutil.TempDir("", dirname)
	if err != nil {
		t.Fatal(err)
	}
	exported := &Exported{
		Config: &packages.Config{
			Dir:     temp,
			Env:     append(os.Environ(), "GOPACKAGESDRIVER=off"),
			Overlay: make(map[string][]byte),
			Tests:   true,
			Mode:    packages.LoadImports,
		},
		Modules:       modules,
		temp:          temp,
		primary:       modules[0].Name,
		written:       map[string]map[string]string{},
		ExpectFileSet: token.NewFileSet(),
	}
	defer func() {
		if t.Failed() || t.Skipped() {
			exported.Cleanup()
		}
	}()
	for _, module := range modules {
		for fragment, value := range module.Files {
			fullpath := exporter.Filename(exported, module.Name, filepath.FromSlash(fragment))
			written, ok := exported.written[module.Name]
			if !ok {
				written = map[string]string{}
				exported.written[module.Name] = written
			}
			written[fragment] = fullpath
			if err := os.MkdirAll(filepath.Dir(fullpath), 0755); err != nil {
				t.Fatal(err)
			}
			switch value := value.(type) {
			case Writer:
				if err := value(fullpath); err != nil {
					t.Fatal(err)
				}
			case string:
				if err := ioutil.WriteFile(fullpath, []byte(value), 0644); err != nil {
					t.Fatal(err)
				}
			default:
				t.Fatalf("Invalid type %T in files, must be string or Writer", value)
			}
		}
		for fragment, value := range module.Overlay {
			fullpath := exporter.Filename(exported, module.Name, filepath.FromSlash(fragment))
			exported.Config.Overlay[fullpath] = value
		}
	}
	if err := exporter.Finalize(exported); err != nil {
		t.Fatal(err)
	}
	return exported
}

// Script returns a Writer that writes out contents to the file and sets the
// executable bit on the created file.
// It is intended for source files that are shell scripts.
func Script(contents string) Writer {
	return func(filename string) error {
		return ioutil.WriteFile(filename, []byte(contents), 0755)
	}
}

// Link returns a Writer that creates a hard link from the specified source to
// the required file.
// This is used to link testdata files into the generated testing tree.
func Link(source string) Writer {
	return func(filename string) error {
		return os.Link(source, filename)
	}
}

// Symlink returns a Writer that creates a symlink from the specified source to the
// required file.
// This is used to link testdata files into the generated testing tree.
func Symlink(source string) Writer {
	if !strings.HasPrefix(source, ".") {
		if abspath, err := filepath.Abs(source); err == nil {
			if _, err := os.Stat(source); !os.IsNotExist(err) {
				source = abspath
			}
		}
	}
	return func(filename string) error {
		return os.Symlink(source, filename)
	}
}

// Copy returns a Writer that copies a file from the specified source to the
// required file.
// This is used to copy testdata files into the generated testing tree.
func Copy(source string) Writer {
	return func(filename string) error {
		stat, err := os.Stat(source)
		if err != nil {
			return err
		}
		if !stat.Mode().IsRegular() {
			// cannot copy non-regular files (e.g., directories,
			// symlinks, devices, etc.)
			return fmt.Errorf("Cannot copy non regular file %s", source)
		}
		contents, err := ioutil.ReadFile(source)
		if err != nil {
			return err
		}
		return ioutil.WriteFile(filename, contents, stat.Mode())
	}
}

// MustCopyFileTree returns a file set for a module based on a real directory tree.
// It scans the directory tree anchored at root and adds a Copy writer to the
// map for every file found.
// This is to enable the common case in tests where you have a full copy of the
// package in your testdata.
// This will panic if there is any kind of error trying to walk the file tree.
func MustCopyFileTree(root string) map[string]interface{} {
	result := map[string]interface{}{}
	if err := filepath.Walk(filepath.FromSlash(root), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		fragment, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		result[filepath.ToSlash(fragment)] = Copy(path)
		return nil
	}); err != nil {
		log.Panic(fmt.Sprintf("MustCopyFileTree failed: %v", err))
	}
	return result
}

// Cleanup removes the temporary directory (unless the --skip-cleanup flag was set)
// It is safe to call cleanup multiple times.
func (e *Exported) Cleanup() {
	if e.temp == "" {
		return
	}
	if *skipCleanup {
		log.Printf("Skipping cleanup of temp dir: %s", e.temp)
		return
	}
	// Make everything read-write so that the Module exporter's module cache can be deleted.
	filepath.Walk(e.temp, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			os.Chmod(path, 0777)
		}
		return nil
	})
	os.RemoveAll(e.temp) // ignore errors
	e.temp = ""
}

// Temp returns the temporary directory that was generated.
func (e *Exported) Temp() string {
	return e.temp
}

// File returns the full path for the given module and file fragment.
func (e *Exported) File(module, fragment string) string {
	if m := e.written[module]; m != nil {
		return m[fragment]
	}
	return ""
}

// FileContents returns the contents of the specified file.
// It will use the overlay if the file is present, otherwise it will read it
// from disk.
func (e *Exported) FileContents(filename string) ([]byte, error) {
	if content, found := e.Config.Overlay[filename]; found {
		return content, nil
	}
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return content, nil
}
