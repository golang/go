// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
)

// libFuzzer State and Entry Points
var libfuzzerInitOnce sync.Once

// libfuzzerFuzzState is the persistent fuzz state shared across libFuzzer calls.
// This ensures seeds are only run once on the first iteration.
var libfuzzerFuzzState = &fuzzState{
	mode: libFuzzerMode,
}

// libfuzzerCorpusDir is the corpus directory passed to the fuzzer.
// If set, f.Add() seeds will be written here in libFuzzer format.
var libfuzzerCorpusDir string

// SetLibFuzzerCorpusDir sets the corpus directory for libFuzzer mode.
// This is called from LLVMFuzzerInitialize with the first positional argument.
func SetLibFuzzerCorpusDir(dir string) {
	libfuzzerCorpusDir = dir
}

// GetLibFuzzerCorpusDir returns the corpus directory for libFuzzer mode.
func GetLibFuzzerCorpusDir() string {
	return libfuzzerCorpusDir
}

// RunLibFuzzerTarget runs the specified fuzz target with the given input data.
// This function is called from the generated _testmain.go when building a
// libFuzzer-compatible binary.
//
// It finds the fuzz target matching targetName, deserializes the input bytes
// into the appropriate typed arguments, and runs the fuzz target function.
func RunLibFuzzerTarget(
	targets []InternalFuzzTarget,
	targetName string,
	data []byte,
) {
	// Initialize testing package on first call
	libfuzzerInitOnce.Do(func() {
		Init()
	})

	// Find the target
	var target *InternalFuzzTarget
	for i := range targets {
		if targets[i].Name == targetName {
			target = &targets[i]
			break
		}
	}
	if target == nil {
		// If no target name specified or not found, use the first one
		if len(targets) > 0 {
			target = &targets[0]
		} else {
			return
		}
	}

	runFuzzIteration(target.Name, target.Fn, libfuzzerFuzzState, data)
}

func init() {
	// Register the corpus reader so fuzz.go can use it in libFuzzer mode
	libfuzzerReadTestdataCorpus = readTestdataCorpus
}

// libFuzzer Source (argument deserialization)
// libfuzzerSource wraps byte data for libFuzzer mode argument parsing.
// It delegates to the shared deserializeLibfuzzerBytes function but returns
// reflect.Value for direct use with fn.Call() in the hot path.
type libfuzzerSource struct {
	data []byte
}

func newLibfuzzerSource(data []byte) *libfuzzerSource {
	return &libfuzzerSource{data: data}
}

// fillArgs creates the arguments for a fuzz function from the byte source.
// types should be the types of the fuzz function parameters (excluding *testing.T).
//
// Format: [fixed-size args][uint32 weights...][dynamic data concatenated]
func (s *libfuzzerSource) fillArgs(types []reflect.Type) []reflect.Value {
	// Use shared deserialization logic
	vals := deserializeLibfuzzerBytes(s.data, types)

	// Convert []any to []reflect.Value for fn.Call()
	args := make([]reflect.Value, len(vals))
	for i, v := range vals {
		args[i] = reflect.ValueOf(v)
	}
	return args
}

// readTestdataCorpus reads all corpus files from testdata/fuzz/<name>/ directory.
// Returns empty slice if directory doesn't exist.
func readTestdataCorpus(name string, types []reflect.Type) ([]corpusEntry, error) {
	dir := filepath.Join(corpusDir, name)
	files, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil // No corpus to read
	} else if err != nil {
		return nil, fmt.Errorf("reading seed corpus from testdata: %v", err)
	}

	var corpus []corpusEntry
	var errs []string
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		filename := filepath.Join(dir, file.Name())
		data, err := os.ReadFile(filename)
		if err != nil {
			return nil, fmt.Errorf("failed to read corpus file: %v", err)
		}
		vals, err := parseGoCorpusFile(data, types)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%q: %v", filename, err))
			continue
		}
		corpus = append(corpus, corpusEntry{Path: filename, Values: vals})
	}

	if len(errs) > 0 {
		return corpus, fmt.Errorf("errors reading corpus:\n%s", strings.Join(errs, "\n"))
	}
	return corpus, nil
}

// convertCorpusDir converts all files in srcDir using the provided conversion
// function and writes the results to dstDir. Returns the number of files
// converted and any errors encountered.
func convertCorpusDir(
	srcDir, dstDir string,
	convert func(data []byte) ([]byte, error),
) (int, error) {
	files, err := os.ReadDir(srcDir)
	if err != nil {
		return 0, fmt.Errorf("reading source directory: %v", err)
	}

	if err := os.MkdirAll(dstDir, 0755); err != nil {
		return 0, fmt.Errorf("creating destination directory: %v", err)
	}

	var converted int
	var errs []string

	for _, file := range files {
		if file.IsDir() {
			continue
		}

		srcPath := filepath.Join(srcDir, file.Name())
		data, err := os.ReadFile(srcPath)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: read error: %v", file.Name(), err))
			continue
		}

		output, err := convert(data)
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", file.Name(), err))
			continue
		}

		dstPath := filepath.Join(dstDir, file.Name())
		if err := os.WriteFile(dstPath, output, 0644); err != nil {
			errs = append(errs, fmt.Sprintf("%s: write error: %v", file.Name(), err))
			continue
		}

		converted++
	}

	if len(errs) > 0 {
		return converted, fmt.Errorf("conversion errors:\n%s", strings.Join(errs, "\n"))
	}
	return converted, nil
}

// ConvertLibfuzzerToGoCorpus converts a libFuzzer corpus directory to
// Go corpus format. It reads all files from srcDir, deserializes them using
// the provided types, and writes Go corpus files to dstDir.
//
// The types parameter specifies the fuzz function parameter types
// (excluding *testing.F).
// Returns the number of files converted and any error encountered.
func ConvertLibfuzzerToGoCorpus(
	srcDir, dstDir string, types []reflect.Type,
) (int, error) {
	return convertCorpusDir(srcDir, dstDir, func(data []byte) ([]byte, error) {
		vals := deserializeLibfuzzerBytes(data, types)
		return marshalGoCorpusFile(vals), nil
	})
}

// ConvertGoCorpusToLibfuzzer converts a Go corpus directory to libFuzzer format.
// It reads all Go corpus files from srcDir and writes raw libFuzzer bytes
// to dstDir.
//
// The types parameter specifies the expected fuzz function parameter types.
// Returns the number of files converted and any error encountered.
func ConvertGoCorpusToLibfuzzer(
	srcDir, dstDir string, types []reflect.Type,
) (int, error) {
	return convertCorpusDir(srcDir, dstDir, func(data []byte) ([]byte, error) {
		vals, err := parseGoCorpusFile(data, types)
		if err != nil {
			return nil, err
		}
		return serializeLibfuzzerBytes(vals), nil
	})
}
