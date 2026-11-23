// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package exportdata provides a codec for type export data.
// The [Encode] function returns an encoding of a [types.Package] that
// may be decoded by calling [Decode].
//
// Issue https://go.dev/issue/69491 proposes to publish this package
// in the go1.28 standard library at go/types/exportdata. Once the
// package exists, this package will delegate to it and eventually
// wither away.
//
// # Protocol stability
//
// The details of the encoding are not specified and it may change;
// indeed it must change when new language features are introduced.
// However, it is a versioned protocol, and Decode is guaranteed to
// support data produced by [Encode] either at the exact same source
// version or at a tagged version no older than the two most recent Go
// toolchain releases. Older versions may be rejected.
//
// This package may be used to decode the files produced by the "go
// list -export" command. (See also [golang.org/x/tools/go/packages.Load],
// which may provide a more convenient interface for doing so.)
//
// This package's encoding, historically called "indexed" format, is
// intentionally not compatible with the "unified" encoding used by
// the Go compiler. (This avoids constraining the compiler; see
// https://go.dev/issue/79427.)
//
// The encoding is deterministic: if the encoder is applied twice to
// the same types.Package data structure, both encodings are equal.
// This property may be important to avoid spurious changes in
// applications such as build systems.
//
// Decoding followed by encoding does not necessarily reproduce the
// exact same bytes. Importing an exported package may yield a
// types.Package that, while it represents the same set of Go objects
// as the original, may differ in the details of its internal
// representation. Because of these differences, re-encoding the
// imported package may yield a different, but equally valid, encoding
// of the package.
//
// # Deep vs shallow
//
// One of the [EncodeOptions] controls whether the encoding is "deep"
// or "shallow". A deep encoding describes all exported package-level
// objects in the encoded package, along with all the objects that are
// reachable from them through the type data structures, following
// function parameters and results, methods, struct fields, and so on.
// The indirect packages may be described incompletely. A deep
// encoding is thus self-contained with respect to transitive
// dependencies: when type checking a package, it suffices to decode
// one "deep" export data file for each of its direct imports.
//
// The export data used by the Go compiler is also "deep". Because the
// set of inputs is determined by the package's import dependencies
// alone, and is not based on the contents of any file, deep export
// data suits build systems (such as Bazel) that need to know each
// action's inputs in advance, for example to distribute them to
// remote workers.
//
// By contrast, a shallow encoding describes only the objects exported
// from the encoded package; objects from packages that it imports,
// either directly or indirectly, are referenced by name. This makes
// the encoding asymptotically smaller, but not self-contained. When
// decoding a shallow encoding, it is necessary to decode additional
// files for transitive dependencies, though typically only a small
// subset, because most imports do not contribute types to a package's
// exported API. (Consider the "fmt" package: nearly every package
// imports it and uses it within the bodies of its functions, yet few
// packages refer to types from "fmt" in their own API.) Thus it may
// be necessary to recursively obtain packages for those dependencies,
// by decoding their export data, before the package itself can be
// decoded.
//
// With shallow export data, the set of files needed to decode a
// package is discovered only by decoding, so a build system that must
// know each action's inputs in advance would have to supply the
// entire transitive closure. Shallow export data therefore suits an
// interactive tool that can read export data for any package on
// demand from a local cache (such as gopls) better than a distributed
// build system.
//
// # Canonical packages and objects
//
// The type checker's data structures, including [types.Package] and
// [types.Object] use referential equality. Thus, a program that
// decodes several packages must represent each package path by a
// single canonical [types.Package], and each object within it by a
// single canonical [types.Object]. Consequently, an object reached
// through the export data of two different imports is the same
// reference in both, just as if the packages had been type-checked
// together from source.
//
// Conceptually, all decode operations share an import mapping from
// each package path to its canonical types.Package. When [Decode]
// encounters a reference to an object of another package, it consults
// the mapping to find the [types.Package]. In "shallow" mode, it is
// the application's responsibility to ensure that the dependency
// package exists and is populated; in "deep" mode, a missing package
// is created using [types.NewPackage] and then partially populated as
// needed as a consequence of decoding the primary package.
//
// Decode does not manipulate the import mapping directly. It takes a
// function, getPackages, that abstracts both the mapping and the
// creation of new packages. For convenience, we provide
// [PackagesFromMap], a simple implementation of this function based
// on an ordinary map.
//
// # Miscellaneous
//
// It is valid to encode and decode packages containing type errors.
// The codec will do its best to preserve the data structures, though
// it does not guarantee to do so without loss.
package exportdata

import (
	"bytes"
	"fmt"
	"go/token"
	"go/types"
)

// Encode returns the encoding of pkg.
// Positions of objects are interpreted using fset, which must be non-nil.
func Encode(fset *token.FileSet, pkg *types.Package, opts EncodeOptions) ([]byte, error) {
	if fset == nil {
		panic("nil FileSet")
	}
	var buf bytes.Buffer
	buf.WriteByte('i')
	if err := iexportCommon(&buf, fset, opts.Shallow, iexportVersion, pkg); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// EncodeOptions controls the encoding of export data by [Encode].
type EncodeOptions struct {
	// Shallow causes the data to include only the package's own
	// objects, and to refer to other packages' objects
	// symbolically. This avoids redundantly re-encoding objects
	// from indirect dependencies at the cost of having to look up
	// (some) indirect dependencies during decoding.
	//
	// By default the data is "deep": it includes a complete
	// description of all objects referred to by the encoded
	// package, even if they are declared in a dependency. Thus
	// the decoder needs access only to export data for its direct
	// dependencies.
	Shallow bool
}

// Decode decodes export data for the package with the given path.
// (The encoding does not record the path.)
//
// Decode calls getPackages once, before decoding any objects, with
// a reference to each package mentioned by the data, including the
// package being decoded, so that the caller may obtain them in
// parallel. getPackages must return a slice of the same length
// whose ith element is the canonical package for refs[i]; its path
// and name must match. In deep mode the packages may be empty, and
// Decode populates them. In shallow mode the packages for
// dependencies must be complete. Any error returned by getPackages
// is returned by Decode, unwrapped; this may be used to read only
// the list of packages mentioned by the data.
//
// Position information is added to fset, which must be non-nil.
//
// The data must have been produced by [Encode], either at the exact
// same source version, or at a tagged version no older than the two
// most recent Go toolchain releases.
func Decode(fset *token.FileSet, path string, data []byte, getPackages func([]PackageRef) ([]*types.Package, error), opts DecodeOptions) (*types.Package, error) {
	if fset == nil {
		panic("nil FileSet")
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("decoding %q: export data is empty", path)
	}
	if data[0] != 'i' {
		return nil, fmt.Errorf("decoding %q: not indexed export data", path)
	}
	return iimportCommon(fset, getPackages, data[1:], path)
}

// A PackageRef identifies a package mentioned by export data.
type PackageRef struct {
	Path, Name string
}

// DecodeOptions controls the decoding of export data by [Decode].
type DecodeOptions struct {
	// for future use
}

// PackagesFromMap is a convenience function that returns a getPackages
// function suitable for [Decode] that obtains packages from m,
// a mapping from package path to canonical package, as used by
// [go/importer] and gcexportdata. Each requested package that is not
// found in m is created with [types.NewPackage] and inserted into m.
func PackagesFromMap(m map[string]*types.Package) func([]PackageRef) ([]*types.Package, error) {
	return func(refs []PackageRef) ([]*types.Package, error) {
		pkgs := make([]*types.Package, len(refs))
		for i, ref := range refs {
			pkg, ok := m[ref.Path]
			if !ok {
				pkg = types.NewPackage(ref.Path, ref.Name)
				m[ref.Path] = pkg
			}
			pkgs[i] = pkg
		}
		return pkgs, nil
	}
}
