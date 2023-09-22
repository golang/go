// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package facts defines a serializable set of analysis.Fact.
//
// It provides a partial implementation of the Fact-related parts of the
// analysis.Pass interface for use in analysis drivers such as "go vet"
// and other build systems.
//
// The serial format is unspecified and may change, so the same version
// of this package must be used for reading and writing serialized facts.
//
// The handling of facts in the analysis system parallels the handling
// of type information in the compiler: during compilation of package P,
// the compiler emits an export data file that describes the type of
// every object (named thing) defined in package P, plus every object
// indirectly reachable from one of those objects. Thus the downstream
// compiler of package Q need only load one export data file per direct
// import of Q, and it will learn everything about the API of package P
// and everything it needs to know about the API of P's dependencies.
//
// Similarly, analysis of package P emits a fact set containing facts
// about all objects exported from P, plus additional facts about only
// those objects of P's dependencies that are reachable from the API of
// package P; the downstream analysis of Q need only load one fact set
// per direct import of Q.
//
// The notion of "exportedness" that matters here is that of the
// compiler. According to the language spec, a method pkg.T.f is
// unexported simply because its name starts with lowercase. But the
// compiler must nonetheless export f so that downstream compilations can
// accurately ascertain whether pkg.T implements an interface pkg.I
// defined as interface{f()}. Exported thus means "described in export
// data".
package facts

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"go/types"
	"io"
	"log"
	"reflect"
	"sort"
	"sync"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/types/objectpath"
	"golang.org/x/tools/internal/typesinternal"
)

const debug = false

// A Set is a set of analysis.Facts.
//
// Decode creates a Set of facts by reading from the imports of a given
// package, and Encode writes out the set. Between these operation,
// the Import and Export methods will query and update the set.
//
// All of Set's methods except String are safe to call concurrently.
type Set struct {
	pkg *types.Package
	mu  sync.Mutex
	m   map[key]analysis.Fact
}

type key struct {
	pkg *types.Package
	obj types.Object // (object facts only)
	t   reflect.Type
}

// ImportObjectFact implements analysis.Pass.ImportObjectFact.
func (s *Set) ImportObjectFact(obj types.Object, ptr analysis.Fact) bool {
	if obj == nil {
		panic("nil object")
	}
	key := key{pkg: obj.Pkg(), obj: obj, t: reflect.TypeOf(ptr)}
	s.mu.Lock()
	defer s.mu.Unlock()
	if v, ok := s.m[key]; ok {
		reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
		return true
	}
	return false
}

// ExportObjectFact implements analysis.Pass.ExportObjectFact.
func (s *Set) ExportObjectFact(obj types.Object, fact analysis.Fact) {
	if obj.Pkg() != s.pkg {
		log.Panicf("in package %s: ExportObjectFact(%s, %T): can't set fact on object belonging another package",
			s.pkg, obj, fact)
	}
	key := key{pkg: obj.Pkg(), obj: obj, t: reflect.TypeOf(fact)}
	s.mu.Lock()
	s.m[key] = fact // clobber any existing entry
	s.mu.Unlock()
}

func (s *Set) AllObjectFacts(filter map[reflect.Type]bool) []analysis.ObjectFact {
	var facts []analysis.ObjectFact
	s.mu.Lock()
	for k, v := range s.m {
		if k.obj != nil && filter[k.t] {
			facts = append(facts, analysis.ObjectFact{Object: k.obj, Fact: v})
		}
	}
	s.mu.Unlock()
	return facts
}

// ImportPackageFact implements analysis.Pass.ImportPackageFact.
func (s *Set) ImportPackageFact(pkg *types.Package, ptr analysis.Fact) bool {
	if pkg == nil {
		panic("nil package")
	}
	key := key{pkg: pkg, t: reflect.TypeOf(ptr)}
	s.mu.Lock()
	defer s.mu.Unlock()
	if v, ok := s.m[key]; ok {
		reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
		return true
	}
	return false
}

// ExportPackageFact implements analysis.Pass.ExportPackageFact.
func (s *Set) ExportPackageFact(fact analysis.Fact) {
	key := key{pkg: s.pkg, t: reflect.TypeOf(fact)}
	s.mu.Lock()
	s.m[key] = fact // clobber any existing entry
	s.mu.Unlock()
}

func (s *Set) AllPackageFacts(filter map[reflect.Type]bool) []analysis.PackageFact {
	var facts []analysis.PackageFact
	s.mu.Lock()
	for k, v := range s.m {
		if k.obj == nil && filter[k.t] {
			facts = append(facts, analysis.PackageFact{Package: k.pkg, Fact: v})
		}
	}
	s.mu.Unlock()
	return facts
}

// gobFact is the Gob declaration of a serialized fact.
type gobFact struct {
	PkgPath string          // path of package
	Object  objectpath.Path // optional path of object relative to package itself
	Fact    analysis.Fact   // type and value of user-defined Fact
}

// A Decoder decodes the facts from the direct imports of the package
// provided to NewEncoder. A single decoder may be used to decode
// multiple fact sets (e.g. each for a different set of fact types)
// for the same package. Each call to Decode returns an independent
// fact set.
type Decoder struct {
	pkg        *types.Package
	getPackage GetPackageFunc
}

// NewDecoder returns a fact decoder for the specified package.
//
// It uses a brute-force recursive approach to enumerate all objects
// defined by dependencies of pkg, so that it can learn the set of
// package paths that may be mentioned in the fact encoding. This does
// not scale well; use [NewDecoderFunc] where possible.
func NewDecoder(pkg *types.Package) *Decoder {
	// Compute the import map for this package.
	// See the package doc comment.
	m := importMap(pkg.Imports())
	getPackageFunc := func(path string) *types.Package { return m[path] }
	return NewDecoderFunc(pkg, getPackageFunc)
}

// NewDecoderFunc returns a fact decoder for the specified package.
//
// It calls the getPackage function for the package path string of
// each dependency (perhaps indirect) that it encounters in the
// encoding. If the function returns nil, the fact is discarded.
//
// This function is preferred over [NewDecoder] when the client is
// capable of efficient look-up of packages by package path.
func NewDecoderFunc(pkg *types.Package, getPackage GetPackageFunc) *Decoder {
	return &Decoder{
		pkg:        pkg,
		getPackage: getPackage,
	}
}

// A GetPackageFunc function returns the package denoted by a package path.
type GetPackageFunc = func(pkgPath string) *types.Package

// Decode decodes all the facts relevant to the analysis of package
// pkgPath. The read function reads serialized fact data from an external
// source for one of pkg's direct imports, identified by package path.
// The empty file is a valid encoding of an empty fact set.
//
// It is the caller's responsibility to call gob.Register on all
// necessary fact types.
//
// Concurrent calls to Decode are safe, so long as the
// [GetPackageFunc] (if any) is also concurrency-safe.
//
// TODO(golang/go#61443): eliminate skipMethodSorting one way or the other.
func (d *Decoder) Decode(skipMethodSorting bool, read func(pkgPath string) ([]byte, error)) (*Set, error) {
	// Read facts from imported packages.
	// Facts may describe indirectly imported packages, or their objects.
	m := make(map[key]analysis.Fact) // one big bucket
	for _, imp := range d.pkg.Imports() {
		logf := func(format string, args ...interface{}) {
			if debug {
				prefix := fmt.Sprintf("in %s, importing %s: ",
					d.pkg.Path(), imp.Path())
				log.Print(prefix, fmt.Sprintf(format, args...))
			}
		}

		// Read the gob-encoded facts.
		data, err := read(imp.Path())
		if err != nil {
			return nil, fmt.Errorf("in %s, can't import facts for package %q: %v",
				d.pkg.Path(), imp.Path(), err)
		}
		if len(data) == 0 {
			continue // no facts
		}
		var gobFacts []gobFact
		if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&gobFacts); err != nil {
			return nil, fmt.Errorf("decoding facts for %q: %v", imp.Path(), err)
		}
		logf("decoded %d facts: %v", len(gobFacts), gobFacts)

		// Parse each one into a key and a Fact.
		for _, f := range gobFacts {
			factPkg := d.getPackage(f.PkgPath) // possibly an indirect dependency
			if factPkg == nil {
				// Fact relates to a dependency that was
				// unused in this translation unit. Skip.
				logf("no package %q; discarding %v", f.PkgPath, f.Fact)
				continue
			}
			key := key{pkg: factPkg, t: reflect.TypeOf(f.Fact)}
			if f.Object != "" {
				// object fact
				obj, err := typesinternal.ObjectpathObject(factPkg, string(f.Object), skipMethodSorting)
				if err != nil {
					// (most likely due to unexported object)
					// TODO(adonovan): audit for other possibilities.
					logf("no object for path: %v; discarding %s", err, f.Fact)
					continue
				}
				key.obj = obj
				logf("read %T fact %s for %v", f.Fact, f.Fact, key.obj)
			} else {
				// package fact
				logf("read %T fact %s for %v", f.Fact, f.Fact, factPkg)
			}
			m[key] = f.Fact
		}
	}

	return &Set{pkg: d.pkg, m: m}, nil
}

// Encode encodes a set of facts to a memory buffer.
//
// It may fail if one of the Facts could not be gob-encoded, but this is
// a sign of a bug in an Analyzer.
func (s *Set) Encode(skipMethodSorting bool) []byte {
	encoder := new(objectpath.Encoder)
	if skipMethodSorting {
		typesinternal.SkipEncoderMethodSorting(encoder)
	}

	// TODO(adonovan): opt: use a more efficient encoding
	// that avoids repeating PkgPath for each fact.

	// Gather all facts, including those from imported packages.
	var gobFacts []gobFact

	s.mu.Lock()
	for k, fact := range s.m {
		if debug {
			log.Printf("%v => %s\n", k, fact)
		}

		// Don't export facts that we imported from another
		// package, unless they represent fields or methods,
		// or package-level types.
		// (Facts about packages, and other package-level
		// objects, are only obtained from direct imports so
		// they needn't be reexported.)
		//
		// This is analogous to the pruning done by "deep"
		// export data for types, but not as precise because
		// we aren't careful about which structs or methods
		// we rexport: it should be only those referenced
		// from the API of s.pkg.
		// TOOD(adonovan): opt: be more precise. e.g.
		// intersect with the set of objects computed by
		// importMap(s.pkg.Imports()).
		// TOOD(adonovan): opt: implement "shallow" facts.
		if k.pkg != s.pkg {
			if k.obj == nil {
				continue // imported package fact
			}
			if _, isType := k.obj.(*types.TypeName); !isType &&
				k.obj.Parent() == k.obj.Pkg().Scope() {
				continue // imported fact about package-level non-type object
			}
		}

		var object objectpath.Path
		if k.obj != nil {
			path, err := encoder.For(k.obj)
			if err != nil {
				if debug {
					log.Printf("discarding fact %s about %s\n", fact, k.obj)
				}
				continue // object not accessible from package API; discard fact
			}
			object = path
		}
		gobFacts = append(gobFacts, gobFact{
			PkgPath: k.pkg.Path(),
			Object:  object,
			Fact:    fact,
		})
	}
	s.mu.Unlock()

	// Sort facts by (package, object, type) for determinism.
	sort.Slice(gobFacts, func(i, j int) bool {
		x, y := gobFacts[i], gobFacts[j]
		if x.PkgPath != y.PkgPath {
			return x.PkgPath < y.PkgPath
		}
		if x.Object != y.Object {
			return x.Object < y.Object
		}
		tx := reflect.TypeOf(x.Fact)
		ty := reflect.TypeOf(y.Fact)
		if tx != ty {
			return tx.String() < ty.String()
		}
		return false // equal
	})

	var buf bytes.Buffer
	if len(gobFacts) > 0 {
		if err := gob.NewEncoder(&buf).Encode(gobFacts); err != nil {
			// Fact encoding should never fail. Identify the culprit.
			for _, gf := range gobFacts {
				if err := gob.NewEncoder(io.Discard).Encode(gf); err != nil {
					fact := gf.Fact
					pkgpath := reflect.TypeOf(fact).Elem().PkgPath()
					log.Panicf("internal error: gob encoding of analysis fact %s failed: %v; please report a bug against fact %T in package %q",
						fact, err, fact, pkgpath)
				}
			}
		}
	}

	if debug {
		log.Printf("package %q: encode %d facts, %d bytes\n",
			s.pkg.Path(), len(gobFacts), buf.Len())
	}

	return buf.Bytes()
}

// String is provided only for debugging, and must not be called
// concurrent with any Import/Export method.
func (s *Set) String() string {
	var buf bytes.Buffer
	buf.WriteString("{")
	for k, f := range s.m {
		if buf.Len() > 1 {
			buf.WriteString(", ")
		}
		if k.obj != nil {
			buf.WriteString(k.obj.String())
		} else {
			buf.WriteString(k.pkg.Path())
		}
		fmt.Fprintf(&buf, ": %v", f)
	}
	buf.WriteString("}")
	return buf.String()
}
