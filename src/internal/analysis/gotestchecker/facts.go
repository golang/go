// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker

// This file defines [factSet], a serializable set of [analysis.Fact].
// It was adapted from [golang.org/x/tools/internal/facts] for this
// driver, in which each analyzer has its own fact set.
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

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"go/analysis"
	"go/types"
	"io"
	"log"
	"reflect"
	"sort"
	"sync"

	"golang.org/x/tools/go/types/objectpath"
)

const debugFacts = false

// A factSet is a set of [analysis.Fact].
//
// Decode creates a factSet of facts by reading from the imports of a given
// package, and Encode writes out the set. Between these operation,
// the Import and Export methods will query and update the set.
//
// All of factSet's methods except String are safe to call concurrently.
type factSet struct {
	pkg *types.Package
	mu  sync.Mutex
	m   map[factKey]analysis.Fact
}

type factKey struct {
	pkg *types.Package
	obj types.Object // (object facts only)
	t   reflect.Type
}

// ImportObjectFact implements analysis.Pass.ImportObjectFact.
func (s *factSet) ImportObjectFact(obj types.Object, ptr analysis.Fact) bool {
	if obj == nil {
		panic("nil object")
	}
	key := factKey{pkg: obj.Pkg(), obj: obj, t: reflect.TypeOf(ptr)}
	s.mu.Lock()
	defer s.mu.Unlock()
	if v, ok := s.m[key]; ok {
		reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
		return true
	}
	return false
}

// ExportObjectFact implements analysis.Pass.ExportObjectFact.
func (s *factSet) ExportObjectFact(obj types.Object, fact analysis.Fact) {
	if obj.Pkg() != s.pkg {
		log.Panicf("in package %s: ExportObjectFact(%s, %T): can't set fact on object belonging another package",
			s.pkg, obj, fact)
	}
	key := factKey{pkg: obj.Pkg(), obj: obj, t: reflect.TypeOf(fact)}
	s.mu.Lock()
	s.m[key] = fact // clobber any existing entry
	s.mu.Unlock()
}

func (s *factSet) AllObjectFacts(filter map[reflect.Type]bool) []analysis.ObjectFact {
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
func (s *factSet) ImportPackageFact(pkg *types.Package, ptr analysis.Fact) bool {
	if pkg == nil {
		panic("nil package")
	}
	key := factKey{pkg: pkg, t: reflect.TypeOf(ptr)}
	s.mu.Lock()
	defer s.mu.Unlock()
	if v, ok := s.m[key]; ok {
		reflect.ValueOf(ptr).Elem().Set(reflect.ValueOf(v).Elem())
		return true
	}
	return false
}

// ExportPackageFact implements analysis.Pass.ExportPackageFact.
func (s *factSet) ExportPackageFact(fact analysis.Fact) {
	key := factKey{pkg: s.pkg, t: reflect.TypeOf(fact)}
	s.mu.Lock()
	s.m[key] = fact // clobber any existing entry
	s.mu.Unlock()
}

func (s *factSet) AllPackageFacts(filter map[reflect.Type]bool) []analysis.PackageFact {
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

// A factDecoder decodes the facts from the direct imports of the package
// provided to newFactDecoder. A single decoder may be used to decode
// multiple fact sets (e.g. each for a different set of fact types)
// for the same package. Each call to Decode returns an independent
// fact set.
type factDecoder struct {
	pkg        *types.Package
	getPackage getPackageFunc
}

// newFactDecoder returns a fact decoder for the specified package.
//
// It calls the getPackage function for the package path string of
// each dependency (perhaps indirect) that it encounters in the
// encoding. If the function returns nil, the fact is discarded.
//
// Typically getPackage consults the package map of the importer used
// to type-check pkg. Any package not in that map cannot be the owner
// of an object or package that an analyzer could query, so facts
// about it are of no use and may safely be discarded.
func newFactDecoder(pkg *types.Package, getPackage getPackageFunc) *factDecoder {
	return &factDecoder{
		pkg:        pkg,
		getPackage: getPackage,
	}
}

// A getPackageFunc function returns the package denoted by a package path.
type getPackageFunc = func(pkgPath string) *types.Package

// Decode decodes all the facts relevant to the analysis of package
// pkgPath. The read function reads serialized fact data from an external
// source for one of pkg's direct imports, identified by package path.
// The empty file is a valid encoding of an empty fact set.
//
// It is the caller's responsibility to call gob.Register on all
// necessary fact types.
//
// Concurrent calls to Decode are safe, so long as the
// [getPackageFunc] (if any) is also concurrency-safe.
//
// It also returns the number of facts decoded, and the number of those
// discarded because they relate to a package or object not present
// in this translation unit.
func (d *factDecoder) Decode(read func(pkgPath string) ([]byte, error)) (_ *factSet, decoded, discarded int, _ error) {
	// Read facts from imported packages.
	// Facts may describe indirectly imported packages, or their objects.
	m := make(map[factKey]analysis.Fact) // one big bucket
	for _, imp := range d.pkg.Imports() {
		logf := func(format string, args ...any) {
			if debugFacts {
				prefix := fmt.Sprintf("in %s, importing %s: ",
					d.pkg.Path(), imp.Path())
				log.Print(prefix, fmt.Sprintf(format, args...))
			}
		}

		// Read the gob-encoded facts.
		data, err := read(imp.Path())
		if err != nil {
			return nil, 0, 0, fmt.Errorf("in %s, can't import facts for package %q: %v",
				d.pkg.Path(), imp.Path(), err)
		}
		if len(data) == 0 {
			continue // no facts
		}
		var gobFacts []gobFact
		if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&gobFacts); err != nil {
			return nil, 0, 0, fmt.Errorf("decoding facts for %q: %v", imp.Path(), err)
		}
		logf("decoded %d facts: %v", len(gobFacts), gobFacts)
		decoded += len(gobFacts)

		// Parse each one into a key and a Fact.
		for _, f := range gobFacts {
			factPkg := d.getPackage(f.PkgPath) // possibly an indirect dependency
			if factPkg == nil {
				// Fact relates to a dependency that was
				// unused in this translation unit. Skip.
				logf("no package %q; discarding %v", f.PkgPath, f.Fact)
				discarded++
				continue
			}
			key := factKey{pkg: factPkg, t: reflect.TypeOf(f.Fact)}
			if f.Object != "" {
				// object fact
				obj, err := objectpath.Object(factPkg, f.Object)
				if err != nil {
					// (most likely due to unexported object)
					// TODO(adonovan): audit for other possibilities.
					logf("no object for path: %v; discarding %s", err, f.Fact)
					discarded++
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

	return &factSet{pkg: d.pkg, m: m}, decoded, discarded, nil
}

// Encode encodes a set of facts to a memory buffer. It also returns
// the number of facts encoded about the package itself (own) and
// re-exported about objects of its dependencies (deps).
//
// It may fail if one of the Facts could not be gob-encoded, but this is
// a sign of a bug in an Analyzer.
func (s *factSet) Encode() (data []byte, own, deps int) {
	encoder := new(objectpath.Encoder)

	// TODO(adonovan): opt: use a more efficient encoding
	// that avoids repeating PkgPath for each fact.

	// Gather all facts, including those from imported packages.
	var gobFacts []gobFact

	s.mu.Lock()
	for k, fact := range s.m {
		if debugFacts {
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
		// TODO(adonovan): opt: be more precise. e.g.
		// intersect with the set of objects reachable
		// from the API of s.pkg.
		// TODO(adonovan): opt: implement "shallow" facts.
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
				if debugFacts {
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
		if k.pkg == s.pkg {
			own++
		} else {
			deps++
		}
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

	if debugFacts {
		log.Printf("package %q: encode %d facts, %d bytes\n",
			s.pkg.Path(), len(gobFacts), buf.Len())
	}

	return buf.Bytes(), own, deps
}

// String is provided only for debugging, and must not be called
// concurrent with any Import/Export method.
func (s *factSet) String() string {
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
