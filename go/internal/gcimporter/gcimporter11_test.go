// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcimporter

import (
	"go/types"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

var importedObjectTests = []struct {
	name string
	want string
}{
	// non-interfaces
	{"crypto.Hash", "type Hash uint"},
	{"go/ast.ObjKind", "type ObjKind int"},
	{"go/types.Qualifier", "type Qualifier func(*Package) string"},
	{"go/types.Comparable", "func Comparable(T Type) bool"},
	{"math.Pi", "const Pi untyped float"},
	{"math.Sin", "func Sin(x float64) float64"},
	{"go/ast.NotNilFilter", "func NotNilFilter(_ string, v reflect.Value) bool"},
	{"go/internal/gcimporter.FindPkg", "func FindPkg(path string, srcDir string) (filename string, id string)"},

	// interfaces
	{"context.Context", "type Context interface{Deadline() (deadline time.Time, ok bool); Done() <-chan struct{}; Err() error; Value(key interface{}) interface{}}"},
	{"crypto.Decrypter", "type Decrypter interface{Decrypt(rand io.Reader, msg []byte, opts DecrypterOpts) (plaintext []byte, err error); Public() PublicKey}"},
	{"encoding.BinaryMarshaler", "type BinaryMarshaler interface{MarshalBinary() (data []byte, err error)}"},
	{"io.Reader", "type Reader interface{Read(p []byte) (n int, err error)}"},
	{"io.ReadWriter", "type ReadWriter interface{Reader; Writer}"},
	{"go/ast.Node", "type Node interface{End() go/token.Pos; Pos() go/token.Pos}"},
	{"go/types.Type", "type Type interface{String() string; Underlying() Type}"},
}

func TestImportedTypes(t *testing.T) {
	testenv.NeedsGo1Point(t, 11)
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	for _, test := range importedObjectTests {
		s := strings.Split(test.name, ".")
		if len(s) != 2 {
			t.Fatal("inconsistent test data")
		}
		importPath := s[0]
		objName := s[1]

		pkg, err := Import(make(map[string]*types.Package), importPath, ".", nil)
		if err != nil {
			t.Error(err)
			continue
		}

		obj := pkg.Scope().Lookup(objName)
		if obj == nil {
			t.Errorf("%s: object not found", test.name)
			continue
		}

		got := types.ObjectString(obj, types.RelativeTo(pkg))
		if got != test.want {
			t.Errorf("%s: got %q; want %q", test.name, got, test.want)
		}

		if named, _ := obj.Type().(*types.Named); named != nil {
			verifyInterfaceMethodRecvs(t, named, 0)
		}
	}
}

// verifyInterfaceMethodRecvs verifies that method receiver types
// are named if the methods belong to a named interface type.
func verifyInterfaceMethodRecvs(t *testing.T, named *types.Named, level int) {
	// avoid endless recursion in case of an embedding bug that lead to a cycle
	if level > 10 {
		t.Errorf("%s: embeds itself", named)
		return
	}

	iface, _ := named.Underlying().(*types.Interface)
	if iface == nil {
		return // not an interface
	}

	// check explicitly declared methods
	for i := 0; i < iface.NumExplicitMethods(); i++ {
		m := iface.ExplicitMethod(i)
		recv := m.Type().(*types.Signature).Recv()
		if recv == nil {
			t.Errorf("%s: missing receiver type", m)
			continue
		}
		if recv.Type() != named {
			t.Errorf("%s: got recv type %s; want %s", m, recv.Type(), named)
		}
	}

	// check embedded interfaces (if they are named, too)
	for i := 0; i < iface.NumEmbeddeds(); i++ {
		// embedding of interfaces cannot have cycles; recursion will terminate
		if etype, _ := iface.EmbeddedType(i).(*types.Named); etype != nil {
			verifyInterfaceMethodRecvs(t, etype, level+1)
		}
	}
}
func TestIssue25301(t *testing.T) {
	testenv.NeedsGo1Point(t, 11)
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	compileAndImportPkg(t, "issue25301")
}
