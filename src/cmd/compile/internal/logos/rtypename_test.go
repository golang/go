// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type nameT struct{ A int }

type nameBox[T any] struct{ v T }

type namePair[K comparable, V any] struct {
	k K
	v V
}

type nameIface interface{ M(int) string }

type nameCasePtr *time.Time

func nameLocalType() reflect.Type {
	type local struct{ a int }
	return reflect.TypeFor[local]()
}

var staticNameCases = []struct {
	src string
	typ reflect.Type
}{
	{"int", reflect.TypeFor[int]()},
	{"byte", reflect.TypeFor[byte]()},
	{"rune", reflect.TypeFor[rune]()},
	{"uintptr", reflect.TypeFor[uintptr]()},
	{"complex128", reflect.TypeFor[complex128]()},
	{"error", reflect.TypeFor[error]()},
	{"any", reflect.TypeFor[any]()},
	{"*int", reflect.TypeFor[*int]()},
	{"[]string", reflect.TypeFor[[]string]()},
	{"[]byte", reflect.TypeFor[[]byte]()},
	{"[3]nameT", reflect.TypeFor[[3]nameT]()},
	{"map[string][]rune", reflect.TypeFor[map[string][]rune]()},
	{"*map[string]int", reflect.TypeFor[*map[string]int]()},
	{"*[]int", reflect.TypeFor[*[]int]()},
	{"*func(int) string", reflect.TypeFor[*func(int) string]()},
	{"*any", reflect.TypeFor[*any]()},
	{"chan int", reflect.TypeFor[chan int]()},
	{"<-chan int", reflect.TypeFor[<-chan int]()},
	{"chan<- int", reflect.TypeFor[chan<- int]()},
	{"chan (<-chan int)", reflect.TypeFor[chan (<-chan int)]()},
	{"func()", reflect.TypeFor[func()]()},
	{"func(a, b int) string", reflect.TypeFor[func(a, b int) string]()},
	{"func(int, ...string) (bool, error)", reflect.TypeFor[func(int, ...string) (bool, error)]()},
	{"struct{}", reflect.TypeFor[struct{}]()},
	{`struct{ X int "json:\"x\""; y string }`, reflect.TypeFor[struct {
		X int "json:\"x\""
		y string
	}]()},
	{"struct{ nameT }", reflect.TypeFor[struct{ nameT }]()},
	{"interface{ M(int) string }", reflect.TypeFor[interface{ M(int) string }]()},
	{"nameIface", reflect.TypeFor[nameIface]()},
	{"fmt.Stringer", reflect.TypeFor[fmt.Stringer]()},
	{"time.Duration", reflect.TypeFor[time.Duration]()},
	{"*time.Time", reflect.TypeFor[*time.Time]()},
	{"nameCasePtr", reflect.TypeFor[nameCasePtr]()},
	{"nameBox[int]", reflect.TypeFor[nameBox[int]]()},
	{"*nameBox[string]", reflect.TypeFor[*nameBox[string]]()},
	{"nameBox[nameT]", reflect.TypeFor[nameBox[nameT]]()},
	{"nameBox[time.Duration]", reflect.TypeFor[nameBox[time.Duration]]()},
	{"nameBox[[]byte]", reflect.TypeFor[nameBox[[]byte]]()},
	{"nameBox[any]", reflect.TypeFor[nameBox[any]]()},
	{"nameBox[error]", reflect.TypeFor[nameBox[error]]()},
	{"nameBox[struct{}]", reflect.TypeFor[nameBox[struct{}]]()},
	{"nameBox[map[string]nameT]", reflect.TypeFor[nameBox[map[string]nameT]]()},
	{"nameBox[func(int) error]", reflect.TypeFor[nameBox[func(int) error]]()},
	{"nameBox[nameBox[int]]", reflect.TypeFor[nameBox[nameBox[int]]]()},
	{"namePair[string, *time.Time]", reflect.TypeFor[namePair[string, *time.Time]]()},
}

const nameSnippetHead = `package logos

import (
	"fmt"
	"time"
)

type nameT struct{ A int }

type nameBox[T any] struct{ v T }

type namePair[K comparable, V any] struct {
	k K
	v V
}

type nameIface interface{ M(int) string }

type nameCasePtr *time.Time

var _ fmt.Stringer
var _ time.Duration

func generic[E any]() {
	type inner int
	var dynSlice []E
	var dynInner inner
	var dynBox nameBox[E]
	_, _, _ = dynSlice, dynInner, dynBox
}

func plain() {
	type local struct{ a int }
	var staticLocal local
	var dynBoxed nameBox[local]
	_, _ = staticLocal, dynBoxed
}

`

func checkNameSnippet(t *testing.T) *types2.Info {
	t.Helper()
	var src strings.Builder
	src.WriteString(nameSnippetHead)
	for i, c := range staticNameCases {
		src.WriteString("var v" + strconv.Itoa(i) + " " + c.src + "\n")
	}
	f, err := syntax.Parse(syntax.NewFileBase("typename.go"), strings.NewReader(src.String()), nil, nil, syntax.CheckBranches)
	if err != nil {
		t.Fatal(err)
	}
	info := &types2.Info{
		Defs: make(map[*syntax.Name]types2.Object),
	}
	var cerrs []error
	conf := types2.Config{
		Importer: newTestImporter(t),
		Error:    func(err error) { cerrs = append(cerrs, err) },
	}
	if _, err := conf.Check("cmd/compile/internal/logos", []*syntax.File{f}, info); err != nil {
		t.Fatalf("type check: %v %v", err, cerrs)
	}
	return info
}

func snippetType(t *testing.T, info *types2.Info, name string) types2.Type {
	t.Helper()
	for id, obj := range info.Defs {
		if obj != nil && id.Value == name {
			return obj.Type()
		}
	}
	t.Fatalf("snippet declares no %s", name)
	return nil
}

func TestStaticTypeNameMatchesReflect(t *testing.T) {
	info := checkNameSnippet(t)
	for i, c := range staticNameCases {
		got, ok := staticTypeName(snippetType(t, info, "v"+strconv.Itoa(i)))
		if !ok {
			t.Errorf("%s: no static name, want %q", c.src, c.typ.String())
			continue
		}
		if got != c.typ.String() {
			t.Errorf("%s: static name %q, reflect says %q", c.src, got, c.typ.String())
		}
	}
	if got, ok := staticTypeName(snippetType(t, info, "staticLocal")); !ok || got != nameLocalType().String() {
		t.Errorf("function-scope type: static name %q (%v), reflect says %q", got, ok, nameLocalType().String())
	}
}

func TestStaticTypeNameDefersTypeDependentTypes(t *testing.T) {
	info := checkNameSnippet(t)
	for _, name := range []string{"dynSlice", "dynInner", "dynBox", "dynBoxed"} {
		if got, ok := staticTypeName(snippetType(t, info, name)); ok {
			t.Errorf("%s: static name %q, want none because the name depends on an instance or a linker number", name, got)
		}
	}
}

func TestStaticTypeNameLitQuotes(t *testing.T) {
	info := checkNameSnippet(t)
	for i, c := range staticNameCases {
		want := strconv.Quote(c.typ.String())
		if got := staticTypeNameLit(snippetType(t, info, "v"+strconv.Itoa(i))); got != want {
			t.Errorf("%s: literal %s, want %s", c.src, got, want)
		}
	}
	if got := staticTypeNameLit(snippetType(t, info, "dynSlice")); got != `""` {
		t.Errorf("type-dependent type: literal %s, want the empty literal", got)
	}
	if got := staticTypeNameLit(nil); got != `""` {
		t.Errorf("nil type: literal %s, want the empty literal", got)
	}
}
