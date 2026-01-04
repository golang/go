package syntax

import (
	"bytes"
	"strings"
	"testing"
)

func TestStaticTypeParams_ParseAndPrint(t *testing.T) {
	const src = `package p

func F[static T any, U any](t T, u U) {}

type S[static T any, U any] struct {
	t T
	u U
}
`

	f, err := Parse(NewFileBase("p.go"), strings.NewReader(src), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if f == nil {
		t.Fatalf("parse returned nil file")
	}

	var gotFunc, gotType bool
	for _, d := range f.DeclList {
		switch d := d.(type) {
		case *FuncDecl:
			if d.Name.Value != "F" {
				continue
			}
			gotFunc = true
			if len(d.TParamList) != 2 {
				t.Fatalf("func F: got %d type params, want 2", len(d.TParamList))
			}
			if !d.TParamList[0].Static {
				t.Fatalf("func F: T should be static")
			}
			if d.TParamList[1].Static {
				t.Fatalf("func F: U should not be static")
			}
		case *TypeDecl:
			if d.Name.Value != "S" {
				continue
			}
			gotType = true
			if len(d.TParamList) != 2 {
				t.Fatalf("type S: got %d type params, want 2", len(d.TParamList))
			}
			if !d.TParamList[0].Static {
				t.Fatalf("type S: T should be static")
			}
			if d.TParamList[1].Static {
				t.Fatalf("type S: U should not be static")
			}
		}
	}
	if !gotFunc {
		t.Fatalf("missing func F decl")
	}
	if !gotType {
		t.Fatalf("missing type S decl")
	}

	var buf bytes.Buffer
	if _, err := Fprint(&buf, f, LineForm); err != nil {
		t.Fatalf("print failed: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "func F[static T any, U any]") {
		t.Fatalf("printed output missing static tparam: %q", out)
	}
	if !strings.Contains(out, "type S[static T any, U any]") {
		t.Fatalf("printed output missing static tparam on type: %q", out)
	}
}

func TestStaticTypeParams_GroupSemantics(t *testing.T) {
	const src = `package p
func G[static T, U any]() {}
`

	f, err := Parse(NewFileBase("p.go"), strings.NewReader(src), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	var got bool
	for _, d := range f.DeclList {
		fn, ok := d.(*FuncDecl)
		if !ok || fn.Name.Value != "G" {
			continue
		}
		got = true
		if len(fn.TParamList) != 2 {
			t.Fatalf("func G: got %d type params, want 2", len(fn.TParamList))
		}
		if !fn.TParamList[0].Static || !fn.TParamList[1].Static {
			t.Fatalf("func G: expected both T and U to be static due to shared constraint group")
		}
	}
	if !got {
		t.Fatalf("missing func G decl")
	}

	var buf bytes.Buffer
	if _, err := Fprint(&buf, f, LineForm); err != nil {
		t.Fatalf("print failed: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "func G[static T, U any]") {
		t.Fatalf("printed output missing grouped static tparam: %q", out)
	}
	if strings.Contains(out, "static U") {
		t.Fatalf("printed output should only include static once per group: %q", out)
	}
}


