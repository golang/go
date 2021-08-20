package template_test

import (
	"bytes"
	"testing"
	"text/template"
)

func TestParent(t *testing.T) {
	parent, err := template.New("parent").Parse(`{{block "content" .}}parent{{end}}`)
	if err != nil {
		t.Fatalf("parsing parent template: %v", err)
	}

	var b bytes.Buffer
	if err := parent.Execute(&b, nil); err != nil {
		t.Fatalf("executing parent template: %v", err)
	}
	if b.String() != "parent" {
		t.Errorf("want %q, got %q", "parent", b.String())
	}

	var child *template.Template
	{
		clone, err := parent.Clone()
		if err != nil {
			t.Fatalf("cloning parent: %v", err)
		}

		child, err = clone.Parse(`{{define "content"}}{{template parent}}child{{end}}`)
		if err != nil {
			t.Fatalf("parsing child template: %v", err)
		}

		b.Reset()
		if err := child.Execute(&b, nil); err != nil {
			t.Fatalf("executing child template: %v", err)
		}
		if b.String() != "parentchild" {
			t.Errorf("want %q, got %q", "child", b.String())
		}
	}

	{
		clone, err := parent.Clone()
		if err != nil {
			t.Fatalf("cloning parent: %v", err)
		}

		child, err := clone.Parse(`{{define "content"}}{{template parent}}cloned child{{end}}`)
		if err != nil {
			t.Fatalf("parsing child template: %v", err)
		}

		b.Reset()
		if err := child.Execute(&b, nil); err != nil {
			t.Fatalf("executing child template: %v", err)
		}
		if b.String() != "parentcloned child" {
			t.Errorf("want %q, got %q", "parentcloned child", b.String())
		}
	}

	{
		clone, err := child.Clone()
		if err != nil {
			t.Fatalf("cloning child: %v", err)
		}

		gc, err := clone.Parse(`{{define "content"}}{{template parent}}grandchild{{end}}`)
		if err != nil {
			t.Fatalf("parsing grandchild template: %v", err)
		}

		b.Reset()
		if err := gc.Execute(&b, nil); err != nil {
			t.Fatalf("executing grandchild template: %v", err)
		}
		if b.String() != "parentchildgrandchild" {
			t.Errorf("want %q, got %q", "parentchildgrandchild", b.String())
		}
	}

}
