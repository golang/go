// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

import (
	"bytes"
	"encoding/json"
	"fmt"
	"internal/diff"
	"internal/txtar"
	"path/filepath"
	"strings"
	"testing"
)

func TestTestdata(t *testing.T) {
	files, _ := filepath.Glob("testdata/*.txt")
	if len(files) == 0 {
		t.Fatalf("no testdata")
	}
	var p Parser
	p.Words = map[string]string{
		"italicword": "",
		"linkedword": "https://example.com/linkedword",
	}
	p.LookupPackage = func(name string) (importPath string, ok bool) {
		if name == "comment" {
			return "go/doc/comment", true
		}
		return DefaultLookupPackage(name)
	}
	p.LookupSym = func(recv, name string) (ok bool) {
		if recv == "Parser" && name == "Parse" ||
			recv == "" && name == "Doc" ||
			recv == "" && name == "NoURL" {
			return true
		}
		return false
	}

	stripDollars := func(b []byte) []byte {
		// Remove trailing $ on lines.
		// They make it easier to see lines with trailing spaces,
		// as well as turning them into lines without trailing spaces,
		// in case editors remove trailing spaces.
		return bytes.ReplaceAll(b, []byte("$\n"), []byte("\n"))
	}
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			var pr Printer
			a, err := txtar.ParseFile(file)
			if err != nil {
				t.Fatal(err)
			}
			if len(a.Comment) > 0 {
				err := json.Unmarshal(a.Comment, &pr)
				if err != nil {
					t.Fatalf("unmarshaling top json: %v", err)
				}
			}
			if len(a.Files) < 1 || a.Files[0].Name != "input" {
				t.Fatalf("first file is not %q", "input")
			}
			d := p.Parse(string(stripDollars(a.Files[0].Data)))
			for _, f := range a.Files[1:] {
				want := stripDollars(f.Data)
				for len(want) >= 2 && want[len(want)-1] == '\n' && want[len(want)-2] == '\n' {
					want = want[:len(want)-1]
				}
				var out []byte
				switch f.Name {
				default:
					t.Fatalf("unknown output file %q", f.Name)
				case "dump":
					out = dump(d)
				case "gofmt":
					out = pr.Comment(d)
				case "html":
					out = pr.HTML(d)
				case "markdown":
					out = pr.Markdown(d)
				case "text":
					out = pr.Text(d)
				}
				if string(out) != string(want) {
					t.Errorf("%s: %s", file, diff.Diff(f.Name, want, "have", out))
				}
			}
		})
	}
}

func dump(d *Doc) []byte {
	var out bytes.Buffer
	dumpTo(&out, 0, d)
	return out.Bytes()
}

func dumpTo(out *bytes.Buffer, indent int, x any) {
	switch x := x.(type) {
	default:
		fmt.Fprintf(out, "?%T", x)

	case *Doc:
		fmt.Fprintf(out, "Doc")
		dumpTo(out, indent+1, x.Content)
		if len(x.Links) > 0 {
			dumpNL(out, indent+1)
			fmt.Fprintf(out, "Links")
			dumpTo(out, indent+2, x.Links)
		}
		fmt.Fprintf(out, "\n")

	case []*LinkDef:
		for _, def := range x {
			dumpNL(out, indent)
			dumpTo(out, indent, def)
		}

	case *LinkDef:
		fmt.Fprintf(out, "LinkDef Used:%v Text:%q URL:%s", x.Used, x.Text, x.URL)

	case []Block:
		for _, blk := range x {
			dumpNL(out, indent)
			dumpTo(out, indent, blk)
		}

	case *Heading:
		fmt.Fprintf(out, "Heading")
		dumpTo(out, indent+1, x.Text)

	case *List:
		fmt.Fprintf(out, "List ForceBlankBefore=%v ForceBlankBetween=%v", x.ForceBlankBefore, x.ForceBlankBetween)
		dumpTo(out, indent+1, x.Items)

	case []*ListItem:
		for _, item := range x {
			dumpNL(out, indent)
			dumpTo(out, indent, item)
		}

	case *ListItem:
		fmt.Fprintf(out, "Item Number=%q", x.Number)
		dumpTo(out, indent+1, x.Content)

	case *Paragraph:
		fmt.Fprintf(out, "Paragraph")
		dumpTo(out, indent+1, x.Text)

	case *Code:
		fmt.Fprintf(out, "Code")
		dumpTo(out, indent+1, x.Text)

	case []Text:
		for _, t := range x {
			dumpNL(out, indent)
			dumpTo(out, indent, t)
		}

	case Plain:
		if !strings.Contains(string(x), "\n") {
			fmt.Fprintf(out, "Plain %q", string(x))
		} else {
			fmt.Fprintf(out, "Plain")
			dumpTo(out, indent+1, string(x))
		}

	case Italic:
		if !strings.Contains(string(x), "\n") {
			fmt.Fprintf(out, "Italic %q", string(x))
		} else {
			fmt.Fprintf(out, "Italic")
			dumpTo(out, indent+1, string(x))
		}

	case string:
		for _, line := range strings.SplitAfter(x, "\n") {
			if line != "" {
				dumpNL(out, indent)
				fmt.Fprintf(out, "%q", line)
			}
		}

	case *Link:
		fmt.Fprintf(out, "Link %q", x.URL)
		dumpTo(out, indent+1, x.Text)

	case *DocLink:
		fmt.Fprintf(out, "DocLink pkg:%q, recv:%q, name:%q", x.ImportPath, x.Recv, x.Name)
		dumpTo(out, indent+1, x.Text)
	}
}

func dumpNL(out *bytes.Buffer, n int) {
	out.WriteByte('\n')
	for i := 0; i < n; i++ {
		out.WriteByte('\t')
	}
}
