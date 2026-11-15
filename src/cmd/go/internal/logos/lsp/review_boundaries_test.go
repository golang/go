// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestProxyCloseUnsavedDocumentRecompilesRetainedPackage(t *testing.T) {
	const doc1Disk = `package a

import "os"

func Read(path string) (string, error) {
	data := os.ReadFile(path)?
	return string(data), nil
}

func Helper() int {
	return 42
}
`
	const doc2Disk = `package a

func Sibling() int {
	return Helper()
}
`
	const depDisk = `package b

import "example.com/m/a"

func Dependent() int {
	return a.Helper()
}
`

	lowDisk, _, ok := lowerQuestion([]byte(doc1Disk))
	if !ok {
		t.Fatal("lowering doc1Disk failed")
	}

	posHelperDisk := findPos(t, doc1Disk, "Helper", 0)
	rangeHelperDisk := span(posHelperDisk, len("Helper"))

	posLowHelperDisk := findPos(t, string(lowDisk), "Helper", 0)
	rangeLowHelperDisk := span(posLowHelperDisk, len("Helper"))

	f := newFixture(t, map[string]string{
		"a/doc1.lgo": doc1Disk,
		"a/doc2.lgo": doc2Disk,
		"b/b.lgo":    depDisk,
	}, fixtureOptions{
		Script: func(root string) fakeScript {
			doc1Virt := pathURI(virtualPath(filepath.Join(root, "a", "doc1.lgo")))
			return fakeScript{
				Responses: map[string]json.RawMessage{
					"textDocument/definition": toJSON(t, []any{
						map[string]any{
							"uri":   doc1Virt,
							"range": rangeLowHelperDisk,
						},
					}),
				},
			}
		},
		Tune: func(f *fixture, o *Options) {
			f.comp.setClosure(f.path("b"), f.path("a"))
		},
	})

	f.open("a/doc1.lgo", 1)
	f.open("a/doc2.lgo", 1)
	f.open("b/b.lgo", 1)

	const prefix = "\n\n\n\n\n\n\n\n\n\n"
	unsavedDoc1 := prefix + doc1Disk
	f.client.notify("textDocument/didChange", map[string]any{
		"textDocument":   map[string]any{"uri": f.uri("a/doc1.lgo"), "version": 2},
		"contentChanges": []any{map[string]any{"text": unsavedDoc1}},
	})

	posUnsaved := Position{Line: posHelperDisk.Line + 10, Character: posHelperDisk.Character}
	f.client.call("textDocument/hover", positionParams(f.uri("a/doc1.lgo"), posUnsaved))

	f.client.notify("textDocument/didClose", map[string]any{
		"textDocument": map[string]any{"uri": f.uri("a/doc1.lgo")},
	})

	siblingCallPos := findPos(t, doc2Disk, "Helper", 0)
	resSibling := f.client.call("textDocument/definition", positionParams(f.uri("a/doc2.lgo"), siblingCallPos))
	wantDef := []any{
		map[string]any{
			"uri":   f.uri("a/doc1.lgo"),
			"range": rangeHelperDisk,
		},
	}
	sameJSON(t, "sibling definition query reflects disk", resSibling, wantDef)

	depCallPos := findPos(t, depDisk, "Helper", 0)
	resDep := f.client.call("textDocument/definition", positionParams(f.uri("b/b.lgo"), depCallPos))
	sameJSON(t, "dependent unit definition query reflects disk", resDep, wantDef)

	var lastVirtText string
	for _, m := range f.receivedMethod("textDocument/didChange") {
		td, _ := m["textDocument"].(map[string]any)
		if td != nil && td["uri"] == f.virtURI("a/doc1.lgo") {
			changes, _ := m["contentChanges"].([]any)
			if len(changes) > 0 {
				c0, _ := changes[0].(map[string]any)
				if txt, ok := c0["text"].(string); ok {
					lastVirtText = txt
				}
			}
		}
	}
	if lastVirtText != string(lowDisk) {
		t.Errorf("gopls virtual document after close was not restored to disk text:\ngot:\n%s\nwant:\n%s", lastVirtText, string(lowDisk))
	}
}

func TestProxyFormattingRejectsLoweredGoForLogosAndPreservesGo(t *testing.T) {
	const origLgo = `package a

import "os"

func Read(path string) (string, error) {
	data := os.ReadFile(path)?
	return string(data), nil
}
`
	lowDisk, _, ok := lowerQuestion([]byte(origLgo))
	if !ok {
		t.Fatal("lowering origLgo failed")
	}

	t.Run("lgo formatting must not emit lowered Go", func(t *testing.T) {
		lowEdit := []any{
			map[string]any{
				"range":   Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 15, Character: 0}},
				"newText": string(lowDisk),
			},
		}
		f := newFixture(t, map[string]string{
			"a/a.lgo": origLgo,
		}, fixtureOptions{
			Script: func(root string) fakeScript {
				return fakeScript{
					Responses: map[string]json.RawMessage{
						"textDocument/formatting": toJSON(t, lowEdit),
					},
				}
			},
		})

		f.open("a/a.lgo", 1)

		msg := f.client.callMessage("textDocument/formatting", map[string]any{
			"textDocument": map[string]any{"uri": f.uri("a/a.lgo")},
			"options":      map[string]any{"tabSize": 8, "insertSpaces": false},
		})

		if len(msg.Error) == 0 {
			t.Fatalf("formatting on .lgo succeeded with result %s, want request failure -32803", msg.Result)
		}
		var rerr struct {
			Code int `json:"code"`
		}
		if err := json.Unmarshal(msg.Error, &rerr); err != nil {
			t.Fatalf("unmarshaling formatting error: %v", err)
		}
		if rerr.Code != -32803 {
			t.Errorf("formatting error code = %d, want -32803", rerr.Code)
		}
		if reqs := f.receivedMethod("textDocument/formatting"); len(reqs) != 0 {
			t.Errorf("gopls got %d formatting requests for .lgo, want none refused before forwarding", len(reqs))
		}
	})

	t.Run("go formatting is forwarded unchanged", func(t *testing.T) {
		const goSrc = "package g\n\nfunc Foo() {\n}\n"
		goEdits := []any{
			map[string]any{
				"range":   Range{Start: Position{Line: 0, Character: 0}, End: Position{Line: 3, Character: 0}},
				"newText": "package g\n\nfunc Foo() {\n\t// formatted\n}\n",
			},
		}
		f := newFixture(t, map[string]string{
			"g/g.go": goSrc,
		}, fixtureOptions{
			Script: func(root string) fakeScript {
				return fakeScript{
					Responses: map[string]json.RawMessage{
						"textDocument/formatting": toJSON(t, goEdits),
					},
				}
			},
		})

		f.open("g/g.go", 1)

		res := f.client.call("textDocument/formatting", map[string]any{
			"textDocument": map[string]any{"uri": f.uri("g/g.go")},
			"options":      map[string]any{"tabSize": 8, "insertSpaces": false},
		})
		sameJSON(t, "formatting on .go file", res, goEdits)

		reqs := f.receivedMethod("textDocument/formatting")
		if len(reqs) != 1 {
			t.Fatalf("gopls got %d formatting requests, want 1", len(reqs))
		}
		td, _ := reqs[0]["textDocument"].(map[string]any)
		if uri, _ := td["uri"].(string); uri != f.uri("g/g.go") {
			t.Errorf("forwarded formatting URI = %s, want %s", uri, f.uri("g/g.go"))
		}
	})
}
