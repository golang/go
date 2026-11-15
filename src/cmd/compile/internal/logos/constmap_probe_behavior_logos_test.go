// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func constmapTableLookupText(name string, keys []constMapKey, layout *constMapLayout) string {
	keyTexts := make([]string, len(keys))
	valTexts := make([]string, len(keys))
	pairs := make([]string, len(keys))
	for i := range keys {
		keyTexts[i] = `"` + keys[i].canon + `"`
		valTexts[i] = `"` + keys[i].canon + `"`
		pairs[i] = keyTexts[i] + ": " + valTexts[i]
	}
	order := make([]int, len(keys))
	for i := range order {
		order[i] = i
	}
	return emitConstMapDecls(constMapEmission{
		name:        name,
		kT:          "string",
		vT:          "string",
		mapT:        "map[string]string",
		keyTexts:    keyTexts,
		valTexts:    valTexts,
		builderBody: "map[string]string{" + strings.Join(pairs, ", ") + "}",
		hashExpr:    "_logos_rt.HashString(_logos_k)",
		order:       order,
		layout:      layout,
		isSwitch:    false,
		bearing:     false,
	})
}

func constmapMissKeys(t *testing.T, keys []constMapKey, l *constMapLayout) (string, string) {
	t.Helper()
	collision, unoccupied := "", ""
	for i := range 1 << 20 {
		if collision != "" && unoccupied != "" {
			break
		}
		c := fmt.Sprintf("__miss%d__", i)
		s := l.probe(constHashString(c))
		if l.slots[s] < 0 {
			if unoccupied == "" {
				unoccupied = c
			}
			continue
		}
		if collision == "" && keys[l.slots[s]].canon != c {
			collision = c
		}
	}
	if collision == "" || unoccupied == "" {
		t.Fatalf("no miss keys found (collision %q unoccupied %q)", collision, unoccupied)
	}
	return collision, unoccupied
}

type constmapMissCase struct {
	text       string
	lookup     string
	keys       []constMapKey
	collision  string
	unoccupied string
}

func runConstmapMissBehavior(t *testing.T, mc constmapMissCase) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	var b strings.Builder
	b.WriteString("package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n\n\t_logos_rt \"logos/rt\"\n)\n\n")
	b.WriteString(mc.text)
	b.WriteString("\nfunc main() {\n")
	for i := range mc.keys {
		fmt.Fprintf(&b, "\tif v, ok := testtab_lookup(%q); !ok || v != %q {\n\t\tfmt.Printf(\"FAIL present %%q = %%q,%%v\\n\", %q, v, ok)\n\t\tos.Exit(1)\n\t}\n", mc.keys[i].canon, mc.keys[i].canon, mc.keys[i].canon)
	}
	fmt.Fprintf(&b, "\tif v, ok := testtab_lookup(%q); ok || v != \"\" {\n\t\tfmt.Printf(\"FAIL collision %%q = %%q,%%v\\n\", %q, v, ok)\n\t\tos.Exit(1)\n\t}\n", mc.collision, mc.collision)
	fmt.Fprintf(&b, "\tif v, ok := testtab_lookup(%q); ok || v != \"\" {\n\t\tfmt.Printf(\"FAIL unoccupied %%q = %%q,%%v\\n\", %q, v, ok)\n\t\tos.Exit(1)\n\t}\n", mc.unoccupied, mc.unoccupied)
	b.WriteString("\tfmt.Println(\"ok constmap-miss\")\n}\n")
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module constmapmiss\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "run", ".")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil || strings.TrimSpace(string(out)) != "ok constmap-miss" {
		t.Fatalf("%d keys: lookup behavior = %q, err %v, want ok constmap-miss", len(mc.keys), out, err)
	}
}

func TestConstmapTableLookupMissesFullKey(t *testing.T) {
	for _, n := range []int{9, 64, 4096} {
		keys := constmapKeys(n)
		layout, err := buildConstMap(keys, defaultConstMapBudget)
		if err != nil {
			t.Fatalf("%d keys: %v", n, err)
		}
		name := "testtab"
		text := constmapTableLookupText(name, keys, layout)
		idx := strings.Index(text, "func "+name+"_lookup")
		if idx < 0 {
			t.Fatalf("%d keys: table decls lack lookup:\n%s", n, text)
		}
		lookup := text[idx:]
		collision, unoccupied := constmapMissKeys(t, keys, layout)
		runConstmapMissBehavior(t, constmapMissCase{
			text:       text,
			lookup:     lookup,
			keys:       keys,
			collision:  collision,
			unoccupied: unoccupied,
		})
	}
}
