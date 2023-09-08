// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/build/constraint"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/internal/bisect"
	"golang.org/x/tools/internal/compat"
	"golang.org/x/tools/internal/diffp"
	"golang.org/x/tools/txtar"
)

var update = flag.Bool("update", false, "update testdata with new stdout/stderr")

func Test(t *testing.T) {
	files, err := filepath.Glob("testdata/*.txt")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		t.Run(strings.TrimSuffix(filepath.Base(file), ".txt"), func(t *testing.T) {
			data, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			a := txtar.Parse(data)
			var wantStdout, wantStderr []byte
			files := a.Files
			if len(files) > 0 && files[0].Name == "stdout" {
				wantStdout = files[0].Data
				files = files[1:]
			}
			if len(files) > 0 && files[0].Name == "stderr" {
				wantStderr = files[0].Data
				files = files[1:]
			}
			if len(files) > 0 {
				t.Fatalf("unexpected txtar entry: %s", files[0].Name)
			}

			var tt struct {
				Fail   string
				Bisect Bisect
			}
			if err := json.Unmarshal(a.Comment, &tt); err != nil {
				t.Fatal(err)
			}

			expr, err := constraint.Parse("//go:build " + tt.Fail)
			if err != nil {
				t.Fatalf("invalid Cmd: %v", err)
			}

			rnd := rand.New(rand.NewSource(1))
			b := &tt.Bisect
			b.Cmd = "test"
			b.Args = []string{"PATTERN"}
			var stdout, stderr bytes.Buffer
			b.Stdout = &stdout
			b.Stderr = &stderr
			b.TestRun = func(env []string, cmd string, args []string) (out []byte, err error) {
				pattern := args[0]
				m, err := bisect.New(pattern)
				if err != nil {
					t.Fatal(err)
				}
				have := make(map[string]bool)
				for i, color := range colors {
					if m.ShouldEnable(uint64(i)) {
						have[color] = true
					}
					if m.ShouldReport(uint64(i)) {
						out = compat.Appendf(out, "%s %s\n", color, bisect.Marker(uint64(i)))
					}
				}
				err = nil
				if eval(rnd, expr, have) {
					err = fmt.Errorf("failed")
				}
				return out, err
			}

			if !b.Search() {
				stderr.WriteString("<bisect failed>\n")
			}
			rewrite := false
			if !bytes.Equal(stdout.Bytes(), wantStdout) {
				if *update {
					rewrite = true
				} else {
					t.Errorf("incorrect stdout: %s", diffp.Diff("have", stdout.Bytes(), "want", wantStdout))
				}
			}
			if !bytes.Equal(stderr.Bytes(), wantStderr) {
				if *update {
					rewrite = true
				} else {
					t.Errorf("incorrect stderr: %s", diffp.Diff("have", stderr.Bytes(), "want", wantStderr))
				}
			}
			if rewrite {
				a.Files = []txtar.File{{Name: "stdout", Data: stdout.Bytes()}, {Name: "stderr", Data: stderr.Bytes()}}
				err := os.WriteFile(file, txtar.Format(a), 0666)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("updated %s", file)
			}
		})
	}
}

func eval(rnd *rand.Rand, z constraint.Expr, have map[string]bool) bool {
	switch z := z.(type) {
	default:
		panic(fmt.Sprintf("unexpected type %T", z))
	case *constraint.NotExpr:
		return !eval(rnd, z.X, have)
	case *constraint.AndExpr:
		return eval(rnd, z.X, have) && eval(rnd, z.Y, have)
	case *constraint.OrExpr:
		return eval(rnd, z.X, have) || eval(rnd, z.Y, have)
	case *constraint.TagExpr:
		if z.Tag == "random" {
			return rnd.Intn(2) == 1
		}
		return have[z.Tag]
	}
}

var colors = strings.Fields(`
	aliceblue
	amaranth
	amber
	amethyst
	applegreen
	applered
	apricot
	aquamarine
	azure
	babyblue
	beige
	brickred
	black
	blue
	bluegreen
	blueviolet
	blush
	bronze
	brown
	burgundy
	byzantium
	carmine
	cerise
	cerulean
	champagne
	chartreusegreen
	chocolate
	cobaltblue
	coffee
	copper
	coral
	crimson
	cyan
	desertsand
	electricblue
	emerald
	erin
	gold
	gray
	green
	harlequin
	indigo
	ivory
	jade
	junglegreen
	lavender
	lemon
	lilac
	lime
	magenta
	magentarose
	maroon
	mauve
	navyblue
	ochre
	olive
	orange
	orangered
	orchid
	peach
	pear
	periwinkle
	persianblue
	pink
	plum
	prussianblue
	puce
	purple
	raspberry
	red
	redviolet
	rose
	ruby
	salmon
	sangria
	sapphire
	scarlet
	silver
	slategray
	springbud
	springgreen
	tan
	taupe
	teal
	turquoise
	ultramarine
	violet
	viridian
	white
	yellow
`)
