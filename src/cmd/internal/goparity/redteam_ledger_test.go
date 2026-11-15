// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goparity

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

var ledgerStatuses = []string{"planned", "tested", "satisfied upstream", "blocked"}

var ledgerPaths = regexp.MustCompile(`(?:src|test|lib)/[A-Za-z0-9_./+-]*[A-Za-z0-9_]`)

type ledgerRow struct {
	id     string
	test   string
	status string
}

func ledgerRows(t *testing.T) []ledgerRow {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(forkGoroot(), "logos-doc", "REDTEAM.md"))
	if err != nil {
		t.Fatal(err)
	}
	var rows []ledgerRow
	for line := range strings.SplitSeq(string(data), "\n") {
		if !strings.HasPrefix(line, "| #") {
			continue
		}
		body, _ := strings.CutPrefix(line, "| ")
		body, _ = strings.CutSuffix(body, " |")
		cells := strings.Split(body, " | ")
		if len(cells) != 5 {
			t.Errorf("ledger row has %d cells, want 5: %.60s", len(cells), line)
			continue
		}
		rows = append(rows, ledgerRow{id: cells[0], test: cells[3], status: cells[4]})
	}
	if len(rows) == 0 {
		t.Fatal("logos-doc/REDTEAM.md holds no attack rows")
	}
	return rows
}

func ledgerPathExists(p string) bool {
	full := filepath.Join(forkGoroot(), filepath.FromSlash(p))
	if _, err := os.Stat(full); err == nil {
		return true
	}
	matches, err := filepath.Glob(full + ".*")
	return err == nil && len(matches) > 0
}

func TestRedteamLedgerRows(t *testing.T) {
	seen := make(map[string]bool)
	for _, r := range ledgerRows(t) {
		if seen[r.id] {
			t.Errorf("%s appears twice", r.id)
		}
		seen[r.id] = true
		if !slices.Contains(ledgerStatuses, r.status) {
			t.Errorf("%s: status %q is not one of %v", r.id, r.status, ledgerStatuses)
			continue
		}
		switch r.status {
		case "tested", "satisfied upstream":
			paths := ledgerPaths.FindAllString(r.test, -1)
			if len(paths) == 0 {
				t.Errorf("%s: a %s row names no proving test path", r.id, r.status)
			}
			for _, p := range paths {
				if !ledgerPathExists(p) {
					t.Errorf("%s: proving test %s does not exist", r.id, p)
				}
			}
		case "blocked":
			for _, want := range []string{"Mechanism:", "Gate:"} {
				if !strings.Contains(r.test, want) {
					t.Errorf("%s: a blocked row lacks %q", r.id, want)
				}
			}
		}
	}
}

func TestRedteamLedgerRelease(t *testing.T) {
	if !*releaseFlag {
		t.Skip("only the release tier needs a ledger with no planned row")
	}
	for _, r := range ledgerRows(t) {
		if r.status == "planned" {
			t.Errorf("%s is planned: a release holds no planned row", r.id)
		}
	}
}
