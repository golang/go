// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func markedTrigger(body, id, name string) bool {
	for line := range strings.Lines(body) {
		if !strings.Contains(line, "ERROR") {
			continue
		}
		if strings.Contains(line, id) || strings.Contains(line, `"`+name) {
			return true
		}
	}
	return false
}

func TestGoRulesTriggerFromGo(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	lintDir := filepath.Join(testenv.GOROOT(t), "test", "logos", "lint")
	for _, id := range goRules {
		_, name, _ := strings.Cut(id, "::")
		hit := ""
		for _, cand := range []string{name + "_go.go", name + ".go"} {
			body, err := os.ReadFile(filepath.Join(lintDir, cand))
			if err != nil {
				continue
			}
			if markedTrigger(string(body), id, name) {
				hit = cand
				break
			}
		}
		if hit == "" {
			t.Errorf("rule %s has no Go trigger fixture in test/logos/lint", id)
		}
	}
}
