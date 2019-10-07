package diff_test

import (
	"testing"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/diff/difftest"
)

func TestApplyEdits(t *testing.T) {
	for _, tc := range difftest.TestCases {
		t.Run(tc.Name, func(t *testing.T) {
			t.Helper()
			if got := diff.ApplyEdits(tc.In, tc.Edits); got != tc.Out {
				t.Errorf("ApplyEdits edits got %q, want %q", got, tc.Out)
			}
		})
	}
}
