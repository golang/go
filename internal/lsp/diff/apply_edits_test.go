package diff

import (
	"testing"

	"golang.org/x/tools/internal/span"
)

func TestApplyEdits(t *testing.T) {
	var testCases = []struct {
		before string
		edits  []TextEdit
		want   string
	}{
		{"", nil, ""},
		{"X", []TextEdit{{newSpan(0, 1), "Y"}}, "Y"},
		{" X ", []TextEdit{{newSpan(1, 2), "Y"}}, " Y "},
		{" X X ", []TextEdit{{newSpan(1, 2), "Y"}, {newSpan(3, 4), "Z"}}, " Y Z "},
	}
	for _, tc := range testCases {
		if got := applyEdits(tc.before, tc.edits); got != tc.want {
			t.Errorf("applyEdits(%v, %v): got %v, want %v", tc.before, tc.edits, got, tc.want)
		}
	}
}

func newSpan(start, end int) span.Span {
	return span.New("", span.NewPoint(0, 0, start), span.NewPoint(0, 0, end))
}
