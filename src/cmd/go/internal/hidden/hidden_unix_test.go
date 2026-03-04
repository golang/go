/*
 * Copyright 2023 The Go Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

package hidden

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsHidden(t *testing.T) {
	// Create temporary test files and directories
	tempDir, err := os.MkdirTemp("", "testcases")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	t.Cleanup(func() {
		os.RemoveAll(tempDir)
	})

	// Test cases
	testCases := []struct {
		name     string
		path     string
		expected bool
	}{
		{
			name:     "non-hidden file",
			path:     filepath.Join(tempDir, "file.txt"),
			expected: false,
		},
		{
			name:     "hidden file",
			path:     filepath.Join(tempDir, ".hidden.txt"),
			expected: true,
		},
	}

	// Run the test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create the test file
			f, err := os.Create(tc.path)
			if err != nil {
				t.Fatalf("Failed to create test file: %v", err)
			}
			f.Close()

			hidden, err := IsHidden(tc.path)

			if err != nil {
				t.Fatalf("isHidden: %s", err)
			}

			// Check the result
			if hidden != tc.expected {
				t.Errorf("Expected %v, got %v", tc.expected, hidden)
			}
		})
	}
}
