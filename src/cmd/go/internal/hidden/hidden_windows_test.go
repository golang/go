/*
 * Copyright 2023 The Go Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

package hidden

import (
	"golang.org/x/sys/windows"
	"os"
	"path/filepath"
	"testing"
)

func TestIsHidden(t *testing.T) {
	// Create temporary test files and directories
	tempDir, err := os.MkdirTemp("", "temp")
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
		{
			name:     "hidden attrib file",
			path:     filepath.Join(tempDir, "hidden-attrib.txt"),
			expected: true,
		},
		{
			name:     "hidden attrib folder",
			path:     tempDir,
			expected: true,
		},
	}

	func() { // set hidden attrib to the last testcase
		setHiddenTestcases = testCases[2:]
		for _, testcase := range setHiddenTestcases {
			path, err := windows.UTF16PtrFromString(testcase)
			if err != nil {
				t.Fatalf("UTF16PtrFromString:%s", err)
			}

			// Get the current file attributes
			attrs, err := windows.GetFileAttributes(path)
			if err != nil {
				t.Fatalf("get file attribs:%s", err)
			}

			// Add the hidden attribute to the file
			err = windows.SetFileAttributes(path, attrs|windows.FILE_ATTRIBUTE_HIDDEN)
			if err != nil {
				t.Fatalf("set file attribs:%s", err)
			}
		}

	}()

	// Run the test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
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
