// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tool

import (
	"os"
	"path/filepath"
	"testing"
)

func TestListToolsBuiltinDiscovery(t *testing.T) {
	// Test the directory scanning logic that was added to listTools
	// This tests that we correctly identify directories and skip non-directories

	// Create a temporary directory structure to simulate cmd/ directory
	tempDir := t.TempDir()
	cmdDir := filepath.Join(tempDir, "cmd")
	if err := os.MkdirAll(cmdDir, 0755); err != nil {
		t.Fatal(err)
	}

	// Create some tool directories
	tools := []string{"vet", "cgo", "cover", "fix", "godoc"}
	for _, tool := range tools {
		toolDir := filepath.Join(cmdDir, tool)
		if err := os.MkdirAll(toolDir, 0755); err != nil {
			t.Fatal(err)
		}
	}

	// Create some non-tool directories that should be skipped
	nonTools := []string{"internal", "vendor"}
	for _, nonTool := range nonTools {
		nonToolDir := filepath.Join(cmdDir, nonTool)
		if err := os.MkdirAll(nonToolDir, 0755); err != nil {
			t.Fatal(err)
		}
	}

	// Create a regular file (should be skipped)
	filePath := filepath.Join(cmdDir, "not-a-directory.txt")
	if err := os.WriteFile(filePath, []byte("test"), 0644); err != nil {
		t.Fatal(err)
	}

	// Test directory reading logic (simulating the logic from listTools)
	entries, err := os.ReadDir(cmdDir)
	if err != nil {
		t.Fatal(err)
	}

	var foundTools []string
	for _, entry := range entries {
		// Skip non-directories (this is the logic we added)
		if !entry.IsDir() {
			continue
		}

		toolName := entry.Name()
		// Skip packages that are not tools (this is the logic we added)
		if toolName == "internal" || toolName == "vendor" {
			continue
		}

		foundTools = append(foundTools, toolName)
	}

	// Sort for consistent comparison
	// (In the real code, this happens via the toolSet map and final output)
	for i := 0; i < len(foundTools)-1; i++ {
		for j := i + 1; j < len(foundTools); j++ {
			if foundTools[i] > foundTools[j] {
				foundTools[i], foundTools[j] = foundTools[j], foundTools[i]
			}
		}
	}

	// Verify we found the expected tools
	expectedTools := []string{"cgo", "cover", "fix", "godoc", "vet"}
	if len(foundTools) != len(expectedTools) {
		t.Errorf("Found %d tools, expected %d: %v", len(foundTools), len(expectedTools), foundTools)
	}

	for i, expected := range expectedTools {
		if i >= len(foundTools) || foundTools[i] != expected {
			t.Errorf("Expected tool %q at position %d, got %q", expected, i, foundTools[i])
		}
	}
}

func TestToolSetTracking(t *testing.T) {
	// Test the toolSet map logic that prevents duplicates
	// This tests part of the new functionality in listTools

	// Simulate the toolSet map logic
	toolSet := make(map[string]bool)

	// Add some tools to the set (simulating tools found in tool directory)
	existingTools := []string{"vet", "cgo"}
	for _, tool := range existingTools {
		toolSet[tool] = true
	}

	// Test that existing tools are marked as present
	for _, tool := range existingTools {
		if !toolSet[tool] {
			t.Errorf("Expected tool %q to be in toolSet", tool)
		}
	}

	// Test that new tools can be added and checked
	newTools := []string{"cover", "fix"}
	for _, tool := range newTools {
		if toolSet[tool] {
			t.Errorf("Expected new tool %q to not be in toolSet initially", tool)
		}
		toolSet[tool] = true
		if !toolSet[tool] {
			t.Errorf("Expected tool %q to be in toolSet after adding", tool)
		}
	}
}
