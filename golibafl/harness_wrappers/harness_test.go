//go:build gocov

package main

import (
	"fmt"
	"os"
	"path/filepath"

	"testing"
)

func FuzzMe(f *testing.F) {
	//path := "./output/queue/"
	path := "REPLACE_ME"
	addCorpusFilesAsSeeds(f, path)
	f.Fuzz(func(t *testing.T, input []byte) {
		harness(input)
	})
}

func addCorpusFilesAsSeeds(f *testing.F, path string) {
	info, err := os.Stat(path)
	if err != nil {
		fmt.Printf("Failed to access path %s: %v\n", path, err)
		return
	}
	if info.IsDir() {
		files, err := os.ReadDir(path)
		if err != nil {
			fmt.Printf("Error reading directory %s: %v\n", path, err)
			return
		}

		for _, file := range files {
			filePath := filepath.Join(path, file.Name())
			addCorpusFilesAsSeeds(f, filePath)
		}
	} else {
		content, err := os.ReadFile(path)
		if err != nil {
			fmt.Printf("Error reading file %s: %v\n", path, err)
			return
		}
		f.Add(content)
	}
}
