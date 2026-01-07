//go:build gocov

package harness

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func FuzzMe(f *testing.F) {
	path := "./corpus"
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

		// Go-118-fuzz-build uses go-fuzz-headers to parse the libFuzzer byte input. With this, the first 4 bytes of the input
		// are interpreted as length. Since we add the fuzzing byte input directly here, skip the first 4 bytes.
		if len(content) > 4 {
			f.Add(content[4:])
		}
	}
}
