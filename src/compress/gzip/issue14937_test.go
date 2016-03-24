package gzip

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Per golang.org/issue/14939, check that every .gz file
// in the tree has a zero mtime.
func TestGZIPFilesHaveZeroMTimes(t *testing.T) {
	var files []string
	err := filepath.Walk(runtime.GOROOT(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(path, ".gz") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal("error collecting list of .gz files in GOROOT: ", err)
	}
	if len(files) == 0 {
		t.Fatal("expected to find some .gz files under GOROOT")
	}
	for _, path := range files {
		checkZeroMTime(t, path)
	}
}

func checkZeroMTime(t *testing.T, path string) {
	f, err := os.Open(path)
	if err != nil {
		t.Error(err)
		return
	}
	defer f.Close()
	gz, err := NewReader(f)
	if err != nil {
		t.Errorf("cannot read gzip file %s: %s", path, err)
		return
	}
	defer gz.Close()
	if !gz.ModTime.Equal(time.Unix(0, 0)) {
		t.Errorf("gzip file %s has non-zero mtime (%s)", path, gz.ModTime)
	}
}
