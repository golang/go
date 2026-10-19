#!/bin/bash
set -euo pipefail
if [ -e ./migrateme ] && [ ! -f ./migrateme/.scaffolded ]; then
  echo "scaffold: ./migrateme exists and was not created by this scaffold; refusing to overwrite" >&2
  exit 1
fi
rm -rf ./migrateme
mkdir -p ./migrateme
touch ./migrateme/.scaffolded
cat > ./migrateme/go.mod <<'EOF'
module example.com/migrateme

go 1.27
EOF
cat > ./migrateme/main.go <<'EOF'
package main

import (
	"fmt"
	"os"
)

func readFirst(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func main() {
	line, err := readFirst("data.txt")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Print(line)
}
EOF
cat > ./migrateme/main_test.go <<'EOF'
package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadFirstMissing(t *testing.T) {
	if _, err := readFirst(filepath.Join(t.TempDir(), "nope.txt")); err == nil {
		t.Fatal("want error for missing file")
	}
}

func TestReadFirstOK(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ok.txt")
	if err := os.WriteFile(p, []byte("hi"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := readFirst(p)
	if err != nil {
		t.Fatal(err)
	}
	if got != "hi" {
		t.Fatalf("got %q", got)
	}
}
EOF
