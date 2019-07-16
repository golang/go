package pkg

import (
	"os"
	"testing"
)

func TestBuilt(t *testing.T) {
	os.Stdout.Write([]byte("A normal test was executed.\n"))
}
