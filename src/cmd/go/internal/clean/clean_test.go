package clean

import (
	"os/exec"
	"strings"
	"testing"
)

func TestCleanCache(t *testing.T) {
	cmd := exec.Command("go", "clean", "-cache")
	// See issue 69997. GOCACHE must be an absolute path.
	cmd.Env = append(cmd.Environ(), "GOCACHE='.cache'")
	_, err := cmd.Output()

	if err != nil {
		ee, ok := err.(*exec.ExitError)
		if !ok || ee.ExitCode() != 1 || !strings.Contains(string(ee.Stderr), "GOCACHE is not an absolute path") {
			t.Errorf("\"go clean -cache\" failed. expected status 1 != %d; error: %s", ee.ExitCode(), ee.Stderr)
		}
	} else {
		t.Errorf("expected go clean -cache to fail")
	}

}
