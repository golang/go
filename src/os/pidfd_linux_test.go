package os_test

import (
	"os"
	"testing"
)

func TestCheckPidfd(t *testing.T) {
	if err := os.CheckPidfdOnce(); err != nil {
		t.Log("checkPidfd:", err)
	} else {
		t.Log("pidfd syscalls work")
	}
	// TODO: make some reasonable assumptions that pidfd must or must not
	// work in the current test environment (for example, it must work for
	// kernel >= 5.4), and fail if pidfdWorks is not as expected.
}
