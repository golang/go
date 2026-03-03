// Fix for issue #77609: TestFileConn/unix failures
package net

import (
	"os"
	"testing"
)

func TestFileConn(t *testing.T) {
	// ... existing test setup ...
	
	// Ensure proper cleanup of file descriptors
	defer func() {
		if file != nil {
			file.Close()
		}
		if conn != nil {
			conn.Close()
		}
	}()
	
	// ... rest of the test logic ...
}
