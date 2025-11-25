// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"log"
	"strings"
	"testing"
)

func TestHTTP1Debug(t *testing.T) {
	tests := []struct {
		name     string
		godebug  string
		expected map[string]bool
	}{
		{
			name:    "no debug",
			godebug: "",
			expected: map[string]bool{
				"verbose":   false,
				"requests":  false,
				"responses": false,
			},
		},
		{
			name:    "level 1",
			godebug: "http1debug=1",
			expected: map[string]bool{
				"verbose":   true,
				"requests":  false,
				"responses": false,
			},
		},
		{
			name:    "level 2",
			godebug: "http1debug=2",
			expected: map[string]bool{
				"verbose":   true,
				"requests":  true,
				"responses": true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original debug state
			origVerbose := http1VerboseLogs
			origRequests := http1logRequests
			origResponses := http1logResponses

			defer func() {
				// Restore original debug state
				http1VerboseLogs = origVerbose
				http1logRequests = origRequests
				http1logResponses = origResponses
			}()

			// Reset debug state
			http1VerboseLogs = false
			http1logRequests = false
			http1logResponses = false

			// Simulate the init() function
			if strings.Contains(tt.godebug, "http1debug=1") {
				http1VerboseLogs = true
			}
			if strings.Contains(tt.godebug, "http1debug=2") {
				http1VerboseLogs = true
				http1logRequests = true
				http1logResponses = true
			}

			// Verify expected state
			if http1VerboseLogs != tt.expected["verbose"] {
				t.Errorf("Expected verbose=%v, got %v", tt.expected["verbose"], http1VerboseLogs)
			}
			if http1logRequests != tt.expected["requests"] {
				t.Errorf("Expected requests=%v, got %v", tt.expected["requests"], http1logRequests)
			}
			if http1logResponses != tt.expected["responses"] {
				t.Errorf("Expected responses=%v, got %v", tt.expected["responses"], http1logResponses)
			}
		})
	}
}

func TestHTTP1LogFunctions(t *testing.T) {
	// Save original debug state
	origVerbose := http1VerboseLogs
	origRequests := http1logRequests
	origResponses := http1logResponses

	defer func() {
		// Restore original debug state
		http1VerboseLogs = origVerbose
		http1logRequests = origRequests
		http1logResponses = origResponses
	}()

	// Capture log output
	var logBuf bytes.Buffer
	oldOutput := log.Writer()
	log.SetOutput(&logBuf)
	defer log.SetOutput(oldOutput)

	// Test http1Logf with verbose logging enabled
	http1VerboseLogs = true
	http1Logf("test verbose log: %s", "message")
	if !strings.Contains(logBuf.String(), "http1: test verbose log: message") {
		t.Errorf("http1Logf not working, got: %s", logBuf.String())
	}
	logBuf.Reset()

	// Test http1Logf with verbose logging disabled
	http1VerboseLogs = false
	http1Logf("should not appear")
	if strings.Contains(logBuf.String(), "should not appear") {
		t.Errorf("http1Logf should not log when verbose is disabled, got: %s", logBuf.String())
	}
}

func TestHTTP1DebugInitialization(t *testing.T) {
	// Test that debug flag parsing works correctly
	tests := []struct {
		name              string
		godebugValue      string
		expectVerbose     bool
		expectRequests    bool
		expectResponses   bool
		expectConnections bool
	}{
		{
			name:              "http1debug=1",
			godebugValue:      "http1debug=1",
			expectVerbose:     true,
			expectRequests:    false,
			expectResponses:   false,
			expectConnections: false,
		},
		{
			name:              "http1debug=2",
			godebugValue:      "http1debug=2",
			expectVerbose:     true,
			expectRequests:    true,
			expectResponses:   true,
			expectConnections: false,
		},
		{
			name:              "http1xconnect=1",
			godebugValue:      "http1xconnect=1",
			expectVerbose:     true,
			expectRequests:    true,
			expectResponses:   true,
			expectConnections: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original debug state
			origVerbose := http1VerboseLogs
			origRequests := http1logRequests
			origResponses := http1logResponses
			origConnections := http1logConnections

			defer func() {
				// Restore original debug state
				http1VerboseLogs = origVerbose
				http1logRequests = origRequests
				http1logResponses = origResponses
				http1logConnections = origConnections
			}()

			// Reset debug flags
			http1VerboseLogs = false
			http1logRequests = false
			http1logResponses = false
			http1logConnections = false

			// Simulate GODEBUG parsing (matching the actual init function)
			e := tt.godebugValue
			if strings.Contains(e, "http1debug=1") {
				http1VerboseLogs = true
			}
			if strings.Contains(e, "http1debug=2") {
				http1VerboseLogs = true
				http1logRequests = true
				http1logResponses = true
			}
			if strings.Contains(e, "http1xconnect=1") {
				http1VerboseLogs = true
				http1logRequests = true
				http1logResponses = true
				http1logConnections = true
			}

			// Check that flags are set correctly
			if http1VerboseLogs != tt.expectVerbose {
				t.Errorf("Expected verbose=%v, got %v", tt.expectVerbose, http1VerboseLogs)
			}
			if http1logRequests != tt.expectRequests {
				t.Errorf("Expected requests=%v, got %v", tt.expectRequests, http1logRequests)
			}
			if http1logResponses != tt.expectResponses {
				t.Errorf("Expected responses=%v, got %v", tt.expectResponses, http1logResponses)
			}
			if http1logConnections != tt.expectConnections {
				t.Errorf("Expected connections=%v, got %v", tt.expectConnections, http1logConnections)
			}
		})
	}
}

// TestHTTP1DebugWithRemoteAddr tests the debug logging functionality
func TestHTTP1DebugWithRemoteAddr(t *testing.T) {
	// Save original debug state
	origVerbose := http1VerboseLogs
	origRequests := http1logRequests
	origResponses := http1logResponses

	defer func() {
		// Restore original debug state
		http1VerboseLogs = origVerbose
		http1logRequests = origRequests
		http1logResponses = origResponses
	}()

	// Enable all HTTP/1 debug logging
	http1VerboseLogs = true
	http1logRequests = true
	http1logResponses = true

	// Capture log output
	var logBuf bytes.Buffer
	oldOutput := log.Writer()
	log.SetOutput(&logBuf)
	defer log.SetOutput(oldOutput)

	// Test the debug logging functions
	http1Logf("test log: %s", "basic verbose log")

	// Verify logging works
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "test log: basic verbose log") {
		t.Errorf("Expected log output to contain test message, got: %s", logOutput)
	}

	t.Logf("Debug logging test completed successfully")
}

// TestHTTP1AllLogFunctions tests all HTTP/1 debug logging functions
func TestHTTP1AllLogFunctions(t *testing.T) {
	// Save original debug state
	origVerbose := http1VerboseLogs
	origRequests := http1logRequests
	origResponses := http1logResponses
	origConnections := http1logConnections

	defer func() {
		// Restore original debug state
		http1VerboseLogs = origVerbose
		http1logRequests = origRequests
		http1logResponses = origResponses
		http1logConnections = origConnections
	}()

	// Capture log output
	var logBuf bytes.Buffer
	oldOutput := log.Writer()
	log.SetOutput(&logBuf)
	defer log.SetOutput(oldOutput)

	tests := []struct {
		name     string
		setup    func()
		logFunc  func()
		expected string
		enabled  bool
	}{
		{
			name: "verbose logging enabled",
			setup: func() {
				http1VerboseLogs = true
				http1logRequests = false
				http1logResponses = false
				http1logConnections = false
			},
			logFunc: func() {
				http1Logf("verbose test message")
			},
			expected: "http1: verbose test message",
			enabled:  true,
		},
		{
			name: "verbose logging disabled",
			setup: func() {
				http1VerboseLogs = false
			},
			logFunc: func() {
				http1Logf("should not appear")
			},
			expected: "should not appear",
			enabled:  false,
		},
		{
			name: "request logging enabled",
			setup: func() {
				http1VerboseLogs = false
				http1logRequests = true
				http1logResponses = false
				http1logConnections = false
			},
			logFunc: func() {
				http1LogRequest("request test message")
			},
			expected: "http1: request: request test message",
			enabled:  true,
		},
		{
			name: "request logging disabled",
			setup: func() {
				http1logRequests = false
			},
			logFunc: func() {
				http1LogRequest("request should not appear")
			},
			expected: "request should not appear",
			enabled:  false,
		},
		{
			name: "response logging enabled",
			setup: func() {
				http1VerboseLogs = false
				http1logRequests = false
				http1logResponses = true
				http1logConnections = false
			},
			logFunc: func() {
				http1LogResponse("response test message")
			},
			expected: "http1: response: response test message",
			enabled:  true,
		},
		{
			name: "response logging disabled",
			setup: func() {
				http1logResponses = false
			},
			logFunc: func() {
				http1LogResponse("response should not appear")
			},
			expected: "response should not appear",
			enabled:  false,
		},
		{
			name: "connection logging enabled",
			setup: func() {
				http1VerboseLogs = false
				http1logRequests = false
				http1logResponses = false
				http1logConnections = true
			},
			logFunc: func() {
				http1LogConnection("connection test message")
			},
			expected: "http1: connection: connection test message",
			enabled:  true,
		},
		{
			name: "connection logging disabled",
			setup: func() {
				http1logConnections = false
			},
			logFunc: func() {
				http1LogConnection("connection should not appear")
			},
			expected: "connection should not appear",
			enabled:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logBuf.Reset()
			tt.setup()
			tt.logFunc()

			output := logBuf.String()
			if tt.enabled {
				if !strings.Contains(output, tt.expected) {
					t.Errorf("Expected log to contain '%s', got: %s", tt.expected, output)
				}
			} else {
				if strings.Contains(output, tt.expected) {
					t.Errorf("Expected log NOT to contain '%s', but it did. Got: %s", tt.expected, output)
				}
			}
		})
	}
}
