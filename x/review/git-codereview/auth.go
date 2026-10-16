// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"syscall"

	"golang.org/x/term"
)

// authenticate handles authentication for the given URL.
// It first checks for a valid cookie, and only prompts for a password
// if no valid cookie is found.
func authenticate(u *url.URL) error {
	// First, try to authenticate using a cookie
	if err := authenticateWithCookie(u); err == nil {
		return nil
	}

	// No valid cookie found, prompt for password
	return authenticateWithPassword(u)
}

// authenticateWithCookie attempts to authenticate using a stored cookie.
// Returns nil if successful, or an error if cookie auth fails.
func authenticateWithCookie(u *url.URL) error {
	// Check if we have a stored cookie for this domain
	cookie, err := getStoredCookie(u)
	if err != nil {
		return fmt.Errorf("no stored cookie found: %v", err)
	}

	// Validate the cookie by making a test request
	req, err := http.NewRequest("GET", u.String(), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}
	req.AddCookie(cookie)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("cookie validation failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return nil
	}

	return fmt.Errorf("cookie rejected with status %d", resp.StatusCode)
}

// authenticateWithPassword prompts the user for a password and authenticates.
func authenticateWithPassword(u *url.URL) error {
	fmt.Fprintf(os.Stderr, "Password for %s: ", u.Host)
	password, err := readPassword()
	if err != nil {
		return fmt.Errorf("failed to read password: %v", err)
	}
	fmt.Fprintln(os.Stderr)

	// Use the password to authenticate
	return authenticateWithCredentials(u, password)
}

// readPassword reads a password from stdin without echoing.
func readPassword() (string, error) {
	password, err := term.ReadPassword(int(syscall.Stdin))
	if err != nil {
		return "", err
	}
	return string(password), nil
}

// authenticateWithCredentials authenticates using the provided password.
func authenticateWithCredentials(u *url.URL, password string) error {
	// Create authentication request with password
	form := url.Values{}
	form.Set("password", password)
	form.Set("email", getEmailForHost(u.Host))

	req, err := http.NewRequest("POST", u.String()+"/login", strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("failed to create auth request: %v", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("authentication request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("authentication failed with status %d", resp.StatusCode)
	}

	// Store the cookie for future use
	if err := storeCookie(u, resp.Cookies()); err != nil {
		return fmt.Errorf("failed to store cookie: %v", err)
	}

	return nil
}

// getStoredCookie retrieves a stored cookie for the given URL.
func getStoredCookie(u *url.URL) (*http.Cookie, error) {
	// Implementation depends on how cookies are stored
	// This is a placeholder - actual implementation would read from a cookie jar
	return nil, fmt.Errorf("cookie storage not implemented")
}

// storeCookie stores cookies for future authentication.
func storeCookie(u *url.URL, cookies []*http.Cookie) error {
	// Implementation depends on how cookies are stored
	// This is a placeholder - actual implementation would write to a cookie jar
	return nil
}

// getEmailForHost returns the email address associated with the given host.
func getEmailForHost(host string) string {
	// This would typically read from git config or a config file
	return ""
}