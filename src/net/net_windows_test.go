// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"sort"
	"strings"
	"syscall"
	"testing"
	"time"
)

func toErrno(err error) (syscall.Errno, bool) {
	operr, ok := err.(*OpError)
	if !ok {
		return 0, false
	}
	syserr, ok := operr.Err.(*os.SyscallError)
	if !ok {
		return 0, false
	}
	errno, ok := syserr.Err.(syscall.Errno)
	if !ok {
		return 0, false
	}
	return errno, true
}

// TestAcceptIgnoreSomeErrors tests that windows TCPListener.AcceptTCP
// handles broken connections. It verifies that broken connections do
// not affect future connections.
func TestAcceptIgnoreSomeErrors(t *testing.T) {
	recv := func(ln Listener, ignoreSomeReadErrors bool) (string, error) {
		c, err := ln.Accept()
		if err != nil {
			// Display windows errno in error message.
			errno, ok := toErrno(err)
			if !ok {
				return "", err
			}
			return "", fmt.Errorf("%v (windows errno=%d)", err, errno)
		}
		defer c.Close()

		b := make([]byte, 100)
		n, err := c.Read(b)
		if err == nil || err == io.EOF {
			return string(b[:n]), nil
		}
		errno, ok := toErrno(err)
		if ok && ignoreSomeReadErrors && (errno == syscall.ERROR_NETNAME_DELETED || errno == syscall.WSAECONNRESET) {
			return "", nil
		}
		return "", err
	}

	send := func(addr string, data string) error {
		c, err := Dial("tcp", addr)
		if err != nil {
			return err
		}
		defer c.Close()

		b := []byte(data)
		n, err := c.Write(b)
		if err != nil {
			return err
		}
		if n != len(b) {
			return fmt.Errorf(`Only %d chars of string "%s" sent`, n, data)
		}
		return nil
	}

	if envaddr := os.Getenv("GOTEST_DIAL_ADDR"); envaddr != "" {
		// In child process.
		c, err := Dial("tcp", envaddr)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Printf("sleeping\n")
		time.Sleep(time.Minute) // process will be killed here
		c.Close()
	}

	ln, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	// Start child process that connects to our listener.
	cmd := exec.Command(os.Args[0], "-test.run=TestAcceptIgnoreSomeErrors")
	cmd.Env = append(os.Environ(), "GOTEST_DIAL_ADDR="+ln.Addr().String())
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("cmd.StdoutPipe failed: %v", err)
	}
	err = cmd.Start()
	if err != nil {
		t.Fatalf("cmd.Start failed: %v\n", err)
	}
	outReader := bufio.NewReader(stdout)
	for {
		s, err := outReader.ReadString('\n')
		if err != nil {
			t.Fatalf("reading stdout failed: %v", err)
		}
		if s == "sleeping\n" {
			break
		}
	}
	defer cmd.Wait() // ignore error - we know it is getting killed

	const alittle = 100 * time.Millisecond
	time.Sleep(alittle)
	cmd.Process.Kill() // the only way to trigger the errors
	time.Sleep(alittle)

	// Send second connection data (with delay in a separate goroutine).
	result := make(chan error)
	go func() {
		time.Sleep(alittle)
		err := send(ln.Addr().String(), "abc")
		if err != nil {
			result <- err
		}
		result <- nil
	}()
	defer func() {
		err := <-result
		if err != nil {
			t.Fatalf("send failed: %v", err)
		}
	}()

	// Receive first or second connection.
	s, err := recv(ln, true)
	if err != nil {
		t.Fatalf("recv failed: %v", err)
	}
	switch s {
	case "":
		// First connection data is received, let's get second connection data.
	case "abc":
		// First connection is lost forever, but that is ok.
		return
	default:
		t.Fatalf(`"%s" received from recv, but "" or "abc" expected`, s)
	}

	// Get second connection data.
	s, err = recv(ln, false)
	if err != nil {
		t.Fatalf("recv failed: %v", err)
	}
	if s != "abc" {
		t.Fatalf(`"%s" received from recv, but "abc" expected`, s)
	}
}

func isWindowsXP(t *testing.T) bool {
	v, err := syscall.GetVersion()
	if err != nil {
		t.Fatalf("GetVersion failed: %v", err)
	}
	major := byte(v)
	return major < 6
}

var (
	modkernel32 = syscall.NewLazyDLL("kernel32.dll")
	procGetACP  = modkernel32.NewProc("GetACP")
)

func isEnglishOS(t *testing.T) bool {
	const windows_1252 = 1252 // ANSI Latin 1; Western European (Windows)
	r0, _, _ := syscall.Syscall(procGetACP.Addr(), 0, 0, 0, 0)
	acp := uint32(r0)
	return acp == windows_1252
}

func runCmd(args ...string) ([]byte, error) {
	removeUTF8BOM := func(b []byte) []byte {
		if len(b) >= 3 && b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF {
			return b[3:]
		}
		return b
	}
	f, err := ioutil.TempFile("", "netcmd")
	if err != nil {
		return nil, err
	}
	f.Close()
	defer os.Remove(f.Name())
	cmd := fmt.Sprintf(`%s | Out-File "%s" -encoding UTF8`, strings.Join(args, " "), f.Name())
	out, err := exec.Command("powershell", "-Command", cmd).CombinedOutput()
	if err != nil {
		if len(out) != 0 {
			return nil, fmt.Errorf("%s failed: %v: %q", args[0], err, string(removeUTF8BOM(out)))
		}
		var err2 error
		out, err2 = ioutil.ReadFile(f.Name())
		if err2 != nil {
			return nil, err2
		}
		if len(out) != 0 {
			return nil, fmt.Errorf("%s failed: %v: %q", args[0], err, string(removeUTF8BOM(out)))
		}
		return nil, fmt.Errorf("%s failed: %v", args[0], err)
	}
	out, err = ioutil.ReadFile(f.Name())
	if err != nil {
		return nil, err
	}
	return removeUTF8BOM(out), nil
}

func netshInterfaceIPShowInterface(ipver string, ifaces map[string]bool) error {
	out, err := runCmd("netsh", "interface", ipver, "show", "interface", "level=verbose")
	if err != nil {
		return err
	}
	// interface information is listed like:
	//
	//Interface Local Area Connection Parameters
	//----------------------------------------------
	//IfLuid                             : ethernet_6
	//IfIndex                            : 11
	//State                              : connected
	//Metric                             : 10
	//...
	var name string
	lines := bytes.Split(out, []byte{'\r', '\n'})
	for _, line := range lines {
		if bytes.HasPrefix(line, []byte("Interface ")) && bytes.HasSuffix(line, []byte(" Parameters")) {
			f := line[len("Interface "):]
			f = f[:len(f)-len(" Parameters")]
			name = string(f)
			continue
		}
		var isup bool
		switch string(line) {
		case "State                              : connected":
			isup = true
		case "State                              : disconnected":
			isup = false
		default:
			continue
		}
		if name != "" {
			if v, ok := ifaces[name]; ok && v != isup {
				return fmt.Errorf("%s:%s isup=%v: ipv4 and ipv6 report different interface state", ipver, name, isup)
			}
			ifaces[name] = isup
			name = ""
		}
	}
	return nil
}

func TestInterfacesWithNetsh(t *testing.T) {
	if isWindowsXP(t) {
		t.Skip("Windows XP netsh command does not provide required functionality")
	}
	if !isEnglishOS(t) {
		t.Skip("English version of OS required for this test")
	}

	toString := func(name string, isup bool) string {
		if isup {
			return name + ":up"
		}
		return name + ":down"
	}

	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	have := make([]string, 0)
	for _, ifi := range ift {
		have = append(have, toString(ifi.Name, ifi.Flags&FlagUp != 0))
	}
	sort.Strings(have)

	ifaces := make(map[string]bool)
	err = netshInterfaceIPShowInterface("ipv6", ifaces)
	if err != nil {
		t.Fatal(err)
	}
	err = netshInterfaceIPShowInterface("ipv4", ifaces)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]string, 0)
	for name, isup := range ifaces {
		want = append(want, toString(name, isup))
	}
	sort.Strings(want)

	if strings.Join(want, "/") != strings.Join(have, "/") {
		t.Fatalf("unexpected interface list %q, want %q", have, want)
	}
}

func netshInterfaceIPv4ShowAddress(name string) ([]string, error) {
	out, err := runCmd("netsh", "interface", "ipv4", "show", "address", "name=\""+name+"\"")
	if err != nil {
		return nil, err
	}
	// adress information is listed like:
	//    IP Address:                           10.0.0.2
	//    Subnet Prefix:                        10.0.0.0/24 (mask 255.255.255.0)
	//    IP Address:                           10.0.0.3
	//    Subnet Prefix:                        10.0.0.0/24 (mask 255.255.255.0)
	addrs := make([]string, 0)
	var addr, subnetprefix string
	lines := bytes.Split(out, []byte{'\r', '\n'})
	for _, line := range lines {
		if bytes.Contains(line, []byte("Subnet Prefix:")) {
			f := bytes.Split(line, []byte{':'})
			if len(f) == 2 {
				f = bytes.Split(f[1], []byte{'('})
				if len(f) == 2 {
					f = bytes.Split(f[0], []byte{'/'})
					if len(f) == 2 {
						subnetprefix = string(bytes.TrimSpace(f[1]))
						if addr != "" && subnetprefix != "" {
							addrs = append(addrs, addr+"/"+subnetprefix)
						}
					}
				}
			}
		}
		addr = ""
		if bytes.Contains(line, []byte("IP Address:")) {
			f := bytes.Split(line, []byte{':'})
			if len(f) == 2 {
				addr = string(bytes.TrimSpace(f[1]))
			}
		}
	}
	return addrs, nil
}

func netshInterfaceIPv6ShowAddress(name string) ([]string, error) {
	// TODO: need to test ipv6 netmask too, but netsh does not outputs it
	out, err := runCmd("netsh", "interface", "ipv6", "show", "address", "interface=\""+name+"\"")
	if err != nil {
		return nil, err
	}
	addrs := make([]string, 0)
	lines := bytes.Split(out, []byte{'\r', '\n'})
	for _, line := range lines {
		if !bytes.HasPrefix(line, []byte("Address")) {
			continue
		}
		if !bytes.HasSuffix(line, []byte("Parameters")) {
			continue
		}
		f := bytes.Split(line, []byte{' '})
		if len(f) != 3 {
			continue
		}
		// remove scope ID if present
		f = bytes.Split(f[1], []byte{'%'})
		addrs = append(addrs, string(bytes.ToLower(bytes.TrimSpace(f[0]))))
	}
	return addrs, nil
}

func TestInterfaceAddrsWithNetsh(t *testing.T) {
	if isWindowsXP(t) {
		t.Skip("Windows XP netsh command does not provide required functionality")
	}
	if !isEnglishOS(t) {
		t.Skip("English version of OS required for this test")
	}
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	for _, ifi := range ift {
		have := make([]string, 0)
		addrs, err := ifi.Addrs()
		if err != nil {
			t.Fatal(err)
		}
		for _, addr := range addrs {
			switch addr := addr.(type) {
			case *IPNet:
				if addr.IP.To4() != nil {
					have = append(have, addr.String())
				}
				if addr.IP.To16() != nil && addr.IP.To4() == nil {
					// netsh does not output netmask for ipv6, so ignore ipv6 mask
					have = append(have, addr.IP.String())
				}
			case *IPAddr:
				if addr.IP.To4() != nil {
					have = append(have, addr.String())
				}
				if addr.IP.To16() != nil && addr.IP.To4() == nil {
					// netsh does not output netmask for ipv6, so ignore ipv6 mask
					have = append(have, addr.IP.String())
				}
			}
		}
		sort.Strings(have)

		want, err := netshInterfaceIPv4ShowAddress(ifi.Name)
		if err != nil {
			t.Fatal(err)
		}
		wantIPv6, err := netshInterfaceIPv6ShowAddress(ifi.Name)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, wantIPv6...)
		sort.Strings(want)

		if strings.Join(want, "/") != strings.Join(have, "/") {
			t.Errorf("%s: unexpected addresses list %q, want %q", ifi.Name, have, want)
		}
	}
}

func TestInterfaceHardwareAddrWithGetmac(t *testing.T) {
	if isWindowsXP(t) {
		t.Skip("Windows XP does not have powershell command")
	}
	if !isEnglishOS(t) {
		t.Skip("English version of OS required for this test")
	}

	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	have := make(map[string]string)
	for _, ifi := range ift {
		if ifi.Flags&FlagLoopback != 0 {
			// no MAC address for loopback interfaces
			continue
		}
		have[ifi.Name] = ifi.HardwareAddr.String()
	}

	out, err := runCmd("getmac", "/fo", "list", "/v")
	if err != nil {
		t.Fatal(err)
	}
	// getmac output looks like:
	//
	//Connection Name:  Local Area Connection
	//Network Adapter:  Intel Gigabit Network Connection
	//Physical Address: XX-XX-XX-XX-XX-XX
	//Transport Name:   \Device\Tcpip_{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}
	//
	//Connection Name:  Wireless Network Connection
	//Network Adapter:  Wireles WLAN Card
	//Physical Address: XX-XX-XX-XX-XX-XX
	//Transport Name:   Media disconnected
	//
	//Connection Name:  Bluetooth Network Connection
	//Network Adapter:  Bluetooth Device (Personal Area Network)
	//Physical Address: XX-XX-XX-XX-XX-XX
	//Transport Name:   Media disconnected
	//
	want := make(map[string]string)
	var name string
	lines := bytes.Split(out, []byte{'\r', '\n'})
	for _, line := range lines {
		if bytes.Contains(line, []byte("Connection Name:")) {
			f := bytes.Split(line, []byte{':'})
			if len(f) != 2 {
				t.Fatal("unexpected \"Connection Name\" line: %q", line)
			}
			name = string(bytes.TrimSpace(f[1]))
			if name == "" {
				t.Fatal("empty name on \"Connection Name\" line: %q", line)
			}
		}
		if bytes.Contains(line, []byte("Physical Address:")) {
			if name == "" {
				t.Fatal("no matching name found: %q", string(out))
			}
			f := bytes.Split(line, []byte{':'})
			if len(f) != 2 {
				t.Fatal("unexpected \"Physical Address\" line: %q", line)
			}
			addr := string(bytes.ToLower(bytes.TrimSpace(f[1])))
			if addr == "" {
				t.Fatal("empty address on \"Physical Address\" line: %q", line)
			}
			addr = strings.Replace(addr, "-", ":", -1)
			want[name] = addr
			name = ""
		}
	}

	for name, wantAddr := range want {
		haveAddr, ok := have[name]
		if !ok {
			t.Errorf("getmac lists %q, but it could not be found among Go interfaces %v", name, have)
			continue
		}
		if haveAddr != wantAddr {
			t.Errorf("unexpected MAC address for %q - %v, want %v", name, haveAddr, wantAddr)
			continue
		}
	}
}
