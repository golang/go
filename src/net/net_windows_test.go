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
	"regexp"
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

func netshSpeaksEnglish(t *testing.T) bool {
	out, err := runCmd("netsh", "help")
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Contains(out, []byte("The following commands are available:"))
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
	if !netshSpeaksEnglish(t) {
		t.Skip("English version of netsh required for this test")
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

func netshInterfaceIPv4ShowAddress(name string, netshOutput []byte) []string {
	// Address information is listed like:
	//
	//Configuration for interface "Local Area Connection"
	//    DHCP enabled:                         Yes
	//    IP Address:                           10.0.0.2
	//    Subnet Prefix:                        10.0.0.0/24 (mask 255.255.255.0)
	//    IP Address:                           10.0.0.3
	//    Subnet Prefix:                        10.0.0.0/24 (mask 255.255.255.0)
	//    Default Gateway:                      10.0.0.254
	//    Gateway Metric:                       0
	//    InterfaceMetric:                      10
	//
	//Configuration for interface "Loopback Pseudo-Interface 1"
	//    DHCP enabled:                         No
	//    IP Address:                           127.0.0.1
	//    Subnet Prefix:                        127.0.0.0/8 (mask 255.0.0.0)
	//    InterfaceMetric:                      50
	//
	addrs := make([]string, 0)
	var addr, subnetprefix string
	var processingOurInterface bool
	lines := bytes.Split(netshOutput, []byte{'\r', '\n'})
	for _, line := range lines {
		if !processingOurInterface {
			if !bytes.HasPrefix(line, []byte("Configuration for interface")) {
				continue
			}
			if !bytes.Contains(line, []byte(`"`+name+`"`)) {
				continue
			}
			processingOurInterface = true
			continue
		}
		if len(line) == 0 {
			break
		}
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
	return addrs
}

func netshInterfaceIPv6ShowAddress(name string, netshOutput []byte) []string {
	// Address information is listed like:
	//
	//Address ::1 Parameters
	//---------------------------------------------------------
	//Interface Luid     : Loopback Pseudo-Interface 1
	//Scope Id           : 0.0
	//Valid Lifetime     : infinite
	//Preferred Lifetime : infinite
	//DAD State          : Preferred
	//Address Type       : Other
	//Skip as Source     : false
	//
	//Address XXXX::XXXX:XXXX:XXXX:XXXX%11 Parameters
	//---------------------------------------------------------
	//Interface Luid     : Local Area Connection
	//Scope Id           : 0.11
	//Valid Lifetime     : infinite
	//Preferred Lifetime : infinite
	//DAD State          : Preferred
	//Address Type       : Other
	//Skip as Source     : false
	//

	// TODO: need to test ipv6 netmask too, but netsh does not outputs it
	var addr string
	addrs := make([]string, 0)
	lines := bytes.Split(netshOutput, []byte{'\r', '\n'})
	for _, line := range lines {
		if addr != "" {
			if len(line) == 0 {
				addr = ""
				continue
			}
			if string(line) != "Interface Luid     : "+name {
				continue
			}
			addrs = append(addrs, addr)
			addr = ""
			continue
		}
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

		// netsh can create IPv4-embedded IPv6 addresses, like fe80::5efe:192.168.140.1.
		// Convert these to all hexadecimal fe80::5efe:c0a8:8c01 for later string comparisons.
		ipv4Tail := regexp.MustCompile(`:\d+\.\d+\.\d+\.\d+$`)
		if ipv4Tail.Match(f[0]) {
			f[0] = []byte(ParseIP(string(f[0])).String())
		}

		addr = string(bytes.ToLower(bytes.TrimSpace(f[0])))
	}
	return addrs
}

func TestInterfaceAddrsWithNetsh(t *testing.T) {
	if isWindowsXP(t) {
		t.Skip("Windows XP netsh command does not provide required functionality")
	}
	if !netshSpeaksEnglish(t) {
		t.Skip("English version of netsh required for this test")
	}

	outIPV4, err := runCmd("netsh", "interface", "ipv4", "show", "address")
	if err != nil {
		t.Fatal(err)
	}
	outIPV6, err := runCmd("netsh", "interface", "ipv6", "show", "address", "level=verbose")
	if err != nil {
		t.Fatal(err)
	}

	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	for _, ifi := range ift {
		// Skip the interface if it's down.
		if (ifi.Flags & FlagUp) == 0 {
			continue
		}
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

		want := netshInterfaceIPv4ShowAddress(ifi.Name, outIPV4)
		wantIPv6 := netshInterfaceIPv6ShowAddress(ifi.Name, outIPV6)
		want = append(want, wantIPv6...)
		sort.Strings(want)

		if strings.Join(want, "/") != strings.Join(have, "/") {
			t.Errorf("%s: unexpected addresses list %q, want %q", ifi.Name, have, want)
		}
	}
}

// check that getmac exists as a powershell command, and that it
// speaks English.
func checkGetmac(t *testing.T) {
	out, err := runCmd("getmac", "/?")
	if err != nil {
		if strings.Contains(err.Error(), "term 'getmac' is not recognized as the name of a cmdlet") {
			t.Skipf("getmac not available")
		}
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte("network adapters on a system")) {
		t.Skipf("skipping test on non-English system")
	}
}

func TestInterfaceHardwareAddrWithGetmac(t *testing.T) {
	if isWindowsXP(t) {
		t.Skip("Windows XP does not have powershell command")
	}
	checkGetmac(t)

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
	//Physical Address: N/A
	//Transport Name:   Hardware not present
	//
	//Connection Name:  VMware Network Adapter VMnet8
	//Network Adapter:  VMware Virtual Ethernet Adapter for VMnet8
	//Physical Address: Disabled
	//Transport Name:   Disconnected
	//
	want := make(map[string]string)
	group := make(map[string]string) // name / values for single adapter
	getValue := func(name string) string {
		value, found := group[name]
		if !found {
			t.Fatalf("%q has no %q line in it", group, name)
		}
		if value == "" {
			t.Fatalf("%q has empty %q value", group, name)
		}
		return value
	}
	processGroup := func() {
		if len(group) == 0 {
			return
		}
		tname := strings.ToLower(getValue("Transport Name"))
		if tname == "n/a" {
			// skip these
			return
		}
		addr := strings.ToLower(getValue("Physical Address"))
		if addr == "disabled" || addr == "n/a" {
			// skip these
			return
		}
		addr = strings.Replace(addr, "-", ":", -1)
		cname := getValue("Connection Name")
		want[cname] = addr
		group = make(map[string]string)
	}
	lines := bytes.Split(out, []byte{'\r', '\n'})
	for _, line := range lines {
		if len(line) == 0 {
			processGroup()
			continue
		}
		i := bytes.IndexByte(line, ':')
		if i == -1 {
			t.Fatalf("line %q has no : in it", line)
		}
		group[string(line[:i])] = string(bytes.TrimSpace(line[i+1:]))
	}
	processGroup()

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
