// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgroup_test

import (
	"fmt"
	"internal/runtime/cgroup"
	"io"
	"strings"
	"testing"
)

func TestParseV1Number(t *testing.T) {
	tests := []struct {
		name     string
		contents string
		want     int64
		wantErr  bool
	}{
		{
			name:     "disabled",
			contents: "-1\n",
			want:     -1,
		},
		{
			name:     "500000",
			contents: "500000\n",
			want:     500000,
		},
		{
			name:     "MaxInt64",
			contents: "9223372036854775807\n",
			want:     9223372036854775807,
		},
		{
			name:     "missing-newline",
			contents: "500000",
			wantErr:  true,
		},
		{
			name:     "not-a-number",
			contents: "123max\n",
			wantErr:  true,
		},
		{
			name:     "v2",
			contents: "1000 5000\n",
			wantErr:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := cgroup.ParseV1Number([]byte(tc.contents))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseV1Number got err nil want non-nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("parseV1Number got err %v want nil", err)
			}

			if got != tc.want {
				t.Errorf("parseV1Number got %d want %d", got, tc.want)
			}
		})
	}
}

func TestParseV2Limit(t *testing.T) {
	tests := []struct {
		name     string
		contents string
		want     float64
		wantOK   bool
		wantErr  bool
	}{
		{
			name:     "disabled",
			contents: "max 100000\n",
			wantOK:   false,
		},
		{
			name:     "5",
			contents: "500000 100000\n",
			want:     5,
			wantOK:   true,
		},
		{
			name:     "0.5",
			contents: "50000 100000\n",
			want:     0.5,
			wantOK:   true,
		},
		{
			name:     "2.5",
			contents: "250000 100000\n",
			want:     2.5,
			wantOK:   true,
		},
		{
			name:     "MaxInt64",
			contents: "9223372036854775807 9223372036854775807\n",
			want:     1,
			wantOK:   true,
		},
		{
			name:     "missing-newline",
			contents: "500000 100000",
			wantErr:  true,
		},
		{
			name:     "v1",
			contents: "500000\n",
			wantErr:  true,
		},
		{
			name:     "quota-not-a-number",
			contents: "500000us 100000\n",
			wantErr:  true,
		},
		{
			name:     "period-not-a-number",
			contents: "500000 100000us\n",
			wantErr:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, gotOK, err := cgroup.ParseV2Limit([]byte(tc.contents))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseV1Limit got err nil want non-nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("parseV2Limit got err %v want nil", err)
			}

			if gotOK != tc.wantOK {
				t.Errorf("parseV2Limit got ok %v want %v", gotOK, tc.wantOK)
			}

			if tc.wantOK && got != tc.want {
				t.Errorf("parseV2Limit got %f want %f", got, tc.want)
			}
		})
	}
}

func readString(contents string) func(fd int, b []byte) (int, uintptr) {
	r := strings.NewReader(contents)
	return func(fd int, b []byte) (int, uintptr) {
		n, err := r.Read(b)
		if err != nil && err != io.EOF {
			const dummyErrno = 42
			return n, dummyErrno
		}
		return n, 0
	}
}

func TestParseCPUCgroup(t *testing.T) {
	veryLongPathName := strings.Repeat("a", cgroup.PathSize+10)
	evenLongerPathName := strings.Repeat("a", cgroup.ParseSize+10)

	tests := []struct {
		name     string
		contents string
		want     string
		wantVer  cgroup.Version
		wantErr  bool
	}{
		{
			name:     "empty",
			contents: "",
			wantErr:  true,
		},
		{
			name:     "too-long",
			contents: "0::/" + veryLongPathName + "\n",
			wantErr:  true,
		},
		{
			name:     "too-long-line",
			contents: "0::/" + evenLongerPathName + "\n",
			wantErr:  true,
		},
		{
			name: "v1",
			contents: `2:cpu,cpuacct:/a/b/cpu
1:blkio:/a/b/blkio
`,
			want:    "/a/b/cpu",
			wantVer: cgroup.V1,
		},
		{
			name:     "v2",
			contents: "0::/a/b/c\n",
			want:     "/a/b/c",
			wantVer:  cgroup.V2,
		},
		{
			name: "mixed",
			contents: `2:cpu,cpuacct:/a/b/cpu
1:blkio:/a/b/blkio
0::/a/b/v2
`,
			want:    "/a/b/cpu",
			wantVer: cgroup.V1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got [cgroup.PathSize]byte
			var scratch [cgroup.ParseSize]byte
			n, gotVer, err := cgroup.ParseCPUCgroup(0, readString(tc.contents), got[:], scratch[:])
			if (err != nil) != tc.wantErr {
				t.Fatalf("parseCPURelativePath got err %v want %v", err, tc.wantErr)
			}

			if gotVer != tc.wantVer {
				t.Errorf("parseCPURelativePath got cgroup version %d want %d", gotVer, tc.wantVer)
			}

			if string(got[:n]) != tc.want {
				t.Errorf("parseCPURelativePath got %q want %q", string(got[:n]), tc.want)
			}
		})
	}
}

func TestParseCPUCgroupMalformed(t *testing.T) {
	for _, contents := range []string{
		"\n",
		"0\n",
		"0:\n",
		"0::\n",
		"0::a\n",
	} {
		t.Run("", func(t *testing.T) {
			var got [cgroup.PathSize]byte
			var scratch [cgroup.ParseSize]byte
			n, v, err := cgroup.ParseCPUCgroup(0, readString(contents), got[:], scratch[:])
			if err != cgroup.ErrMalformedFile {
				t.Errorf("ParseCPUCgroup got %q (v%d), %v, want ErrMalformedFile", string(got[:n]), v, err)
			}
		})
	}
}

func TestContainsCPU(t *testing.T) {
	tests := []struct {
		in   string
		want bool
	}{
		{
			in:   "",
			want: false,
		},
		{
			in:   ",",
			want: false,
		},
		{
			in:   "cpu",
			want: true,
		},
		{
			in:   "memory,cpu",
			want: true,
		},
		{
			in:   "cpu,memory",
			want: true,
		},
		{
			in:   "memory,cpu,block",
			want: true,
		},
		{
			in:   "memory,cpuacct,block",
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			got := cgroup.ContainsCPU([]byte(tc.in))
			if got != tc.want {
				t.Errorf("containsCPU(%q) got %v want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestParseCPUMount(t *testing.T) {
	// Used for v2-longline. We want an overlayfs mount to have an option
	// so long that the entire line can't possibly fit in the scratch
	// buffer.
	const lowerPath = "/so/many/overlay/layers"
	overlayLongLowerDir := lowerPath
	for i := 0; len(overlayLongLowerDir) < cgroup.ScratchSize; i++ {
		overlayLongLowerDir += fmt.Sprintf(":%s%d", lowerPath, i)
	}

	var longPath [4090]byte
	for i := range longPath {
		longPath[i] = byte(i)
	}
	escapedLongPath := escapePath(string(longPath[:]))
	if len(escapedLongPath) <= cgroup.PathSize {
		// ensure we actually support over PathSize long escaped path
		t.Fatalf("escapedLongPath is too short to test")
	}

	tests := []struct {
		name     string
		contents string
		cgroup   string
		version  cgroup.Version
		want     string
		wantErr  bool
	}{
		{
			name:     "empty",
			contents: "",
			wantErr:  true,
		},
		{
			name:     "invalid-root",
			contents: "56 22 0:40 /\\1 /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n",
			cgroup:   "/",
			version:  cgroup.V1,
			wantErr:  true,
		},
		{
			name:     "invalid-mount",
			contents: "56 22 0:40 / /sys/fs/cgroup/\\1 rw - cgroup cgroup rw,cpu,cpuacct\n",
			cgroup:   "/",
			version:  cgroup.V1,
			wantErr:  true,
		},
		{
			name: "v1",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
49 22 0:37 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory
54 22 0:38 / /sys/fs/cgroup/io rw - cgroup cgroup rw,io
56 22 0:40 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct
58 22 0:42 / /sys/fs/cgroup/net rw - cgroup cgroup rw,net
59 22 0:43 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset
`,
			cgroup:  "/",
			version: cgroup.V1,
			want:    "/sys/fs/cgroup/cpu",
		},
		{
			name: "v2",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
`,
			cgroup:  "/",
			version: cgroup.V2,
			want:    "/sys/fs/cgroup",
		},
		{
			name: "mixed",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
49 22 0:37 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory
54 22 0:38 / /sys/fs/cgroup/io rw - cgroup cgroup rw,io
56 22 0:40 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct
58 22 0:42 / /sys/fs/cgroup/net rw - cgroup cgroup rw,net
59 22 0:43 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset
`,
			cgroup:  "/",
			version: cgroup.V1,
			want:    "/sys/fs/cgroup/cpu",
		},
		{
			name: "mixed-choose-v2",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
49 22 0:37 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory
54 22 0:38 / /sys/fs/cgroup/io rw - cgroup cgroup rw,io
56 22 0:40 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct
58 22 0:42 / /sys/fs/cgroup/net rw - cgroup cgroup rw,net
59 22 0:43 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset
`,
			cgroup:  "/",
			version: cgroup.V2,
			want:    "/sys/fs/cgroup",
		},
		{
			name: "v2-escaped",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 / /sys/fs/cgroup/tab\011tab rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
`,
			cgroup:  "/",
			version: cgroup.V2,
			want:    `/sys/fs/cgroup/tab	tab`,
		},
		{
			// Overly long line on a different mount doesn't matter.
			name: "v2-longline",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
262 31 0:72 / /tmp/overlay2/0143e063b02f4801de9c847ad1c5ddc21fd2ead00653064d0c72ea967b248870/merged rw,relatime shared:729 - overlay overlay rw,lowerdir=` + overlayLongLowerDir + `,upperdir=/tmp/diff,workdir=/tmp/work
25 21 0:22 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
`,
			cgroup:  "/",
			version: cgroup.V2,
			want:    "/sys/fs/cgroup",
		},
		{
			name: "long-escaped-path",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 / /sys/` + escapedLongPath + ` rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
`,
			cgroup:  "/",
			version: cgroup.V2,
			want:    "/sys/" + string(longPath[:]),
		},
		{
			name: "too-long-escaped-path",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 / /sys/` + escapedLongPath + ` rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
`,
			cgroup:  "/container", // compared to above, this makes the path too long
			version: cgroup.V2,
			wantErr: true,
		},
		{
			name: "non-root_mount",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 /sand /unrelated/cgroup1 rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
25 21 0:22 /stone /unrelated/cgroup2 rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
25 21 0:22 /sandbox/container/group /sys/fs/cgroup/mygroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
25 21 0:22 /sandbox /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
25 21 0:22 / /ignored/second/match rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
`,
			cgroup:  "/sandbox/container",
			version: cgroup.V2,
			want:    "/sys/fs/cgroup/container",
		},
		{
			name: "v2-escaped-root",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 /tab\011tab /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
`,
			cgroup:  "/tab	tab/container",
			version: cgroup.V2,
			want:    `/sys/fs/cgroup/container`,
		},
		{
			name: "non-root_cgroup",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 / /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
`,
			cgroup:  "/sandbox/container",
			version: cgroup.V2,
			want:    "/sys/fs/cgroup/sandbox/container",
		},
		{
			name: "mixed_non-root",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
25 21 0:22 /sandbox /sys/fs/cgroup rw,nosuid,nodev,noexec - cgroup2 cgroup2 rw
49 22 0:37 /sandbox /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory
54 22 0:38 /sandbox /sys/fs/cgroup/io rw - cgroup cgroup rw,io
56 22 0:40 /sand /unrelated/cgroup1 rw - cgroup cgroup rw,cpu,cpuacct
56 22 0:40 /stone /unrelated/cgroup2 rw - cgroup cgroup rw,cpu,cpuacct
56 22 0:40 /sandbox /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct
56 22 0:40 /sandbox/container/group /sys/fs/cgroup/cpu/mygroup rw - cgroup cgroup rw,cpu,cpuacct
56 22 0:40 / /ignored/second/match rw - cgroup cgroup rw,cpu,cpuacct
58 22 0:42 /sandbox /sys/fs/cgroup/net rw - cgroup cgroup rw,net
59 22 0:43 /sandbox /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset
`,
			cgroup:  "/sandbox/container",
			version: cgroup.V1,
			want:    "/sys/fs/cgroup/cpu/container",
		},
		{
			// to see an example of this, for a PID in a cgroup namespace, run:
			// nsenter -t <PID> -C -- cat /proc/self/cgroup
			// nsenter -t <PID> -C -- grep cgroup /proc/self/mountinfo
			// /mnt can be generated with `mount --bind /sys/fs/cgroup/kubepods.slice /mnt`,
			// assuming PID is in cgroup /kubepods.slice
			name: "out_of_namespace",
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
1243 61 0:26 /../../.. /mnt rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw
29 22 0:26 /../../../.. /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw`,
			cgroup:  "/../../../../init.scope",
			version: cgroup.V2,
			want:    "/sys/fs/cgroup/init.scope",
		},
		{
			name: "out_of_namespace-root", // the process is directly in the root cgroup
			contents: `22 1 8:1 / / rw,relatime - ext4 /dev/root rw
20 22 0:19 / /proc rw,nosuid,nodev,noexec - proc proc rw
21 22 0:20 / /sys rw,nosuid,nodev,noexec - sysfs sysfs rw
1243 61 0:26 /../../.. /mnt rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw
29 22 0:26 /../../../.. /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw`,
			cgroup:  "/../../../..",
			version: cgroup.V2,
			want:    "/sys/fs/cgroup",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got [cgroup.PathSize]byte
			var scratch [cgroup.ParseSize]byte
			n := copy(got[:], tc.cgroup)
			n, err := cgroup.ParseCPUMount(0, readString(tc.contents), got[:],
				got[:n], tc.version, scratch[:])
			if (err != nil) != tc.wantErr {
				t.Fatalf("parseCPUMount got err %v want %v", err, tc.wantErr)
			}

			if string(got[:n]) != tc.want {
				t.Errorf("parseCPUMount got %q want %q", string(got[:n]), tc.want)
			}
		})
	}
}

func TestParseCPUMountMalformed(t *testing.T) {
	for _, contents := range []string{
		"\n",
		"22\n",
		"22 1 8:1\n",
		"22 1 8:1 /\n",
		"22 1 8:1 / /cgroup\n",
		"22 1 8:1 / /cgroup rw\n",
		"22 1 8:1 / /cgroup rw -\n",
		"22 1 8:1 / /cgroup rw - \n",
		"22 1 8:1 / /cgroup rw - cgroup\n",
		"22 1 8:1 / /cgroup rw - cgroup cgroup\n",
		"22 1 8:1 a /cgroup rw - cgroup cgroup cpu\n",
	} {
		t.Run("", func(t *testing.T) {
			var got [cgroup.PathSize]byte
			var scratch [cgroup.ParseSize]byte
			n, err := cgroup.ParseCPUMount(0, readString(contents), got[:], []byte("/"), cgroup.V1, scratch[:])
			if err != cgroup.ErrMalformedFile {
				t.Errorf("parseCPUMount got %q, %v, want ErrMalformedFile", string(got[:n]), err)
			}
		})
	}
}

// escapePath performs escaping equivalent to Linux's show_path.
//
// That is, '\', ' ', '\t', and '\n' are converted to octal escape sequences,
// like '\040' for space.
func escapePath(s string) string {
	out := make([]byte, 0, len(s))
	for _, c := range []byte(s) {
		switch c {
		case '\\', ' ', '\t', '\n':
			out = fmt.Appendf(out, "\\%03o", c)
		default:
			out = append(out, c)
		}
	}
	return string(out)
}

func TestEscapePath(t *testing.T) {
	tests := []struct {
		name      string
		unescaped string
		escaped   string
	}{
		{
			name:      "boring",
			unescaped: `/a/b/c`,
			escaped:   `/a/b/c`,
		},
		{
			name:      "space",
			unescaped: `/a/b b/c`,
			escaped:   `/a/b\040b/c`,
		},
		{
			name:      "tab",
			unescaped: `/a/b	b/c`,
			escaped:   `/a/b\011b/c`,
		},
		{
			name: "newline",
			unescaped: `/a/b
b/c`,
			escaped: `/a/b\012b/c`,
		},
		{
			name:      "slash",
			unescaped: `/a/b\b/c`,
			escaped:   `/a/b\134b/c`,
		},
		{
			name:      "beginning",
			unescaped: `\b/c`,
			escaped:   `\134b/c`,
		},
		{
			name:      "ending",
			unescaped: `/a/\`,
			escaped:   `/a/\134`,
		},
		{
			name:      "non-utf8",
			unescaped: "/a/b\xff\x20/c",
			escaped:   "/a/b\xff\\040/c",
		},
	}

	t.Run("escapePath", func(t *testing.T) {
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := escapePath(tc.unescaped)
				if got != tc.escaped {
					t.Errorf("escapePath got %q want %q", got, tc.escaped)
				}
			})
		}
	})

	t.Run("unescapePath", func(t *testing.T) {
		for _, tc := range tests {
			runTest := func(in, out []byte) {
				n, err := cgroup.UnescapePath(out, in)
				if err != nil {
					t.Errorf("unescapePath got err %v want nil", err)
				}
				got := string(out[:n])
				if got != tc.unescaped {
					t.Errorf("unescapePath got %q want %q", got, tc.escaped)
				}
			}
			t.Run(tc.name, func(t *testing.T) {
				in := []byte(tc.escaped)
				out := make([]byte, len(in))
				runTest(in, out)
			})
			t.Run("inplace/"+tc.name, func(t *testing.T) {
				in := []byte(tc.escaped)
				runTest(in, in)
			})
		}
	})
}
